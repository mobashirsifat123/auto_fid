"""Concurrent, provenance-complete study-search pipeline."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from autoresearch.domain.models import (
    DownloadRecord,
    DownloadStatus,
    PaperRecord,
    SearchCandidate,
    SeedPaper,
    content_checksum,
)
from autoresearch.providers.redaction import redact_text
from autoresearch.storage.repository import ResearchRepository

from .deduplication import SearchProvenance, SourcedCandidate, deduplicate, deduplication_key, normalized_title
from .metadata_sources import MetadataPaper, MetadataSource, SourceSearchResult, metadata_payload
from .pdf_resolver import PdfResolver
from .ranking import CandidateScore, rank_candidates


@dataclass(frozen=True)
class SearchStrategy:
    name: str
    query: str


@dataclass(frozen=True)
class DiscoveryResult:
    project_id: str
    run_id: str
    requested: int
    papers: tuple[PaperRecord, ...]
    downloads: tuple[DownloadRecord, ...]
    source_errors: tuple[str, ...]
    raw_candidate_count: int = 0
    unique_candidate_count: int = 0
    duplicate_count: int = 0
    seed_excluded_count: int = 0

    @property
    def downloaded_count(self) -> int:
        return sum(record.status is DownloadStatus.DOWNLOADED for record in self.downloads)


class StudySearchPipeline:
    def __init__(self, repository: ResearchRepository, sources: Sequence[MetadataSource], pdf_resolver: PdfResolver, download_root: Path, *, search_concurrency: int = 8, download_concurrency: int = 4, citation_expansion: bool = True, citation_anchors: int = 5, citation_neighbors: int = 10) -> None:
        if not sources:
            raise ValueError("At least one metadata source is required.")
        self.repository = repository
        self.sources = tuple(sources)
        self.pdf_resolver = pdf_resolver
        self.download_root = download_root
        self.search_semaphore = asyncio.Semaphore(max(1, search_concurrency))
        self.download_semaphore = asyncio.Semaphore(max(1, download_concurrency))
        self.citation_expansion = citation_expansion
        self.citation_anchors = max(0, citation_anchors)
        self.citation_neighbors = max(1, citation_neighbors)

    async def discover(self, *, project_id: str, run_id: str, seed: SeedPaper, strategies: Sequence[SearchStrategy], limit: int, since_year: int | None = None) -> DiscoveryResult:
        if not 1 <= limit <= 100:
            raise ValueError("Discovery limit must be between 1 and 100.")
        if not strategies:
            raise ValueError("At least one search strategy is required.")
        tasks = [self._search(source, strategy, project_id, run_id, max(limit * 2, 20), since_year) for source in self.sources for strategy in strategies]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        sourced: list[SourcedCandidate] = []
        errors: list[str] = []
        for result in results:
            if isinstance(result, Exception):
                errors.append(f"{type(result).__name__}: {result}")
                continue
            for rank, paper in enumerate(result.papers, start=1):
                if not paper.title.strip():
                    continue
                sourced.append(SourcedCandidate(paper, SearchProvenance(result.source, result.strategy.name, result.query, rank, result.raw_checksum)))
        if not sourced:
            raise RuntimeError("All scholarly searches failed or returned no usable candidates.")

        raw_candidate_count = len(sourced)
        unique = deduplicate(sourced)
        # Citation chaining (deterministic, zero tokens, parallel): anchor on
        # the top-ranked candidates and pull their forward/backward citation
        # neighbors into the same pool before final selection. Re-deduplication
        # merges overlaps; any chaining failure degrades to the plain pool.
        seed_title = normalized_title(seed.title)
        chained: list[SourcedCandidate] = []
        if self.citation_expansion and unique and self.citation_anchors:
            pre_ranked = rank_candidates(unique, seed_text=f"{seed.title} {seed.text[:8000]}", seed_methods=seed.methods)
            chained = await self._citation_expansion(
                pre_ranked[: self.citation_anchors],
                project_id=project_id,
                run_id=run_id,
                seed_title=seed_title,
                seed_doi=seed.doi,
            )
            if chained:
                unique = deduplicate([*sourced, *chained])
        # True duplicates merged across the whole pool (search + chained),
        # so the number can never go negative when expansion adds candidates.
        duplicate_count = len(sourced) + len(chained) - len(unique)
        unique_candidate_count = len(unique)
        before_seed_exclusion = len(unique)
        unique = [
            candidate
            for candidate in unique
            if not (
                seed.doi
                and candidate.paper.doi
                and candidate.paper.doi.lower() == seed.doi.lower()
            )
            and normalized_title(candidate.paper.title) != seed_title
        ]
        seed_excluded_count = before_seed_exclusion - len(unique)
        ranked = rank_candidates(unique, seed_text=f"{seed.title} {seed.text[:8000]}", seed_methods=seed.methods)
        selected = ranked[:limit]
        records: list[PaperRecord] = []
        metadata_by_id: dict[str, MetadataPaper] = {}
        for candidate, score in selected:
            record = self._paper_record(project_id, run_id, candidate.paper)
            records.append(record)
            metadata_by_id[record.id] = candidate.paper
            self.repository.save_paper(record)
            self.repository.save_discovery_score(project_id=project_id, run_id=run_id, paper_id=record.id, components=score.components())
            for provenance in candidate.provenance:
                self.repository.save_candidate(SearchCandidate(uuid4().hex, project_id, run_id, record.id, provenance.source, provenance.strategy, provenance.source_rank, provenance.query, provenance.raw_checksum))

        downloads = await asyncio.gather(*(self._acquire(record, metadata_by_id[record.id]) for record in records))
        return DiscoveryResult(
            project_id,
            run_id,
            limit,
            tuple(records),
            tuple(downloads),
            tuple(errors),
            raw_candidate_count,
            unique_candidate_count,
            duplicate_count,
            seed_excluded_count,
        )

    async def _citation_expansion(self, anchors: Sequence[Any], *, project_id: str, run_id: str, seed_title: str, seed_doi: str | None) -> list[SourcedCandidate]:
        """Pull forward/backward citation neighbors of the top anchors.

        Zero-token, deterministic recall expansion. Only the OpenAlex source
        supports it; when absent or unsupported this is a silent no-op (the
        plain candidate pool is already a valid, complete discovery result).
        Every failure is recorded as a search event and degrades gracefully —
        citation expansion must never fail the discovery itself.
        """
        openalex = next((source for source in self.sources if getattr(source, "name", None) == "openalex" and hasattr(source, "related")), None)
        if openalex is None:
            return []
        try:
            from autoresearch.application.progress import report as _progress
        except Exception:  # noqa: BLE001 — reporting must never affect discovery
            _progress = None
        tasks: list[tuple[str, str, str]] = []  # (item_key, work_id, direction)
        seen: set[str] = set()
        for anchor in anchors:
            paper = anchor[0].paper
            work_id = _openalex_work_id(paper)
            if work_id is None and paper.doi:
                # DOI-only anchor: one bounded lookup resolves its OpenAlex id.
                try:
                    resolved = await openalex.lookup(paper.doi)
                except Exception as exc:  # noqa: BLE001
                    self.repository.record_search_event(
                        project_id=project_id, run_id=run_id, source="openalex",
                        strategy="citation-anchor-lookup", query=paper.doi,
                        requested_limit=1, result_count=0, raw_checksum=content_checksum(""),
                        error=redact_text(f"{type(exc).__name__}: {exc}"),
                    )
                    continue
                work_id = _openalex_work_id(resolved) if resolved else None
            if work_id is None or work_id in seen:
                continue
            seen.add(work_id)
            for direction in ("forward", "backward"):
                tasks.append((f"{work_id}:{direction}", work_id, direction))
        if not tasks:
            return []
        completed = 0
        total = len(tasks)

        async def _chain(item_key: str, work_id: str, direction: str) -> list[SourcedCandidate]:
            nonlocal completed
            try:
                checkpoint = self.repository.checkpoint(run_id, "citation_expansion", item_key)
                if checkpoint and checkpoint["status"] == "completed" and checkpoint.get("payload"):
                    payload = checkpoint["payload"]
                    result = SourceSearchResult(
                        "openalex", f"{direction}:{work_id}",
                        tuple(_metadata_paper(item) for item in payload.get("papers") or []),
                        str(payload.get("raw_checksum", "")),
                    )
                else:
                    self.repository.begin_checkpoint(run_id, "citation_expansion", item_key)
                    result = await openalex.related(work_id, direction, limit=self.citation_neighbors)
                    self.repository.complete_checkpoint(
                        run_id, "citation_expansion", item_key,
                        payload={"papers": [asdict(paper) for paper in result.papers], "raw_checksum": result.raw_checksum},
                    )
                self.repository.record_search_event(
                    project_id=project_id, run_id=run_id, source="openalex",
                    strategy=f"citation-{direction}", query=result.query,
                    requested_limit=self.citation_neighbors, result_count=len(result.papers),
                    raw_checksum=result.raw_checksum,
                )
                return [
                    SourcedCandidate(
                        paper,
                        SearchProvenance(
                            result.source, f"citation-{direction}", result.query,
                            index + 1, result.raw_checksum,
                        ),
                    )
                    for index, paper in enumerate(result.papers)
                    if paper.title.strip() and normalized_title(paper.title) != seed_title
                ]
            except Exception as exc:  # noqa: BLE001 — degrade, never fail discovery
                self.repository.fail_checkpoint(
                    run_id, "citation_expansion", item_key,
                    error=redact_text(f"{type(exc).__name__}: {exc}"),
                )
                self.repository.record_search_event(
                    project_id=project_id, run_id=run_id, source="openalex",
                    strategy=f"citation-{direction}", query=f"{direction}:{work_id}",
                    requested_limit=self.citation_neighbors, result_count=0,
                    raw_checksum=content_checksum(""),
                    error=redact_text(f"{type(exc).__name__}: {exc}"),
                )
                return []
            finally:
                completed += 1
                if _progress is not None:
                    try:
                        _progress(
                            "discovery.citations", completed, total,
                            detail=f"{completed}/{total} citation chains expanded",
                        )
                    except Exception:  # noqa: BLE001
                        pass

        results = await asyncio.gather(*(_chain(item_key, work_id, direction) for item_key, work_id, direction in tasks))
        return [candidate for batch in results for candidate in batch]

    async def _search(self, source: MetadataSource, strategy: SearchStrategy, project_id: str, run_id: str, limit: int, since_year: int | None):
        item_key = f"{source.name}:{strategy.name}:{content_checksum(strategy.query)[:12]}"
        checkpoint = self.repository.checkpoint(run_id, "source_search", item_key)
        if checkpoint and checkpoint["status"] == "completed" and checkpoint.get("payload"):
            payload = checkpoint["payload"]
            papers = tuple(_metadata_paper(item) for item in payload.get("papers") or [])
            return _StrategyResult(str(payload["source"]), strategy, str(payload["query"]), papers, str(payload["raw_checksum"]))
        async with self.search_semaphore:
            last_error: Exception | None = None
            for attempt in range(3):
                self.repository.begin_checkpoint(run_id, "source_search", item_key)
                try:
                    result = await source.search(strategy.query, limit=limit, since_year=since_year)
                    break
                except Exception as exc:
                    last_error = exc
                    self.repository.fail_checkpoint(run_id, "source_search", item_key, error=redact_text(f"{type(exc).__name__}: {exc}"))
                    if attempt < 2:
                        await asyncio.sleep(_retry_delay(exc, attempt))
            else:
                assert last_error is not None
                error = redact_text(f"{source.name}/{strategy.name}: {type(last_error).__name__}: {last_error}")
                self.repository.record_search_event(project_id=project_id, run_id=run_id, source=source.name, strategy=strategy.name, query=strategy.query, requested_limit=limit, result_count=0, raw_checksum=content_checksum(""), error=error)
                raise RuntimeError(error) from last_error
        checkpoint_payload = {
            "source": result.source,
            "query": result.query,
            "papers": [asdict(paper) for paper in result.papers],
            "raw_checksum": result.raw_checksum,
        }
        self.repository.complete_checkpoint(run_id, "source_search", item_key, payload=checkpoint_payload)
        self.repository.record_search_event(project_id=project_id, run_id=run_id, source=source.name, strategy=strategy.name, query=strategy.query, requested_limit=limit, result_count=len(result.papers), raw_checksum=result.raw_checksum)
        return _StrategyResult(result.source, strategy, result.query, result.papers, result.raw_checksum)

    async def _acquire(self, paper: PaperRecord, metadata: MetadataPaper) -> DownloadRecord:
        item_key = paper.id
        checkpoint = self.repository.checkpoint(paper.run_id, "pdf_acquisition", item_key)
        existing = self.repository.download_for_paper(paper.id)
        if checkpoint and checkpoint["status"] == "completed" and existing:
            if existing.status is not DownloadStatus.DOWNLOADED:
                return existing
            if existing.path and Path(existing.path).is_file():
                payload = Path(existing.path).read_bytes()
                if existing.checksum == content_checksum(payload):
                    return existing
        self.repository.begin_checkpoint(paper.run_id, "pdf_acquisition", item_key)
        async with self.download_semaphore:
            for attempt in range(3):
                try:
                    resolution = await self.pdf_resolver.resolve(metadata)
                    if not resolution.available:
                        record = DownloadRecord(_download_id(paper), paper.project_id, paper.run_id, paper.id, DownloadStatus.UNAVAILABLE, resolution.source, resolution.url, resolution.legal_basis, resolution.license, None, resolution.reason, None)
                    else:
                        target = self.download_root / paper.project_id / f"{paper.id}.pdf"
                        downloaded = await self.pdf_resolver.download(resolution, target)
                        record = DownloadRecord(_download_id(paper), paper.project_id, paper.run_id, paper.id, DownloadStatus.DOWNLOADED, resolution.source, resolution.url, resolution.legal_basis, resolution.license, str(downloaded.path), None, downloaded.checksum)
                    break
                except ValueError as exc:
                    record = DownloadRecord(_download_id(paper), paper.project_id, paper.run_id, paper.id, DownloadStatus.REJECTED, metadata.source, metadata.open_access_pdf_url, None, None, None, redact_text(str(exc)), None)
                    break
                except Exception as exc:
                    if attempt < 2:
                        await asyncio.sleep(_retry_delay(exc, attempt))
                        continue
                    record = DownloadRecord(_download_id(paper), paper.project_id, paper.run_id, paper.id, DownloadStatus.FAILED, metadata.source, metadata.open_access_pdf_url, None, None, None, redact_text(f"{type(exc).__name__}: {exc}"), None)
        self.repository.save_download(record)
        if record.status is DownloadStatus.FAILED:
            self.repository.fail_checkpoint(paper.run_id, "pdf_acquisition", item_key, error=record.reason or "PDF acquisition failed")
        else:
            self.repository.complete_checkpoint(paper.run_id, "pdf_acquisition", item_key, payload={"record_id": record.id, "status": record.status.value, "checksum": record.checksum})
        return record

    @staticmethod
    def _paper_record(project_id: str, run_id: str, paper: MetadataPaper) -> PaperRecord:
        key = deduplication_key(paper)
        # A paper may occur in many independent projects. Scope its storage ID
        # to the project while keeping deduplication deterministic within it.
        identifier = content_checksum(f"{project_id}|{key}")[:24]
        payload = metadata_payload(paper)
        sources = tuple(sorted(filter(None, paper.source.split(","))))
        return PaperRecord(identifier, project_id, run_id, paper.title, paper.abstract, paper.authors, paper.year, paper.publication_date, paper.doi, paper.arxiv_id, paper.url, sources, paper.open_access_pdf_url, paper.is_open_access, paper.citation_count, paper.venue, content_checksum(payload))

    async def close(self) -> None:
        await asyncio.gather(*(source.close() for source in self.sources), return_exceptions=True)
        await self.pdf_resolver.close()


@dataclass(frozen=True)
class _StrategyResult:
    source: str
    strategy: SearchStrategy
    query: str
    papers: tuple[MetadataPaper, ...]
    raw_checksum: str


def _metadata_paper(payload: dict) -> MetadataPaper:
    values = dict(payload)
    values["authors"] = tuple(values.get("authors") or [])
    return MetadataPaper(**values)


def _openalex_work_id(paper: MetadataPaper) -> str | None:
    external = (paper.external_id or "").strip()
    return external if external.startswith("https://openalex.org/") else None


def _download_id(paper: PaperRecord) -> str:
    return content_checksum(f"download|{paper.project_id}|{paper.id}")[:32]


def _retry_delay(exc: Exception, attempt: int) -> float:
    response = getattr(exc, "response", None)
    retry_after = response.headers.get("retry-after") if response is not None else None
    try:
        if retry_after:
            return max(0.0, min(30.0, float(retry_after)))
        if response is not None and response.status_code == 429:
            return 2.0 * (2**attempt)
        return 0.5 * (2**attempt)
    except (TypeError, ValueError):
        return 0.5 * (2**attempt)
