"""Durable orchestration for the first AutoResearch milestone."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

logger = logging.getLogger(__name__)

from autoresearch.domain.models import (
    ArtifactProvenance,
    ArtifactReference,
    CandidateStatus,
    DownloadStatus,
    FindingStatus,
    GapClassification,
    NoveltyCandidate,
    PaperRecord,
    PipelineKind,
    PriorArtStatus,
    ResearchFinding,
    ResearchGap,
    RunStatus,
    SeedPaper,
    ExperimentSpec,
    content_checksum,
    utc_now,
)
from autoresearch.ingestion.chunking import DocumentChunk, chunk_document
from autoresearch.ingestion.document_parser import (
    DocumentParser,
    ParsedDocument,
    ParsedPage,
)
from autoresearch.ingestion.extraction import (
    SEED_PROMPT_VERSION,
    SeedAnalysis,
    SeedAnalyzer,
)
from autoresearch.papers.search_pipeline import (
    DiscoveryResult,
    SearchStrategy,
    StudySearchPipeline,
)
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.errors import ErrorKind, ProviderError
from autoresearch.experiments.integrity import IntegrityFailure
from autoresearch.experiments.matrix import ExperimentMatrix
from autoresearch.application.findings import FindingsMemory
from autoresearch.providers.redaction import redact_text
from autoresearch.providers.router import ModelRouter


def require_model_role(router: ModelRouter, role: ModelRole) -> None:
    """Actionable early diagnostic: fail fast with the remedy when a stage's
    required model role is unconfigured, instead of a downstream symptom."""
    try:
        router.identity(role)
    except ProviderError as exc:
        raise ProviderError(
            f"provider_unconfigured: {exc.args[0]} "
            "Copy .autoresearch/config.example.toml to .autoresearch/config.toml, "
            "fill the provider placeholders, and export the referenced key "
            "environment variables (see README 'Provider configuration'). "
            "Offline commands (discover --from-catalog, skill status, audit) "
            "work without a provider.",
            getattr(exc, "kind", None),
        ) from exc


def require_provider_credentials(router: ModelRouter, role: ModelRole) -> None:
    """Fail fast (once, up front) when the role's provider credentials are
    unavailable.

    A missing API key is deterministic and global: without this check every
    per-paper call fails identically, one failure is recorded per paper, and
    the run only ends with a misleading 'Only 0 papers produced
    evidence-backed reviews' count error (observed 2026-08-09: 140 identical
    environment-variable failures). Providers without an api_key resolver
    (deterministic test fakes) are skipped.
    """
    assignment = router.assignments.get(role)
    if not assignment:
        return  # require_model_role reports unconfigured roles with the remedy
    names = [assignment.provider, *(name for name, _model in assignment.fallbacks)]
    last_error: ProviderError | None = None
    for name in names:
        resolver = getattr(router.registry.get(name), "api_key", None)
        if resolver is None:
            return  # non-HTTP provider; nothing to resolve
        try:
            resolver()
            return  # the first resolvable credential satisfies the role
        except ProviderError as exc:
            last_error = exc
    detail = f" Underlying error: {last_error}" if last_error else ""
    raise ProviderError(
        "provider_unauthenticated: no usable credential for role "
        f"'{role.value}' (tried providers: {', '.join(names)}).{detail} "
        "Export the provider's api_key_env variable in the shell that runs "
        "autoresearch (OpenCode-agent sessions must inject it) or reconnect "
        "the provider (OpenCode /connect). See .autoresearch/config.toml.",
        ErrorKind.AUTHENTICATION,
    ) from last_error


from autoresearch.review.extraction import PaperReviewExtractor, ReviewBundle
from autoresearch.review.gap_analysis import GapAnalyzer
from autoresearch.review.similarity import ReviewRanker
from autoresearch.review.synthesis import ReviewSynthesisWriter
from autoresearch.storage.repository import ResearchRepository
from autoresearch.novelty.pipeline import NoveltyPipeline
from autoresearch.novelty.candidate_generator import LLMCandidateGenerator
from autoresearch.novelty.prior_art_checker import EvidenceBackedPriorArtChecker
from autoresearch.novelty.novelty_critic import ScientificCritic
from autoresearch.novelty.selection import PortfolioSelector
from autoresearch.experiments.pipeline import ExperimentPipeline
from autoresearch.experiments.planner import ExperimentSpecGenerator
from autoresearch.experiments.environment import IsolatedEnvironmentBuilder
from autoresearch.experiments.executor import DurableExperimentExecutor
from autoresearch.experiments.result_analysis import StatisticalAnalyzer
from autoresearch.experiments.report_writer import ScientificReportWriter
from autoresearch.experiments.policy import (
    ActionType,
    ExecutionAction,
    specification_checksum,
)
from autoresearch.workers.local_typed import LocalTypedWorker

SEARCH_PROMPT_VERSION = "search-strategies-v2"
SEARCH_STRATEGIES_SCHEMA = {
    "type": "object",
    "properties": {
        "strategies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "query": {"type": "string"}},
                "required": ["name", "query"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["strategies"],
    "additionalProperties": False,
}

# Complementary retrieval perspectives. Each runs as an independent parallel
# generation call (same wall-clock as one call, wider recall), and the union
# is deduplicated. Partial perspective failure degrades to the survivors.
_SEARCH_PERSPECTIVES = (
    ("topic-method", "Emphasize the seed's core topic and its primary methods."),
    ("application-domain", "Emphasize applications, datasets, benchmarks, and downstream domains where these methods apply."),
    ("limitation-counter", "Emphasize the seed's stated limitations and counter-evidence; surface work that challenges or circumvents them."),
)


@dataclass(frozen=True)
class DiscoveryOutcome:
    project_id: str
    run_id: str
    requested: int
    retrieved: int
    downloaded: int
    unavailable: int
    rejected: int
    failed: int
    artifact_path: str


@dataclass(frozen=True)
class ReviewOutcome:
    project_id: str
    run_id: str
    reviewed: int
    failed: int
    top_paper_ids: tuple[str, ...]
    gap: ResearchGap
    json_report: str
    markdown_report: str
    reused: int = 0


class SearchStrategyGenerator:
    def __init__(self, router: ModelRouter, *, perspectives: int = 3) -> None:
        self.router = router
        self.perspectives = max(1, min(8, perspectives))

    async def generate(
        self, analysis: SeedAnalysis, *, project_id: str, run_id: str
    ) -> tuple[SearchStrategy, ...]:
        """Generate strategies from parallel complementary perspectives.

        Width is the recall lever: each perspective call runs concurrently, so
        the wall-clock is that of a single call. One perspective failing must
        not discard the others' strategies; only fewer than three distinct
        strategies in total is a hard failure (as before).
        """
        selected = _SEARCH_PERSPECTIVES[: self.perspectives]
        results = await asyncio.gather(
            *(
                self._generate_perspective(
                    analysis, perspective, directive, project_id, run_id
                )
                for perspective, directive in selected
            ),
            return_exceptions=True,
        )
        strategies: list[SearchStrategy] = []
        seen: set[str] = set()
        for result in results:
            if isinstance(result, Exception):
                continue
            for raw in result.get("strategies") or []:
                if not isinstance(raw, dict):
                    continue
                name = " ".join(str(raw.get("name") or "").split())
                query = " ".join(str(raw.get("query") or "").split())
                if name and query and query.lower() not in seen:
                    strategies.append(SearchStrategy(name, query))
                    seen.add(query.lower())
        if len(strategies) < 3:
            raise ValueError(
                "Provider must return at least three distinct search strategies."
            )
        return tuple(strategies[:32])

    async def _generate_perspective(
        self,
        analysis: SeedAnalysis,
        perspective: str,
        directive: str,
        project_id: str,
        run_id: str,
    ) -> dict[str, Any]:
        return await self.router.structured_output(
            ModelRole.RESEARCH_REASONING,
            messages=[
                {
                    "role": "system",
                    "content": "Generate complementary scholarly-search queries. Queries must be concise and grounded in the supplied seed analysis. "
                    + directive,
                },
                {
                    "role": "user",
                    "content": f"Prompt version: {SEARCH_PROMPT_VERSION}\nPerspective: {perspective}\nTopic: {analysis.topic}\nQuestion: {analysis.research_question}\nMethods: {list(analysis.methods)}\nDatasets: {list(analysis.datasets)}\nKeywords: {list(analysis.keywords)}\nReturn strategies[] with unique name and query.",
                },
            ],
            schema={"name": "search_strategies", "schema": SEARCH_STRATEGIES_SCHEMA},
            project_id=project_id,
            run_id=run_id,
            operation=f"search_strategy_generation:{perspective}",
        )


class DiscoveryService:
    def __init__(
        self,
        repository: ResearchRepository,
        router: ModelRouter,
        search_pipeline: StudySearchPipeline,
        *,
        parser: DocumentParser | None = None,
        strategy_perspectives: int = 3,
    ) -> None:
        self.repository = repository
        self.router = router
        self.search_pipeline = search_pipeline
        self.parser = parser or DocumentParser()
        self.seed_analyzer = SeedAnalyzer(router)
        self.strategy_generator = SearchStrategyGenerator(
            router, perspectives=strategy_perspectives
        )

    async def import_catalog(
        self,
        catalog_path: Path,
        *,
        project_title: str | None = None,
    ) -> DiscoveryOutcome:
        """Import papers from a local JSON catalog without external API calls."""
        catalog_path = catalog_path.expanduser().resolve()
        if not catalog_path.is_file():
            raise ValueError(f"Catalog file not found: {catalog_path}")
        with open(catalog_path, encoding="utf-8") as handle:
            catalog = json.load(handle)
        if not isinstance(catalog, list):
            raise ValueError("Catalog must be a JSON array of paper entries.")
        title = project_title or f"Local catalog: {catalog_path.name}"
        project = self.repository.create_project(title)
        run = self.repository.create_run(
            project.id,
            PipelineKind.DISCOVERY,
            {
                "seed_input": str(catalog_path),
                "limit": len(catalog),
                "since_year": None,
                "max_cost_usd": None,
                "max_runtime_seconds": None,
                "source": "local_catalog",
            },
        )
        papers: list[PaperRecord] = []
        skipped = 0
        for entry in catalog:
            if not isinstance(entry, dict):
                skipped += 1
                continue
            raw_pid = str(entry.get("id") or uuid4().hex)
            paper_id = f"{project.id[:8]}_{raw_pid}"
            title_text = " ".join(str(entry.get("title") or "").split())
            if not title_text:
                skipped += 1
                continue
            authors_raw = entry.get("authors")
            if isinstance(authors_raw, str) and authors_raw.strip():
                authors = tuple(a.strip() for a in authors_raw.split(",") if a.strip())
            elif isinstance(authors_raw, list):
                authors = tuple(str(a) for a in authors_raw if str(a).strip())
            else:
                authors = ()
            year = None
            raw_year = entry.get("year")
            if raw_year is not None:
                try:
                    year = int(raw_year)
                except (TypeError, ValueError):
                    pass
            doi = str(entry.get("doi") or "").strip() or None
            venue = str(entry.get("venue") or "").strip()
            abstract = str(entry.get("abstract") or entry.get("raw_text") or "").strip()
            url = str(entry.get("pdf_url") or entry.get("url") or "").strip() or None
            source_names = ("local_catalog",)
            checksum = content_checksum(f"{paper_id}|{title_text}|{abstract[:200]}")
            paper = PaperRecord(
                id=paper_id,
                project_id=project.id,
                run_id=run.id,
                title=title_text,
                abstract=abstract,
                authors=authors,
                year=year,
                publication_date=None,
                doi=doi,
                arxiv_id=None,
                url=url,
                source_names=source_names,
                open_access_pdf_url=url,
                is_open_access=bool(url),
                citation_count=0,
                venue=venue,
                checksum=checksum,
            )
            papers.append(paper)
        if not papers:
            raise ValueError("No valid paper entries found in catalog.")
        imported = self.repository.import_papers_batch(papers)
        if papers:
            p0 = papers[0]
            # Provenance label only — catalog import makes no model call. With
            # no provider configured, record the deterministic local origin
            # instead of failing an otherwise fully offline import.
            try:
                extraction_provider, extraction_model = self.router.identity(
                    ModelRole.LITERATURE_EXTRACTION
                )
            except ProviderError:
                extraction_provider, extraction_model = "local", "catalog_import"
            seed_prov = ArtifactProvenance.create(
                project_id=project.id,
                run_id=run.id,
                source_evidence=(f"catalog-checksum:{p0.checksum}",),
                provider=extraction_provider,
                model=extraction_model,
                prompt_version="catalog-v1",
                content={"title": p0.title},
            )
            seed = SeedPaper(
                id=uuid4().hex,
                project_id=project.id,
                run_id=run.id,
                input_type="catalog",
                input_value=str(catalog_path),
                title=p0.title,
                text=p0.abstract,
                topic="Catalog Literature",
                research_question=f"Literature analysis for {p0.title}",
                checksum=p0.checksum,
                provenance=seed_prov,
                doi=p0.doi,
                url=p0.url,
            )
            self.repository.save_seed(seed)
        self.repository.finish_run(run.id)
        artifact_path = str(
            self.repository.artifact_root / project.id / run.id / "catalog_import.json"
        )
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
        Path(artifact_path).write_text(
            json.dumps(
                {
                    "project_id": project.id,
                    "run_id": run.id,
                    "catalog_path": str(catalog_path),
                    "entries": len(catalog),
                    "imported": imported,
                    "skipped": skipped,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return DiscoveryOutcome(
            project_id=project.id,
            run_id=run.id,
            requested=len(catalog),
            retrieved=imported,
            downloaded=0,
            unavailable=0,
            rejected=skipped,
            failed=0,
            artifact_path=artifact_path,
        )

    async def discover(
        self,
        seed_input: str | Path,
        *,
        limit: int = 20,
        since_year: int | None = None,
        max_cost_usd: float | None = None,
        max_runtime_seconds: float | None = None,
    ) -> DiscoveryOutcome:
        parsed = self.parser.parse_seed(seed_input)
        project = self.repository.create_project(parsed.title)
        run = self.repository.create_run(
            project.id,
            PipelineKind.DISCOVERY,
            {
                "seed_input": str(seed_input),
                "limit": limit,
                "since_year": since_year,
                "max_cost_usd": max_cost_usd,
                "max_runtime_seconds": max_runtime_seconds,
            },
        )
        if max_cost_usd is not None:
            self.repository.set_project_budget(project.id, hard_stop=max_cost_usd)
        return await self._execute(
            project.id,
            run.id,
            parsed,
            limit=limit,
            since_year=since_year,
            max_runtime_seconds=max_runtime_seconds,
        )

    async def resume(self, run_id: str) -> DiscoveryOutcome:
        run = self.repository.get_run(run_id)
        if not run or run.kind is not PipelineKind.DISCOVERY:
            raise ValueError(f"Discovery run '{run_id}' was not found.")
        failed = [
            item
            for item in self.repository.checkpoints(run_id)
            if item["status"] == "failed"
        ]
        if run.status is RunStatus.COMPLETED and not failed:
            return self._completed_outcome(run.project_id, run.id)
        data = dict(run.input_data)
        if data.get("max_cost_usd") is not None:
            self.repository.set_project_budget(
                run.project_id, hard_stop=float(data["max_cost_usd"])
            )
        self.repository.resume_run(run.id)
        parsed = self.parser.parse_seed(str(data["seed_input"]))
        return await self._execute(
            run.project_id,
            run.id,
            parsed,
            limit=int(data.get("limit", 20)),
            since_year=int(data["since_year"])
            if data.get("since_year") is not None
            else None,
            max_runtime_seconds=float(data["max_runtime_seconds"])
            if data.get("max_runtime_seconds") is not None
            else None,
        )

    async def _execute(
        self,
        project_id: str,
        run_id: str,
        parsed: ParsedDocument,
        *,
        limit: int,
        since_year: int | None,
        max_runtime_seconds: float | None,
    ) -> DiscoveryOutcome:
        guard = _ExecutionGuard(max_runtime_seconds)
        try:
            hydrated_checkpoint = self.repository.checkpoint(run_id, "seed_hydration")
            if (
                hydrated_checkpoint
                and hydrated_checkpoint["status"] == "completed"
                and hydrated_checkpoint.get("payload")
            ):
                parsed = _parsed_document(hydrated_checkpoint["payload"])
            else:
                self.repository.begin_checkpoint(run_id, "seed_hydration")
                parsed = await self._hydrate_identifier_seed(parsed)
                self.repository.complete_checkpoint(
                    run_id, "seed_hydration", payload=_parsed_payload(parsed)
                )
            guard.require_allowed("seed analysis")
            analysis_checkpoint = self.repository.checkpoint(run_id, "seed_analysis")
            if (
                analysis_checkpoint
                and analysis_checkpoint["status"] == "completed"
                and analysis_checkpoint.get("payload")
            ):
                analysis = _seed_analysis(analysis_checkpoint["payload"])
            else:
                self.repository.begin_checkpoint(run_id, "seed_analysis")
                analysis = await self.seed_analyzer.analyze(
                    parsed, project_id=project_id, run_id=run_id
                )
                self.repository.complete_checkpoint(
                    run_id, "seed_analysis", payload=asdict(analysis)
                )
            extraction_provider, extraction_model = self.router.identity(
                ModelRole.LITERATURE_EXTRACTION
            )
            analysis_content = {
                "title": analysis.title,
                "topic": analysis.topic,
                "research_question": analysis.research_question,
                "methods": analysis.methods,
                "datasets": analysis.datasets,
                "citations": analysis.citations,
                "limitations": analysis.limitations,
                "keywords": analysis.keywords,
            }
            seed = self.repository.get_seed(project_id)
            if not seed:
                seed_provenance = ArtifactProvenance.create(
                    project_id=project_id,
                    run_id=run_id,
                    source_evidence=(f"seed-checksum:{parsed.checksum}",),
                    provider=extraction_provider,
                    model=extraction_model,
                    prompt_version=SEED_PROMPT_VERSION,
                    content=analysis_content,
                )
                seed = SeedPaper(
                    id=uuid4().hex,
                    project_id=project_id,
                    run_id=run_id,
                    input_type=parsed.input_type,
                    input_value=parsed.input_value,
                    title=analysis.title,
                    text=parsed.text,
                    topic=analysis.topic,
                    research_question=analysis.research_question,
                    checksum=parsed.checksum,
                    provenance=seed_provenance,
                    doi=parsed.doi,
                    url=parsed.url,
                    methods=analysis.methods,
                    datasets=analysis.datasets,
                    citations=analysis.citations,
                    limitations=analysis.limitations,
                )
                self.repository.save_seed(seed)
            guard.require_allowed("search strategy generation")
            strategy_checkpoint = self.repository.checkpoint(
                run_id, "search_strategies"
            )
            if (
                strategy_checkpoint
                and strategy_checkpoint["status"] == "completed"
                and strategy_checkpoint.get("payload")
            ):
                strategies = tuple(
                    SearchStrategy(str(item["name"]), str(item["query"]))
                    for item in strategy_checkpoint["payload"]
                )
            else:
                self.repository.begin_checkpoint(run_id, "search_strategies")
                strategies = await self.strategy_generator.generate(
                    analysis, project_id=project_id, run_id=run_id
                )
                self.repository.complete_checkpoint(
                    run_id,
                    "search_strategies",
                    payload=[asdict(item) for item in strategies],
                )
            guard.require_allowed("scholarly discovery")
            result = await self.search_pipeline.discover(
                project_id=project_id,
                run_id=run_id,
                seed=seed,
                strategies=strategies,
                limit=limit,
                since_year=since_year,
            )
            artifact = self._write_discovery_artifact(
                seed, analysis, strategies, result
            )
            self.repository.complete_checkpoint(
                run_id,
                "discovery_artifact",
                payload={"path": artifact.path, "checksum": artifact.checksum},
            )
            self.repository.finish_run(run_id)
            return DiscoveryOutcome(
                project_id,
                run_id,
                limit,
                len(result.papers),
                result.downloaded_count,
                sum(
                    item.status is DownloadStatus.UNAVAILABLE
                    for item in result.downloads
                ),
                sum(
                    item.status is DownloadStatus.REJECTED for item in result.downloads
                ),
                sum(item.status is DownloadStatus.FAILED for item in result.downloads),
                artifact.path,
            )
        except BaseException as exc:
            self.repository.finish_run(
                run_id, error=redact_text(f"{type(exc).__name__}: {exc}")
            )
            raise

    def _completed_outcome(self, project_id: str, run_id: str) -> DiscoveryOutcome:
        artifacts = [
            item
            for item in self.repository.artifacts_for_run(run_id)
            if item.artifact_type == "discovery_summary"
        ]
        if not artifacts:
            raise RuntimeError(
                f"Completed discovery run '{run_id}' has no discovery artifact."
            )
        payload = json.loads(Path(artifacts[-1].path).read_text(encoding="utf-8"))
        statuses = [item.get("status") for item in payload.get("downloads") or []]
        return DiscoveryOutcome(
            project_id,
            run_id,
            int(payload["requested"]),
            int(payload["retrieved"]),
            statuses.count("downloaded"),
            statuses.count("unavailable"),
            statuses.count("rejected"),
            statuses.count("failed"),
            artifacts[-1].path,
        )

    async def _hydrate_identifier_seed(self, parsed: ParsedDocument) -> ParsedDocument:
        if parsed.text:
            return parsed
        query = parsed.doi or parsed.url or parsed.input_value
        results = await asyncio.gather(
            *(source.search(query, limit=3) for source in self.search_pipeline.sources),
            return_exceptions=True,
        )
        candidates = [
            paper
            for result in results
            if not isinstance(result, Exception)
            for paper in result.papers
        ]
        if parsed.doi:
            exact = [paper for paper in candidates if paper.doi == parsed.doi]
            candidates = exact or candidates
        if not candidates:
            raise ValueError(
                "Seed DOI/URL could not be resolved to scholarly metadata."
            )
        paper = max(candidates, key=lambda item: len(item.abstract))
        return ParsedDocument(
            parsed.input_type,
            parsed.input_value,
            paper.title or parsed.title,
            (ParsedPage(1, paper.abstract),),
            parsed.checksum,
            parsed.doi or paper.doi,
            parsed.url or paper.url,
        )

    def _write_discovery_artifact(
        self,
        seed: SeedPaper,
        analysis: SeedAnalysis,
        strategies: Sequence[SearchStrategy],
        result: DiscoveryResult,
    ) -> ArtifactReference:
        provider, model = self.router.identity(ModelRole.RESEARCH_REASONING)
        source_summary: dict[str, dict[str, int]] = {}
        for event in self.repository.search_events(result.run_id):
            summary = source_summary.setdefault(
                event["source"],
                {"requests": 0, "successes": 0, "failures": 0, "records": 0},
            )
            summary["requests"] += 1
            summary["records"] += int(event["result_count"])
            summary["failures" if event.get("error") else "successes"] += 1
        payload = {
            "project_id": seed.project_id,
            "run_id": seed.run_id,
            "seed_id": seed.id,
            "provider": provider,
            "model": model,
            "prompt_version": SEARCH_PROMPT_VERSION,
            "timestamp": utc_now(),
            "source_evidence": [seed.id, *(paper.id for paper in result.papers)],
            "seed_analysis": analysis.__dict__,
            "strategies": [item.__dict__ for item in strategies],
            "requested": result.requested,
            "retrieved": len(result.papers),
            "downloaded": result.downloaded_count,
            "downloads": [
                {
                    "paper_id": item.paper_id,
                    "status": item.status.value,
                    "source": item.source,
                    "legal_basis": item.legal_basis,
                    "reason": item.reason,
                    "checksum": item.checksum,
                }
                for item in result.downloads
            ],
            "source_errors": result.source_errors,
            "source_summary": source_summary,
            "deduplication": {
                "raw_candidates": result.raw_candidate_count,
                "unique_candidates": result.unique_candidate_count,
                "duplicates_merged": result.duplicate_count,
                "seed_records_excluded": result.seed_excluded_count,
                "selected": len(result.papers),
            },
        }
        root = self.repository.artifact_root / seed.project_id / seed.run_id
        root.mkdir(parents=True, exist_ok=True)
        path = root / "discovery.json"
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        path.write_text(text, encoding="utf-8")
        artifact = ArtifactReference(
            content_checksum(f"discovery_summary|{seed.run_id}")[:32],
            seed.project_id,
            seed.run_id,
            "discovery_summary",
            str(path),
            tuple(payload["source_evidence"]),
            provider,
            model,
            SEARCH_PROMPT_VERSION,
            content_checksum(text),
        )
        self.repository.save_artifact(artifact)
        return artifact


def _review_fanout(router: ModelRouter) -> int:
    """Size the review fan-out from the extraction provider's throughput ceiling.

    The provider's own gate enforces the real request limit; this outer
    semaphore only bounds concurrent review tasks and must not silently
    strangle a faster provider configuration. Falls back to 4 when the
    provider exposes no config (e.g. test fakes).
    """
    assignment = router.assignments.get(ModelRole.LITERATURE_EXTRACTION)
    provider = router.registry.get(assignment.provider) if assignment else None
    config = getattr(provider, "config", None)
    if config is None:
        return 4
    ceiling = getattr(config, "max_concurrency", None) or getattr(
        config, "concurrency", 4
    )
    return max(4, int(ceiling))


class LiteratureReviewService:
    def __init__(
        self,
        repository: ResearchRepository,
        router: ModelRouter,
        *,
        parser: DocumentParser | None = None,
        ensemble: bool = True,
    ) -> None:
        self.repository = repository
        self.router = router
        self.parser = parser or DocumentParser()
        self.extractor = PaperReviewExtractor(router)
        self.ranker = ReviewRanker(router)
        self.gap_analyzer = GapAnalyzer(router)
        self.writer = ReviewSynthesisWriter(repository)
        self.review_semaphore = asyncio.Semaphore(_review_fanout(router))
        self.ensemble = ensemble

    async def review(
        self,
        project_id: str,
        *,
        top_k: int = 5,
        hierarchical: bool = False,
        max_cost_usd: float | None = None,
        max_runtime_seconds: float | None = None,
        fresh: bool = False,
    ) -> ReviewOutcome:
        require_model_role(self.router, ModelRole.LITERATURE_EXTRACTION)
        if not 1 <= top_k <= 15:
            raise ValueError("top_k must be between 1 and 15.")
        project = self.repository.get_project(project_id)
        seed = self.repository.get_seed(project_id)
        if not project or not seed:
            raise ValueError("Project or seed paper was not found.")
        papers = self.repository.list_papers(project_id)
        if not papers:
            raise ValueError("Run discovery before literature review.")
        # A previous review process that died mid-run leaves an orphaned
        # 'running' record that blocks clean status forever. Paper checkpoints
        # and verified reviews are reusable ACROSS runs, so retiring the
        # orphan loses nothing and makes the new run the single source of
        # truth.
        for row in self.repository.rows(
            "SELECT id FROM pipeline_runs WHERE project_id=? AND kind=? AND status=?",
            (project_id, PipelineKind.REVIEW.value, RunStatus.RUNNING.value),
        ):
            self.repository.cancel_run(row["id"])
        run = self.repository.create_run(
            project_id,
            PipelineKind.REVIEW,
            {
                "top_k": top_k,
                "paper_count": len(papers),
                "hierarchical": hierarchical,
                "max_cost_usd": max_cost_usd,
                "max_runtime_seconds": max_runtime_seconds,
            },
        )
        if max_cost_usd is not None:
            self.repository.set_project_budget(project_id, hard_stop=max_cost_usd)
        return await self._execute(
            run.id,
            project_id,
            top_k=top_k,
            hierarchical=hierarchical,
            max_runtime_seconds=max_runtime_seconds,
            fresh=fresh,
        )

    async def resume(self, run_id: str) -> ReviewOutcome:
        run = self.repository.get_run(run_id)
        if not run or run.kind is not PipelineKind.REVIEW:
            raise ValueError(f"Review run '{run_id}' was not found.")
        failed = [
            item
            for item in self.repository.checkpoints(run_id)
            if item["status"] == "failed"
        ]
        if run.status is RunStatus.COMPLETED and not failed:
            return self._completed_outcome(run.project_id, run.id)
        data = dict(run.input_data)
        if data.get("max_cost_usd") is not None:
            self.repository.set_project_budget(
                run.project_id, hard_stop=float(data["max_cost_usd"])
            )
        self.repository.resume_run(run.id)
        return await self._execute(
            run.id,
            run.project_id,
            top_k=int(data.get("top_k", 5)),
            hierarchical=bool(data.get("hierarchical", False)),
            max_runtime_seconds=float(data["max_runtime_seconds"])
            if data.get("max_runtime_seconds") is not None
            else None,
        )

    async def _execute(
        self,
        run_id: str,
        project_id: str,
        *,
        top_k: int,
        hierarchical: bool,
        max_runtime_seconds: float | None,
        fresh: bool = False,
    ) -> ReviewOutcome:
        limit = max_runtime_seconds
        if limit is not None and limit <= 0:
            raise ValueError("max_runtime_seconds must be positive.")
        deadline = asyncio.timeout(limit)
        try:
            async with deadline:
                return await self._execute_review(
                    run_id,
                    project_id,
                    top_k=top_k,
                    hierarchical=hierarchical,
                    max_runtime_seconds=limit,
                    fresh=fresh,
                )
        except TimeoutError as exc:
            if not deadline.expired():
                raise
            message = f"runtime_limit: review exceeded {limit:g}s; completed paper checkpoints are reusable."
            self.repository.finish_run(run_id, error=message)
            raise TimeoutError(message) from exc

    async def _execute_review(
        self,
        run_id: str,
        project_id: str,
        *,
        top_k: int,
        hierarchical: bool,
        max_runtime_seconds: float | None,
        fresh: bool = False,
    ) -> ReviewOutcome:
        guard = _ExecutionGuard(max_runtime_seconds)
        project = self.repository.get_project(project_id)
        seed = self.repository.get_seed(project_id)
        papers = self.repository.list_papers(project_id)
        if not project or not seed or not papers:
            raise ValueError("Project discovery state is incomplete.")
        try:
            guard.require_allowed("paper review")
            require_provider_credentials(self.router, ModelRole.LITERATURE_EXTRACTION)
            from autoresearch.application.progress import report as _progress

            completed = {"count": 0}
            total = len(papers)
            fatal = asyncio.Event()
            fatal_error: list[BaseException] = []
            tasks: list[asyncio.Task] = []

            async def _review_tracked(paper):
                try:
                    return await self._review_one(
                        paper, run_id, hierarchical=hierarchical, allow_reuse=not fresh
                    )
                except BaseException as exc:
                    if _is_hard_error(exc) and not fatal.is_set():
                        # Systemic failure (dead credential, exhausted budget,
                        # invalid contract): every remaining paper would fail
                        # identically. Cancel the rest and surface the cause
                        # NOW instead of grinding through N identical failures.
                        fatal_error.append(exc)
                        fatal.set()
                        for task in tasks:
                            if task is not asyncio.current_task():
                                task.cancel()
                        raise
                    if _is_transient_error(exc):
                        # Provider retries already ran inside the call; one
                        # extra review-level attempt absorbs cross-call flake.
                        try:
                            return await self._review_one(
                                paper, run_id, hierarchical=hierarchical, allow_reuse=not fresh
                            )
                        except BaseException as second:
                            raise second from exc
                    raise
                finally:
                    completed["count"] += 1
                    _progress(
                        "review.papers", completed["count"], total,
                        detail=f"{completed['count']}/{total} papers reviewed",
                    )

            tasks = [asyncio.create_task(_review_tracked(paper)) for paper in papers]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            if fatal.is_set():
                cause = redact_text(str(fatal_error[0]))
                raise RuntimeError(
                    f"review_aborted: systemic provider failure — {cause}. "
                    "Completed paper checkpoints are reusable; fix the cause and resume."
                )
            bundles: list[ReviewBundle] = []
            failure_reasons: list[str] = []
            for paper, result in zip(papers, results):
                if isinstance(result, asyncio.CancelledError):
                    continue  # cancelled by the systemic fail-fast path
                if isinstance(result, Exception):
                    reason = redact_text(f"{type(result).__name__}: {result}")
                    failure_reasons.append(reason)
                    self.repository.record_review_failure(
                        project_id=project_id,
                        run_id=run_id,
                        paper_id=paper.id,
                        reason=reason,
                    )
                else:
                    bundles.append(result)
            if failure_reasons:
                self.repository.append_api_event(
                    run_id, "review.partial_coverage",
                    {"reviewed": len(bundles), "failed": len(failure_reasons), "total": total},
                )
            if len(bundles) < top_k:
                # Surface the actual dominant cause, not just the count: the
                # 2026-08-09 failure looked like a review-quality problem but
                # was 140 identical missing-credential errors.
                detail = (
                    f" First failure: {failure_reasons[0][:200]}"
                    if failure_reasons
                    else ""
                )
                raise RuntimeError(
                    f"Only {len(bundles)} papers produced evidence-backed reviews; {top_k} are required.{detail}"
                )
            reviews = [bundle.review for bundle in bundles]
            claims = [claim for bundle in bundles for claim in bundle.claims]
            reviewed_papers = [
                paper
                for paper in papers
                if paper.id in {review.paper_id for review in reviews}
            ]
            guard.require_allowed("ranking embeddings")
            embedding_checkpoint = self.repository.checkpoint(
                run_id, "ranking_embeddings"
            )
            cached_vectors = None
            if (
                embedding_checkpoint
                and embedding_checkpoint["status"] == "completed"
                and embedding_checkpoint.get("payload")
            ):
                cached_vectors = embedding_checkpoint["payload"].get("vectors")
            else:
                self.repository.begin_checkpoint(run_id, "ranking_embeddings")
            try:
                all_scores = await self.ranker.rank(
                    seed=seed,
                    papers=reviewed_papers,
                    reviews=reviews,
                    run_id=run_id,
                    embedded_vectors=cached_vectors,
                    embedding_sink=lambda vectors: self.repository.complete_checkpoint(
                        run_id, "ranking_embeddings", payload={"vectors": vectors}
                    ),
                )
            except Exception as exc:
                self.repository.fail_checkpoint(
                    run_id,
                    "ranking_embeddings",
                    error=redact_text(f"{type(exc).__name__}: {exc}"),
                )
                raise
            for score in all_scores:
                self.repository.save_score(score)
            self.repository.complete_checkpoint(
                run_id,
                "ranking",
                payload={"paper_ids": [score.paper_id for score in all_scores]},
            )
            top_scores = all_scores[:top_k]
            top_ids = {score.paper_id for score in top_scores}
            top_papers = [paper for paper in reviewed_papers if paper.id in top_ids]
            top_reviews = [review for review in reviews if review.paper_id in top_ids]
            top_claims = [claim for claim in claims if claim.paper_id in top_ids]
            guard.require_allowed("gap analysis")
            gap = self.repository.gap_for_run(run_id)
            if (
                not gap
                or not (self.repository.checkpoint(run_id, "gap_analysis") or {}).get(
                    "status"
                )
                == "completed"
            ):
                self.repository.begin_checkpoint(run_id, "gap_analysis")
                try:
                    gap = await _with_transient_retry(
                        lambda: self.gap_analyzer.analyze(
                            project_id=project_id,
                            run_id=run_id,
                            papers=top_papers,
                            reviews=top_reviews,
                            claims=top_claims,
                            scores=top_scores,
                        ),
                        operation_name="gap analysis",
                    )
                    self.repository.save_gap(gap)
                    self.repository.complete_checkpoint(
                        run_id,
                        "gap_analysis",
                        payload={"gap_id": gap.id, "checksum": gap.provenance.checksum},
                    )
                except Exception as exc:
                    self.repository.fail_checkpoint(
                        run_id,
                        "gap_analysis",
                        error=redact_text(f"{type(exc).__name__}: {exc}"),
                    )
                    raise
            json_artifact, markdown_artifact = self.writer.write(
                project_id=project_id,
                run_id=run_id,
                papers=top_papers,
                reviews=top_reviews,
                scores=top_scores,
                claims=top_claims,
                gap=gap,
            )
            self.repository.complete_checkpoint(
                run_id,
                "review_artifacts",
                payload={
                    "json": json_artifact.path,
                    "markdown": markdown_artifact.path,
                },
            )
            # Decision-useful evidence matrix + coverage record (deterministic
            # assembly; failures never block the review itself).
            try:
                from autoresearch.review.matrix import save_evidence_matrix

                matrix_artifact = save_evidence_matrix(
                    self.repository, project_id=project_id, run_id=run_id
                )
                self.repository.complete_checkpoint(
                    run_id,
                    "evidence_matrix",
                    payload={
                        "path": matrix_artifact.path,
                        "checksum": matrix_artifact.checksum,
                    },
                )
            except Exception:
                logger.warning(
                    "Evidence-matrix persistence failed for run %s",
                    run_id,
                    exc_info=True,
                )
            self.repository.finish_run(run_id)
            reused = sum(1 for review in reviews if review.run_id != run_id)
            return ReviewOutcome(
                project_id,
                run_id,
                len(reviews),
                len(results) - len(reviews),
                tuple(score.paper_id for score in top_scores),
                gap,
                json_artifact.path,
                markdown_artifact.path,
                reused,
            )
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                self.repository.cancel_run(run_id)
            else:
                self.repository.finish_run(
                    run_id, error=redact_text(f"{type(exc).__name__}: {exc}")
                )
            raise

    async def _review_one(
        self, paper, run_id: str, *, hierarchical: bool, allow_reuse: bool = True
    ) -> ReviewBundle:
        checkpoint = self.repository.checkpoint(run_id, "paper_review", paper.id)
        existing = self.repository.review_for_paper(run_id, paper.id)
        if checkpoint and checkpoint["status"] == "completed" and existing:
            return ReviewBundle(
                existing,
                tuple(
                    self.repository.claims(
                        paper.project_id, ids=existing.evidence_claim_ids
                    )
                ),
            )
        cached = (
            self.repository.latest_reusable_review(paper, hierarchical=hierarchical)
            if allow_reuse
            else None
        )
        self.repository.begin_checkpoint(run_id, "paper_review", paper.id)
        try:
            async with self.review_semaphore:
                chunks: tuple[DocumentChunk, ...] = ()
                download = self.repository.download_for_paper(paper.id)
                if (
                    download
                    and download.status is DownloadStatus.DOWNLOADED
                    and download.path
                ):
                    try:
                        chunks = await asyncio.to_thread(
                            lambda: chunk_document(
                                self.parser.parse_seed(download.path)
                            )
                        )
                    except Exception:
                        chunks = ()
                bundle = None
                if cached is not None:
                    cached_claims = (
                        tuple(
                            self.repository.claims(
                                paper.project_id, ids=cached.evidence_claim_ids
                            )
                        )
                        if cached.evidence_claim_ids
                        else ()
                    )
                    source = " ".join(
                        (
                            "\n\n".join(chunk.text for chunk in chunks)
                            or paper.abstract
                        ).split()
                    )
                    # Older releases accepted approximate excerpts. Verify cached
                    # quotes against today's source before avoiding a model call.
                    if cached_claims and all(
                        " ".join(claim.excerpt.split()) in source
                        for claim in cached_claims
                    ):
                        bundle = ReviewBundle(cached, cached_claims)
                if bundle is None:
                    if self.ensemble:
                        # Dual-leg parallel extraction with a merged result:
                        # wider evidence-span coverage at the same wall-clock
                        # as a single leg (the legs run concurrently under the
                        # same semaphore slot). One failing leg degrades to
                        # the survivor — an ensemble never makes a paper
                        # review harder to obtain than a single pass.
                        bundle = await self.extractor.extract_ensemble(
                            paper,
                            run_id=run_id,
                            chunks=chunks,
                            fallback_text=paper.abstract,
                            hierarchical=hierarchical,
                        )
                    else:
                        bundle = await self.extractor.extract(
                            paper,
                            run_id=run_id,
                            chunks=chunks,
                            fallback_text=paper.abstract,
                            hierarchical=hierarchical,
                        )
            for claim in bundle.claims:
                self.repository.save_claim(claim)
            self.repository.save_review(bundle.review)
            self.repository.clear_review_failures(run_id, paper.id)
            self.repository.complete_checkpoint(
                run_id,
                "paper_review",
                paper.id,
                payload={
                    "review_id": bundle.review.id,
                    "claim_ids": bundle.review.evidence_claim_ids,
                    "hierarchical": hierarchical,
                    "reused_from_run": bundle.review.run_id
                    if bundle.review.run_id != run_id
                    else None,
                },
            )
            return bundle
        except BaseException as exc:
            self.repository.fail_checkpoint(
                run_id,
                "paper_review",
                paper.id,
                error=redact_text(f"{type(exc).__name__}: {exc}"),
            )
            raise

    def _completed_outcome(self, project_id: str, run_id: str) -> ReviewOutcome:
        gap = self.repository.gap_for_run(run_id)
        artifacts = self.repository.artifacts_for_run(run_id)
        json_artifacts = [
            item for item in artifacts if item.artifact_type == "gap_report_json"
        ]
        markdown_artifacts = [
            item for item in artifacts if item.artifact_type == "gap_report_markdown"
        ]
        if not gap or not json_artifacts or not markdown_artifacts:
            raise RuntimeError(
                f"Completed review run '{run_id}' has incomplete artifacts."
            )
        scores = self.repository.scores(project_id, run_id)
        reviews = self.repository.reviews(project_id, run_id)
        failures = self.repository.review_failures(run_id)
        run = self.repository.get_run(run_id)
        top_k = int(run.input_data.get("top_k", 5)) if run else 5
        return ReviewOutcome(
            project_id,
            run_id,
            len(reviews),
            len(failures),
            tuple(score.paper_id for score in scores[:top_k]),
            gap,
            json_artifacts[-1].path,
            markdown_artifacts[-1].path,
        )


class _ExecutionGuard:
    def __init__(self, max_runtime_seconds: float | None) -> None:
        if max_runtime_seconds is not None and max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive.")
        self.max_runtime_seconds = max_runtime_seconds
        self.started = time.monotonic()

    def require_allowed(self, operation: str) -> None:
        if self.max_runtime_seconds is None:
            return
        elapsed = time.monotonic() - self.started
        if elapsed >= self.max_runtime_seconds:
            raise TimeoutError(
                f"Runtime limit of {self.max_runtime_seconds:g}s reached before {operation}."
            )


_TRANSIENT_KINDS = frozenset(
    {ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT, ErrorKind.UNAVAILABLE, ErrorKind.SERVER}
)


async def _with_transient_retry(operation, *, operation_name: str, attempts: int = 2, delay: float = 1.0):
    """Retry a model-backed operation only on errors a retry plausibly fixes
    (throttles, deadlines, outages). Hard errors re-raise immediately — this
    never masks a dead credential or an exhausted budget."""
    for attempt in range(attempts):
        try:
            return await operation()
        except BaseException as exc:
            if not _is_transient_error(exc) or attempt == attempts - 1:
                raise
            logger.warning("Transient failure in %s (attempt %d/%d): %s", operation_name, attempt + 1, attempts, type(exc).__name__)
            await asyncio.sleep(delay)
    raise AssertionError("unreachable")


def _is_transient_error(exc: BaseException) -> bool:
    """Errors a retry plausibly fixes: throttles, deadlines, outages."""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    return isinstance(exc, ProviderError) and exc.kind in _TRANSIENT_KINDS


def _is_hard_error(exc: BaseException) -> bool:
    """Errors that will strike EVERY remaining call identically — a dead
    credential, an exhausted budget, a malformed contract. Retrying (or
    letting the rest of the gather grind through N identical failures) is
    waste: fail fast so the user sees the real cause immediately and the
    completed paper checkpoints stay reusable."""
    if isinstance(exc, PermissionError):
        return True
    return isinstance(exc, ProviderError) and exc.kind not in _TRANSIENT_KINDS and exc.kind is not ErrorKind.UNKNOWN


def _parsed_payload(document: ParsedDocument) -> dict:
    return {
        "input_type": document.input_type,
        "input_value": document.input_value,
        "title": document.title,
        "pages": [asdict(page) for page in document.pages],
        "checksum": document.checksum,
        "doi": document.doi,
        "url": document.url,
    }


def _parsed_document(payload: dict) -> ParsedDocument:
    return ParsedDocument(
        str(payload["input_type"]),
        str(payload["input_value"]),
        str(payload["title"]),
        tuple(
            ParsedPage(int(page["page"]), str(page["text"]))
            for page in payload.get("pages") or []
        ),
        str(payload["checksum"]),
        str(payload["doi"]) if payload.get("doi") else None,
        str(payload["url"]) if payload.get("url") else None,
    )


def _seed_analysis(payload: dict) -> SeedAnalysis:
    return SeedAnalysis(
        str(payload["title"]),
        str(payload["topic"]),
        str(payload["research_question"]),
        tuple(str(item) for item in payload.get("methods") or []),
        tuple(str(item) for item in payload.get("datasets") or []),
        tuple(str(item) for item in payload.get("citations") or []),
        tuple(str(item) for item in payload.get("limitations") or []),
        tuple(str(item) for item in payload.get("keywords") or []),
    )


class NoveltyService:
    def __init__(
        self,
        repository: ResearchRepository,
        router: ModelRouter,
        search_pipeline: StudySearchPipeline,
    ) -> None:
        self.repository = repository
        self.router = router
        self.findings = FindingsMemory(repository)
        self.pipeline = NoveltyPipeline(
            repository=repository,
            generator=LLMCandidateGenerator(router),
            prior_art_checker=EvidenceBackedPriorArtChecker(router, search_pipeline),
            critic=ScientificCritic(router),
            selector=PortfolioSelector(repository),
        )

    async def import_gap(
        self, project_id: str, gap_data: dict[str, Any]
    ) -> ResearchGap:
        """Import a research gap from external data (e.g., from a prior literature review).

        The gap row references pipeline_runs, so the import gets its own
        completed run instead of a dangling run_id that would violate the
        foreign key (research_gaps.run_id REFERENCES pipeline_runs(id)).
        """
        statement = " ".join(
            str(
                gap_data.get("statement") or gap_data.get("gap_statement") or ""
            ).split()
        )
        if not statement:
            raise ValueError(
                "Gap data must include a 'statement' or 'gap_statement' field."
            )
        why_unsolved = " ".join(
            str(
                gap_data.get("why_unsolved")
                or gap_data.get("why_existing_work_does_not_solve_it")
                or ""
            ).split()
        )
        missing_experiment = " ".join(
            str(gap_data.get("missing_experiment") or "").split()
        )
        falsification_test = " ".join(
            str(gap_data.get("falsification_test") or "").split()
        )
        confidence = float(gap_data.get("confidence", 0.5))
        classification = (
            GapClassification.EVIDENCE_SUPPORTED
            if gap_data.get("supporting_paper_ids")
            else GapClassification.HYPOTHESIS
        )
        run = self.repository.create_run(
            project_id, PipelineKind.REVIEW, {"kind": "gap_import"}
        )
        self.repository.finish_run(run.id)
        provenance = ArtifactProvenance.create(
            project_id=project_id,
            run_id=run.id,
            source_evidence=(),
            provider="manual_import",
            model="user",
            prompt_version="imported-gap-v1",
            content={
                "statement": statement,
                "why_unsolved": why_unsolved,
                "missing_experiment": missing_experiment,
                "falsification_test": falsification_test,
            },
        )
        gap = ResearchGap(
            id=uuid4().hex,
            project_id=project_id,
            run_id=run.id,
            statement=statement,
            supporting_paper_ids=tuple(
                str(p) for p in (gap_data.get("supporting_paper_ids") or [])
            ),
            supporting_claim_ids=tuple(
                str(c) for c in (gap_data.get("supporting_claim_ids") or [])
            ),
            why_unsolved=why_unsolved,
            missing_experiment=missing_experiment,
            falsification_test=falsification_test,
            confidence=confidence,
            counter_evidence_claim_ids=(),
            classification=classification,
            provenance=provenance,
        )
        self.repository.save_gap(gap)
        return gap

    async def generate_portfolio(
        self,
        project_id: str,
        *,
        gap_id: str | None = None,
        run_id: str | None = None,
        max_runtime_seconds: float | None = None,
    ) -> str:
        if max_runtime_seconds is not None and max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive.")
        require_model_role(self.router, ModelRole.RESEARCH_REASONING)
        require_provider_credentials(self.router, ModelRole.RESEARCH_REASONING)
        gap = self.repository.row(
            "SELECT * FROM research_gaps WHERE project_id=? AND (? IS NULL OR id=?) ORDER BY confidence DESC LIMIT 1",
            (project_id, gap_id, gap_id),
        )
        if not gap:
            raise ValueError(f"No research gap found for project {project_id}.")

        # We need the full object, since gap is a dict right now (returned from row).
        # We can construct it, but easier to use a repository method if one existed,
        # or simply adapt it:
        gap_obj = ResearchGap(
            gap["id"],
            gap["project_id"],
            gap["run_id"],
            gap["statement"],
            tuple(json.loads(gap["supporting_paper_ids_json"])),
            tuple(json.loads(gap["supporting_claim_ids_json"])),
            gap["why_unsolved"],
            gap["missing_experiment"],
            gap["falsification_test"],
            float(gap["confidence"]),
            tuple(json.loads(gap["counter_evidence_claim_ids_json"])),
            GapClassification(gap["classification"]),
            ArtifactProvenance(**json.loads(gap["provenance_json"])),
        )

        run = (
            self.repository.get_run(run_id)
            if run_id
            else self.repository.create_run(
                project_id, PipelineKind.NOVELTY, {"gap_id": gap_obj.id}
            )
        )
        if not run:
            raise ValueError(f"Novelty run {run_id} not found.")
        owns_run = run_id is None
        try:
            # We mock the run_id inside gap so provenance links to this run.
            run_gap = ResearchGap(
                gap_obj.id,
                project_id,
                run.id,
                gap_obj.statement,
                gap_obj.supporting_paper_ids,
                gap_obj.supporting_claim_ids,
                gap_obj.why_unsolved,
                gap_obj.missing_experiment,
                gap_obj.falsification_test,
                gap_obj.confidence,
                gap_obj.counter_evidence_claim_ids,
                gap_obj.classification,
                gap_obj.provenance,
            )
            async with asyncio.timeout(max_runtime_seconds):
                # Findings Memory (DeepScientist mechanism, adapted): the
                # generator also sees prior outcomes for this question —
                # including refutations and implementation failures — so it
                # does not repropose what already failed under the same
                # conditions.
                findings = self.findings.context_block(
                    project_id, gap_obj.statement, question_id=gap_obj.id
                )
                portfolio = await self.pipeline.generate_portfolio(
                    run_gap, findings=findings
                )
            if owns_run:
                self.repository.finish_run(run.id)
            return portfolio.id
        except BaseException as exc:
            error = (
                f"runtime_limit: novelty exceeded {max_runtime_seconds:g}s"
                if isinstance(exc, TimeoutError) and max_runtime_seconds is not None
                else f"{type(exc).__name__}: {exc}"
            )
            if owns_run:
                if isinstance(exc, asyncio.CancelledError):
                    self.repository.cancel_run(run.id)
                else:
                    self.repository.finish_run(run.id, error=redact_text(error))
            if isinstance(exc, TimeoutError):
                raise TimeoutError(error) from exc
            raise


class ExperimentService:
    def __init__(
        self,
        repository: ResearchRepository,
        router: ModelRouter,
        workspace_root: Path,
        worker: LocalTypedWorker | None = None,
    ) -> None:
        self.repository = repository
        self.router = router
        self.findings = FindingsMemory(repository)
        self.pipeline = ExperimentPipeline(
            repository=repository,
            planner=ExperimentSpecGenerator(router),
            environment_builder=IsolatedEnvironmentBuilder(workspace_root),
            executor=DurableExperimentExecutor(repository, None),
            analyzer=StatisticalAnalyzer(router),
            report_writer=ScientificReportWriter(router),
        )
        self.worker = worker

    async def import_candidate(
        self, candidate_data: dict[str, Any]
    ) -> NoveltyCandidate:
        """Import a novelty candidate from external data (e.g., from a prior analysis)."""
        method = " ".join(
            str(
                candidate_data.get("method") or candidate_data.get("title") or ""
            ).split()
        )
        if not method:
            raise ValueError("Candidate data must include a 'method' or 'title' field.")
        project_id = str(candidate_data.get("project_id") or "")
        if not project_id:
            raise ValueError("Candidate data must include a 'project_id' field.")
        # The candidate row references pipeline_runs; a dangling run_id would
        # violate the foreign key, so imports get their own completed run.
        # A supplied run must belong to the same project and be a novelty run,
        # otherwise the candidate would silently attach to a foreign scope.
        supplied_run_id = str(candidate_data.get("run_id") or "")
        supplied_run = (
            self.repository.get_run(supplied_run_id) if supplied_run_id else None
        )
        if supplied_run is not None and (
            supplied_run.project_id != project_id
            or supplied_run.kind is not PipelineKind.NOVELTY
        ):
            raise ValueError(
                "run_mismatch: supplied run_id must reference a novelty run of the same project."
            )
        owns_run = supplied_run is None
        run_id = (
            supplied_run_id
            if supplied_run is not None
            else self.repository.create_run(
                project_id,
                PipelineKind.NOVELTY,
                {"kind": "candidate_import"},
            ).id
        )
        try:
            return self._save_imported_candidate(candidate_data, project_id, run_id)
        finally:
            if owns_run:
                self.repository.finish_run(run_id)

    def _save_imported_candidate(
        self, candidate_data: dict[str, Any], project_id: str, run_id: str
    ) -> NoveltyCandidate:
        method = " ".join(
            str(
                candidate_data.get("method") or candidate_data.get("title") or ""
            ).split()
        )
        if not method:
            raise ValueError("Candidate data must include a 'method' or 'title' field.")
        candidate = NoveltyCandidate(
            id=uuid4().hex,
            project_id=project_id,
            run_id=run_id,
            gap_id=str(candidate_data.get("gap_id") or ""),
            strategy=str(candidate_data.get("strategy") or "imported"),
            method=method,
            causal_mechanism=" ".join(
                str(
                    candidate_data.get("causal_mechanism")
                    or candidate_data.get("mechanism")
                    or ""
                ).split()
            ),
            target_limitation=" ".join(
                str(candidate_data.get("target_limitation") or "").split()
            ),
            difference_from_existing=" ".join(
                str(candidate_data.get("difference_from_existing") or "").split()
            ),
            expected_effect=str(candidate_data.get("expected_effect") or ""),
            falsification_condition=" ".join(
                str(candidate_data.get("falsification_condition") or "").split()
            ),
            minimum_validating_experiment=" ".join(
                str(candidate_data.get("minimum_validating_experiment") or "").split()
            ),
            required_ablations=tuple(
                str(item) for item in candidate_data.get("required_ablations") or []
            ),
            implementation_complexity=str(
                candidate_data.get("implementation_complexity") or "unknown"
            ),
            expected_compute=str(candidate_data.get("expected_compute") or "unknown"),
            risks_and_objections=tuple(
                str(item) for item in candidate_data.get("risks_and_objections") or []
            ),
            status=CandidateStatus.GENERATED,
            provenance=ArtifactProvenance.create(
                project_id=project_id,
                run_id=run_id,
                source_evidence=(),
                provider="manual_import",
                model="user",
                prompt_version="imported-candidate-v1",
                content={"method": method},
            ),
        )
        self.repository.save_novelty_candidate(candidate)
        return candidate

    def action_plan(self, spec: ExperimentSpec) -> tuple[ExecutionAction, ...]:
        if not spec.expected_outputs:
            raise ValueError("experiment_action_missing: expected_outputs is empty")
        scripts = [item for item in spec.expected_outputs if str(item).endswith(".py")]
        if scripts:
            return (ExecutionAction(ActionType.RUN_PYTHON_SCRIPT, str(scripts[0]), ()),)
        return (
            ExecutionAction(
                ActionType.RUN_PYTHON_SCRIPT, "scripts/run_experiment.py", ()
            ),
        )

    def _execution_worker(self, spec: ExperimentSpec) -> str:
        """Bind approval to the worker that will actually execute.

        ``spec.target_worker`` is an LLM-chosen preference and may name a worker
        that is not configured (e.g. a hallucinated GPU worker). The checksum
        must use the real worker identity, or approval never matches execution.
        """
        if self.worker is not None:
            return self.worker.worker_id
        return spec.target_worker or "local"

    def approval_preview(self, run_id: str) -> dict[str, object]:
        run = self.repository.get_run(run_id)
        if not run:
            raise ValueError(f"Experiment run {run_id} not found")
        spec = self.repository.row(
            "SELECT * FROM experiment_specs WHERE run_id=? ORDER BY rowid DESC LIMIT 1",
            (run_id,),
        )
        if not spec:
            raise ValueError(
                "approval_required: plan the experiment before approving execution"
            )
        from autoresearch.storage.repository import _experiment_spec

        typed_spec = _experiment_spec(spec)
        actions = self.action_plan(typed_spec)
        worker = self._execution_worker(typed_spec)
        checksum = specification_checksum(typed_spec, actions, worker)
        return {
            "run_id": run_id,
            "spec_id": typed_spec.id,
            "candidate_id": typed_spec.candidate_id,
            "checksum": checksum,
            "worker": worker,
            "actions": [
                {
                    "kind": action.kind.value,
                    "target": action.target,
                    "arguments": list(action.arguments),
                }
                for action in actions
            ],
            "budget": 0.0,
            "expires_at": None,
        }

    def approval_preview_for_candidate(self, candidate_id: str) -> dict[str, object]:
        spec = self.repository.get_experiment_spec(candidate_id)
        if not spec:
            raise ValueError(
                "approval_required: plan the experiment before approving execution"
            )
        actions = self.action_plan(spec)
        worker = self._execution_worker(spec)
        checksum = specification_checksum(spec, actions, worker)
        return {
            "run_id": spec.run_id,
            "spec_id": spec.id,
            "candidate_id": candidate_id,
            "checksum": checksum,
            "worker": worker,
            "actions": [
                {
                    "kind": action.kind.value,
                    "target": action.target,
                    "arguments": list(action.arguments),
                }
                for action in actions
            ],
            "budget": 3600.0,
            "expires_at": None,
        }

    def _methodology_for_spec(self, spec: ExperimentSpec) -> dict[str, Any] | None:
        """The frozen methodology record when it belongs to this spec's candidate."""
        freeze = self.repository.latest_methodology_freeze(spec.project_id)
        if freeze is None or freeze.candidate_id != spec.candidate_id:
            return None
        try:
            return json.loads(Path(freeze.artifact_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    async def implement_candidate(self, candidate_id: str) -> dict[str, Any]:
        """Generate and statically verify the experiment implementation for a
        planned spec (research code generation, AI-Researcher mechanism
        adapted). Reuses an existing script; no execution happens here — the
        approval gate is untouched."""
        from autoresearch.experiments.implementation import ImplementationGenerator

        spec = self.repository.get_experiment_spec(candidate_id)
        if not spec:
            raise ValueError(
                f"no_experiment_spec: plan the experiment for candidate {candidate_id} first"
            )
        require_provider_credentials(self.router, ModelRole.RESEARCH_REASONING)
        freeze = self.repository.latest_methodology_freeze(spec.project_id)
        methodology = self._methodology_for_spec(spec)
        findings = self.findings.context_block(spec.project_id, spec.hypothesis)
        generator = ImplementationGenerator(self.repository, self.router)
        record = await generator.implement(
            spec, methodology=methodology, findings=findings, freeze=freeze
        )

        already = any(
            f.spec_id == spec.id and f.status is FindingStatus.IMPLEMENTED_METHOD
            for f in self.repository.findings_for_project(spec.project_id, limit=200)
        )
        if record["fidelity"] == "STATIC_VERIFIED" and not already:
            self.repository.save_finding(self._implementation_finding(spec, record))
        elif record["status"] == "failed":
            self.findings.record_failure(
                spec,
                "implementation generation failed static verification: "
                + "; ".join(record["verification"]["reasons"][:3]),
                fidelity="STATIC",
            )
        self.repository.append_api_event(
            spec.run_id,
            "experiment.implementation",
            {
                "spec_id": spec.id,
                "status": record["status"],
                "fidelity": record["fidelity"],
                "mechanism_coverage": record["verification"]["mechanism_coverage"],
            },
        )
        return record

    def _implementation_finding(
        self, spec: ExperimentSpec, record: dict[str, Any]
    ) -> ResearchFinding:
        return ResearchFinding(
            id=uuid4().hex,
            project_id=spec.project_id,
            run_id=spec.run_id,
            status=FindingStatus.IMPLEMENTED_METHOD,
            question_id=str(spec.gap_id),
            candidate_id=spec.candidate_id,
            methodology_freeze_id=str(spec.methodology_freeze_id),
            spec_id=spec.id,
            analysis_id="",
            report_id="",
            hypothesis=spec.hypothesis,
            expected_observation="",
            implementation_digest=str(record.get("script_path") or ""),
            observed_outcome_json=json.dumps(
                {
                    "verification": record.get("verification", {}),
                }
            ),
            interpretation="",
            uncertainty="static verification only; not yet executed",
            fidelity="STATIC_VERIFIED",
            failure_classification="none",
            lesson=f"Implementation generated and statically verified for spec {spec.id} "
            f"(mechanism coverage {record['verification']['mechanism_coverage']:.0%}); "
            f"awaiting authorized execution.",
            applicability=f"dataset={spec.dataset_identity}; split={spec.split_definition}",
            open_questions_json="[]",
            next_action="Execute under the existing approval gate; then analysis updates this finding's status.",
            keywords_json=json.dumps(
                record["verification"].get("matched_components", [])[:8]
            ),
            provenance=ArtifactProvenance.create(
                project_id=spec.project_id,
                run_id=spec.run_id,
                source_evidence=[spec.id],
                provider="model",
                model="implementation-generator",
                prompt_version="findings-memory-v1",
                content={"status": "implemented_method", "spec": spec.id},
            ),
        )

    async def report_for_candidate(self, candidate_id: str) -> dict[str, Any]:
        """Complete the result path for an executed candidate: reuse the
        validated statistical analysis and write (or reuse) the deterministic
        report. Fails with insufficient_evidence when no validated analysis
        exists — never fabricates a decision from exit codes."""
        spec = self.repository.get_experiment_spec(candidate_id)
        if not spec:
            raise ValueError(
                "insufficient_evidence: no experiment spec exists for this candidate"
            )
        existing = self.repository.row(
            "SELECT * FROM experiment_reports WHERE run_id=? ORDER BY rowid DESC LIMIT 1",
            (spec.run_id,),
        )
        if existing:
            completeness = ExperimentMatrix.from_spec(spec).completeness(["main"])
            return {
                "report_id": existing["id"],
                "decision": existing["decision"],
                "decision_evidence": existing["decision_evidence"],
                "analysis_id": existing["analysis_id"],
                "spec_id": spec.id,
                "candidate_id": candidate_id,
                "experiment_completeness": completeness,
            }
        analysis = self.repository.get_statistical_analysis(spec.id)
        if not analysis:
            raise ValueError(
                "insufficient_evidence: no validated statistical analysis exists for this candidate; "
                "execute the approved experiment (measurement ingestion and analysis run automatically) "
                "or inspect the recorded integrity failure"
            )
        report = await self.pipeline.report_writer.write(spec, analysis)
        self.repository.save_experiment_report(report)
        completeness = ExperimentMatrix.from_spec(spec).completeness(["main"])
        return {
            "report_id": report.id,
            "decision": report.decision.value,
            "decision_evidence": report.decision_evidence,
            "analysis_id": analysis.id,
            "spec_id": spec.id,
            "candidate_id": candidate_id,
            "experiment_completeness": completeness,
        }

    async def execute_candidate(
        self,
        candidate_id: str,
        *,
        run_id: str | None = None,
        mode: str = "plan",
        max_runtime_seconds: float | None = None,
    ) -> str:
        if max_runtime_seconds is not None and max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive.")
        require_model_role(self.router, ModelRole.RESEARCH_REASONING)
        candidate_row = self.repository.row(
            "SELECT * FROM novelty_candidates WHERE id=?", (candidate_id,)
        )
        if not candidate_row:
            raise ValueError(f"Candidate {candidate_id} not found.")

        from autoresearch.storage.repository import _novelty_candidate

        candidate = _novelty_candidate(candidate_row)

        run = (
            self.repository.get_run(run_id)
            if run_id
            else self.repository.create_run(
                candidate.project_id,
                PipelineKind.EXPERIMENT,
                {"candidate_id": candidate_id},
            )
        )
        if not run:
            raise ValueError(f"Experiment run {run_id} not found.")
        owns_run = run_id is None
        try:
            # mock candidate to use this run_id for tracking
            from dataclasses import replace

            run_candidate = replace(candidate, run_id=run.id)
            if mode == "report":
                outcome = await self.report_for_candidate(candidate_id)
                if owns_run:
                    self.repository.finish_run(run.id)
                return str(outcome["spec_id"])
            # Inspection/validation must not generate a new plan or invalidate an approval.
            spec = (
                self.repository.get_experiment_spec(candidate_id)
                if mode in {"execute", "inspect", "validate"}
                else None
            )
            if spec is None:
                async with asyncio.timeout(max_runtime_seconds):
                    spec = await self.pipeline.plan_for_candidate(run_candidate)
                self.repository.save_experiment_spec(spec)
            if mode in {"inspect", "validate"}:
                manifest = self.repository.get_environment_manifest(spec.id)
                if manifest is None:
                    manifest = await self.pipeline.environment_builder.prepare(spec)
                    self.repository.save_environment_manifest(manifest)
            if mode == "execute":
                if not self.worker:
                    raise RuntimeError(
                        "worker_unavailable: no policy-controlled execution worker is configured"
                    )
                preview = self.approval_preview_for_candidate(candidate_id)
                approval = self.repository.approval_by_checksum(
                    str(preview["checksum"])
                )
                if not approval or approval.get("status") != "approved":
                    raise PermissionError(
                        "approval_required: exact experiment approval is required before execution"
                    )
                from datetime import datetime, timezone

                if approval.get("expires_at") and datetime.fromisoformat(
                    str(approval["expires_at"])
                ) < datetime.now(timezone.utc):
                    raise PermissionError(
                        "approval_expired: experiment approval has expired"
                    )
                from autoresearch.application.progress import report as _progress

                _progress("experiment.worker", 0, 4, detail="approved execution starting", remaining=["execution", "ingestion", "analysis", "report"])
                result = await self.worker.run(
                    spec,
                    self.action_plan(spec),
                    approved_checksum=str(preview["checksum"]),
                    timeout_seconds=int(approval.get("approved_budget") or 3600),
                )
                _progress("experiment.worker", 1, 4, detail=f"worker state: {result.get('state')}", remaining=["ingestion", "analysis", "report"])
                if result.get("state") != "COMPLETED":
                    raise RuntimeError(
                        "execution_failed: worker did not complete the approved scientific action"
                    )
                # Complete the result path: ingest the measured artifacts the
                # implementation emitted, run the deterministic statistical
                # analysis, and write the fail-closed report. Any integrity
                # failure is recorded and leaves the honest pending state —
                # a completed command never implies a scientific result.
                try:
                    analysis = self.repository.get_statistical_analysis(spec.id)
                    if analysis is None:
                        from autoresearch.experiments.ingestion import (
                            MeasurementIngester,
                        )

                        ingested = MeasurementIngester(self.repository).ingest(
                            spec, result, job_id=str(result.get("job_id"))
                        )
                        _progress(
                            "experiment.worker", 2, 4,
                            detail="measurements ingested and identity-checked",
                            remaining=["analysis", "report"],
                        )
                        analysis = await self.pipeline.analyzer.analyze(spec, ingested)
                        self.repository.save_statistical_analysis(analysis)
                    _progress(
                        "experiment.worker", 3, 4,
                        detail="deterministic statistical analysis complete",
                        remaining=["report"],
                    )
                    report = await self.report_for_candidate(candidate_id)
                    _progress(
                        "experiment.worker", 4, 4,
                        detail="report written; result path complete",
                        remaining=[],
                    )
                    self.repository.append_api_event(
                        run.id,
                        "experiment.analyzed",
                        {
                            "job_id": result.get("job_id"),
                            "analysis_id": analysis.id,
                            "report_id": report.get("report_id"),
                            "decision": report.get("decision"),
                        },
                    )
                    # Persist the outcome to the shared Findings Memory so later
                    # novelty/review stages retrieve it (incl. failures). Bookkeeping
                    # must not fail the scientific path.
                    try:
                        report_row = (
                            self.repository.row(
                                "SELECT * FROM experiment_reports WHERE id=?",
                                (str(report.get("report_id")),),
                            )
                            or {}
                        )
                        methodology = self._methodology_for_spec(spec)
                        finding = self.findings.record_from_analysis(
                            spec, analysis, report_row, methodology=methodology
                        )
                        self.repository.append_api_event(
                            run.id,
                            "findings.recorded",
                            {
                                "finding_id": finding.id,
                                "status": finding.status.value,
                            },
                        )
                    except Exception as exc:  # noqa: BLE001 — memory bookkeeping only
                        logger.warning(
                            "Findings recording failed for spec %s: %s", spec.id, exc
                        )
                except (IntegrityFailure, ValueError) as exc:
                    self.repository.append_api_event(
                        run.id,
                        "experiment.execution_completed",
                        {
                            "job_id": result.get("job_id"),
                            "scientific_result": "INSUFFICIENT_EVIDENCE",
                            "reason": redact_text(f"{type(exc).__name__}: {exc}"),
                        },
                    )
                    try:
                        self.findings.record_failure(
                            spec, f"{type(exc).__name__}: {exc}"
                        )
                    except Exception:  # memory bookkeeping only
                        logger.warning(
                            "Failure-finding recording failed for spec %s",
                            spec.id,
                            exc_info=True,
                        )
            if owns_run:
                self.repository.finish_run(run.id)
            return spec.id
        except BaseException as exc:
            error = (
                f"runtime_limit: experiment planning exceeded {max_runtime_seconds:g}s"
                if isinstance(exc, TimeoutError) and max_runtime_seconds is not None
                else f"{type(exc).__name__}: {exc}"
            )
            if owns_run:
                if isinstance(exc, asyncio.CancelledError):
                    self.repository.cancel_run(run.id)
                else:
                    self.repository.finish_run(run.id, error=redact_text(error))
            if isinstance(exc, TimeoutError):
                raise TimeoutError(error) from exc
            raise
