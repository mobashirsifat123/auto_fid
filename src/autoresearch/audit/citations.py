"""Citation-resolution auditor (trust-audit metric, DESIGN §5.1).

Resolves every bibliography entry against Semantic Scholar, OpenAlex and
Crossref through the checksummed fetch cache; arXiv is deliberately excluded
from resolution sources (search tarpit, LESSONS 2026-09-22 — the three
chosen sources already index arXiv content). Verdicts are deterministic with
frozen thresholds:

* ``RESOLVED_EXACT`` — confident title match (token-set Jaccard on
  ``papers.deduplication.normalized_title`` ≥ 0.85) in ≥1 source and no field
  mismatch in any source (family-name overlap ≥ 0.5, year ±1, venue
  alias-compatible).
* ``RESOLVED_MISATTRIBUTED`` — strong title match but ≥1 corroborated field
  mismatch. Accusation requires ≥2-source corroboration OR a DOI-title
  conflict (single-source sufficient) — risk R1 mitigation, so a single
  noisy source can never flag a baseline as misattributed.
* ``UNRESOLVABLE`` — no confident match (``no_match`` / ``no_confident_match``)
  or a defective entry (``unparseable``).

Harness-side failures stay fail-closed (§4.3): a source that errors is
recorded and excluded from pooling; an entry whose every source call failed
is *skipped* (never scored against the audited paper); if every call fails
the whole metric is ``NOT_RUN``. A fetch-cache miss in replay mode is also
``NOT_RUN``. This module intentionally defines its own lightweight
``BibEntryRecord``/``BundleHandle`` mirroring the documented schemas; the
integrator maps them onto the landed ``audit.contract`` types.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

import httpx

from autoresearch.papers.deduplication import normalized_title
from autoresearch.papers.metadata_sources import (
    CrossrefSource,
    MetadataPaper,
    OpenAlexSource,
    SemanticScholarSource,
    SourceSearchResult,
    normalize_arxiv_id,
    normalize_doi,
)

from .fetchcache import FetchCache, FetchCacheMiss, FetchCacheTransport

TITLE_MATCH_THRESHOLD = 0.85
TITLE_AMBIGUOUS_THRESHOLD = 0.70
YEAR_TOLERANCE = 1
AUTHOR_FAMILY_OVERLAP_MIN = 0.5
CORROBORATING_SOURCES_MIN = 2
SEARCH_LIMIT = 25
TITLE_CANDIDATE_REFINE_THRESHOLD = 20

#: Small frozen venue alias table (R1 mitigation): canonical token form for
#: common proceedings/journal nicknames before compatibility comparison.
_VENUE_ALIASES: dict[str, str] = {
    "nips": "neurips",
    "advances in neural information processing systems": "neurips",
    "neural information processing systems": "neurips",
    "iclr": "international conference on learning representations",
    "icml": "international conference on machine learning",
    "acl": "annual meeting of the association for computational linguistics",
    "emnlp": "empirical methods in natural language processing",
    "corr": "arxiv",
    "arxiv org": "arxiv",
    "pmlr": "proceedings of machine learning research",
}


class CitationVerdict(StrEnum):
    RESOLVED_EXACT = "RESOLVED_EXACT"
    RESOLVED_MISATTRIBUTED = "RESOLVED_MISATTRIBUTED"
    UNRESOLVABLE = "UNRESOLVABLE"


class CitationSubreason(StrEnum):
    NONE = "none"
    NO_MATCH = "no_match"
    NO_CONFIDENT_MATCH = "no_confident_match"
    UNPARSEABLE = "unparseable"
    DOI_TITLE_CONFLICT = "doi_title_conflict"


class CitationStatus(StrEnum):
    RUN_COMPLETE = "RUN_COMPLETE"
    NOT_RUN = "NOT_RUN"


@dataclass(frozen=True)
class BibEntryRecord:
    """Minimal normalized bibliography entry (maps to contract.BibEntry §8.2)."""

    key: str
    title: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    venue: str = ""
    doi: str | None = None
    arxiv_id: str | None = None
    entry_type: str = "article"
    unparseable: bool = False


@dataclass(frozen=True)
class BundleHandle:
    """Minimal paper-bundle handle carrying the bibliography to audit."""

    system_id: str
    bibliography: tuple[BibEntryRecord, ...] = ()


@dataclass(frozen=True)
class CitationQueryRecord:
    """Evidence for one source consultation (§8.2 citation detail queries)."""

    source: str
    kind: str
    query: str
    candidates: int = 0
    raw_checksum: str | None = None
    fetched_at: str | None = None
    cache_hit: bool | None = None
    error: str | None = None


@dataclass(frozen=True)
class MatchedRecord:
    """The best confident match across sources (§8.2 matched)."""

    source: str
    external_id: str
    doi: str | None
    title: str
    title_similarity: float
    via: str


@dataclass(frozen=True)
class FieldMismatch:
    """One field contradiction, with the sources that corroborate it."""

    field: str
    claimed: Any
    found: tuple[Any, ...]
    corroborating_sources: tuple[str, ...]


@dataclass(frozen=True)
class CitationEntryResult:
    """Per-entry verdict plus full query evidence."""

    bib_key: str
    raw: BibEntryRecord
    verdict: CitationVerdict
    subreason: CitationSubreason
    queries: tuple[CitationQueryRecord, ...] = ()
    matched: MatchedRecord | None = None
    mismatches: tuple[FieldMismatch, ...] = ()
    corroborated: bool = False
    skipped: bool = False


@dataclass(frozen=True)
class CitationAuditResult:
    """Per-paper citation metric output designed to feed report.py."""

    status: CitationStatus
    not_run_reason: str | None
    entries: tuple[CitationEntryResult, ...]
    counts: dict[str, int]
    source_error_counts: dict[str, int]

    def coverage(self) -> dict[str, Any]:
        skipped = sum(1 for entry in self.entries if entry.skipped)
        return {
            "items_examined": self.counts["total_entries"],
            "items_skipped": skipped,
            "skip_reasons": {"source_error": skipped},
        }

    def detail_records(self) -> list[dict[str, Any]]:
        """§8.2-shaped citation detail dicts (JSON-serializable)."""
        return [_entry_detail(entry) for entry in self.entries]


class CitationMetadataSource(Protocol):
    """Metadata source with the additive direct-lookup capability."""

    name: str

    async def search(self, query: str, *, limit: int, since_year: int | None = None) -> SourceSearchResult: ...

    async def lookup(self, identifier: str) -> MetadataPaper | None: ...

    async def close(self) -> None: ...


@dataclass(frozen=True)
class _SourceMatch:
    source: str
    paper: MetadataPaper
    similarity: float
    via: str


@dataclass(frozen=True)
class _SourceConflict:
    source: str
    paper: MetadataPaper


@dataclass
class _MismatchDraft:
    claimed: Any
    found: dict[Any, list[str]]


class CitationAuditor:
    """Deterministic citation-resolution auditor (§5.1) over the fetch cache."""

    def __init__(self, sources: Iterable[CitationMetadataSource], cache: FetchCache) -> None:
        self._sources = tuple(sources)
        self._cache = cache
        names = [source.name for source in self._sources]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate source names in citation sources: {sorted(names)}")

    async def audit(self, bundle: BundleHandle) -> CitationAuditResult:
        entries: list[CitationEntryResult] = []
        try:
            for bib in bundle.bibliography:
                entries.append(await self._audit_entry(bib))
        except FetchCacheMiss as miss:
            return self._finalize(tuple(entries), not_run_reason=f"fetch_cache_miss: {miss}")
        attempted = sum(1 for entry in entries for query in entry.queries)
        errored = sum(1 for entry in entries for query in entry.queries if query.error)
        if attempted > 0 and errored == attempted:
            return self._finalize(tuple(entries), not_run_reason="all_source_calls_failed")
        return self._finalize(tuple(entries), not_run_reason=None)

    async def _audit_entry(self, bib: BibEntryRecord) -> CitationEntryResult:
        queries: list[CitationQueryRecord] = []
        matches: dict[str, _SourceMatch] = {}
        conflicts: list[_SourceConflict] = []
        ambiguous_seen = False
        attempted = 0
        succeeded = 0
        bib_tokens = _title_tokens(bib.title)
        identifier = _entry_identifier(bib)

        if bib.unparseable or (not bib_tokens and not identifier):
            return CitationEntryResult(
                bib_key=bib.key, raw=bib,
                verdict=CitationVerdict.UNRESOLVABLE, subreason=CitationSubreason.UNPARSEABLE,
            )

        # Identifier fast path (§5.1 step 1): direct lookup by DOI (all three
        # sources) or arXiv id (SemanticScholar). Sources that do not support
        # the identifier type return None without a fetch.
        if identifier:
            for source in self._sources:
                attempted += 1
                query, paper, ok = await self._lookup(source, identifier)
                queries.append(query)
                if not ok:
                    continue
                succeeded += 1
                if paper is None:
                    continue
                similarity = _title_similarity(bib.title, paper.title)
                if not bib_tokens or similarity >= TITLE_MATCH_THRESHOLD:
                    matches[source.name] = _SourceMatch(source.name, paper, similarity, "identifier")
                elif similarity < TITLE_AMBIGUOUS_THRESHOLD:
                    conflicts.append(_SourceConflict(source.name, paper))
                else:
                    ambiguous_seen = True

        # Title query path (§5.1 step 2): only when fewer than two sources
        # already produced a confident identifier match (two independent
        # matches are enough for every corroboration decision). Conflict and
        # already-matched sources are not re-queried.
        conflict_names = {conflict.source for conflict in conflicts}
        if len(matches) < CORROBORATING_SOURCES_MIN:
            for source in self._sources:
                if source.name in matches or source.name in conflict_names:
                    continue
                attempted += 1
                query, result, ok = await self._search(source, bib.title, "title_search")
                queries.append(query)
                if not ok:
                    continue
                succeeded += 1
                best = _best_candidate(bib.title, result) if result is not None else None
                if best is not None and TITLE_AMBIGUOUS_THRESHOLD <= best[0] < TITLE_MATCH_THRESHOLD:
                    ambiguous_seen = True
                if (best is None or best[0] < TITLE_MATCH_THRESHOLD) and result is not None and bib.authors:
                    refined_query = f"{bib.title} {bib.authors[0]}"
                    if len(result.papers) > TITLE_CANDIDATE_REFINE_THRESHOLD:
                        attempted += 1
                        refined_record, refined_result, refined_ok = await self._search(source, refined_query, "author_refined_search")
                        queries.append(refined_record)
                        if refined_ok:
                            succeeded += 1
                            refined_best = _best_candidate(bib.title, refined_result) if refined_result is not None else None
                            if refined_best is not None and (best is None or refined_best[0] > best[0]):
                                best = refined_best
                if best is not None and best[0] >= TITLE_MATCH_THRESHOLD:
                    matches[source.name] = _SourceMatch(source.name, best[1], best[0], "title")

        mismatch_fields: dict[str, _MismatchDraft] = {}
        for conflict in conflicts:
            _add_mismatch(mismatch_fields, "title", bib.title, conflict.paper.title, conflict.source)
        for match in matches.values():
            _compare_fields(bib, match, mismatch_fields)
        mismatches = tuple(
            FieldMismatch(
                field=field,
                claimed=draft.claimed,
                found=tuple(sorted(draft.found, key=str)),
                corroborating_sources=tuple(sorted({name for names in draft.found.values() for name in names})),
            )
            for field, draft in mismatch_fields.items()
        )
        corroborated = any(len(mismatch.corroborating_sources) >= CORROBORATING_SOURCES_MIN for mismatch in mismatches)

        best_match = max(matches.values(), key=lambda match: match.similarity) if matches else None
        if conflicts:
            verdict, subreason = CitationVerdict.RESOLVED_MISATTRIBUTED, CitationSubreason.DOI_TITLE_CONFLICT
        elif matches:
            if corroborated:
                verdict, subreason = CitationVerdict.RESOLVED_MISATTRIBUTED, CitationSubreason.NONE
            elif mismatches:
                verdict, subreason = CitationVerdict.UNRESOLVABLE, CitationSubreason.NO_CONFIDENT_MATCH
            else:
                verdict, subreason = CitationVerdict.RESOLVED_EXACT, CitationSubreason.NONE
        elif ambiguous_seen:
            verdict, subreason = CitationVerdict.UNRESOLVABLE, CitationSubreason.NO_CONFIDENT_MATCH
        else:
            verdict, subreason = CitationVerdict.UNRESOLVABLE, CitationSubreason.NO_MATCH
        matched = (
            MatchedRecord(
                source=best_match.source, external_id=best_match.paper.external_id, doi=best_match.paper.doi,
                title=best_match.paper.title, title_similarity=best_match.similarity, via=best_match.via,
            )
            if best_match is not None
            else None
        )
        return CitationEntryResult(
            bib_key=bib.key, raw=bib, verdict=verdict, subreason=subreason,
            queries=tuple(queries), matched=matched, mismatches=mismatches,
            corroborated=corroborated, skipped=attempted > 0 and succeeded == 0,
        )

    async def _lookup(self, source: CitationMetadataSource, identifier: str) -> tuple[CitationQueryRecord, MetadataPaper | None, bool]:
        before = len(self._cache.accesses)
        try:
            paper = await source.lookup(identifier)
        except FetchCacheMiss:
            raise
        except Exception as error:  # noqa: BLE001 - a failing source must degrade to recorded NOT_RUN evidence, never crash the audit
            return (
                CitationQueryRecord(source=source.name, kind="identifier_lookup", query=identifier, error=type(error).__name__),
                None,
                False,
            )
        accesses = self._cache.accesses[before:]
        access = accesses[-1] if accesses else None
        return (
            CitationQueryRecord(
                source=source.name, kind="identifier_lookup", query=identifier, candidates=1 if paper is not None else 0,
                raw_checksum=access.raw_checksum if access else None,
                fetched_at=access.fetched_at if access else None,
                cache_hit=access.cache_hit if access else None,
            ),
            paper,
            True,
        )

    async def _search(self, source: CitationMetadataSource, query: str, kind: str) -> tuple[CitationQueryRecord, SourceSearchResult | None, bool]:
        before = len(self._cache.accesses)
        try:
            result = await source.search(query, limit=SEARCH_LIMIT)
        except FetchCacheMiss:
            raise
        except Exception as error:  # noqa: BLE001 - a failing source must degrade to recorded NOT_RUN evidence, never crash the audit
            return (
                CitationQueryRecord(source=source.name, kind=kind, query=query, error=type(error).__name__),
                None,
                False,
            )
        accesses = self._cache.accesses[before:]
        access = accesses[-1] if accesses else None
        return (
            CitationQueryRecord(
                source=source.name, kind=kind, query=query, candidates=len(result.papers),
                raw_checksum=access.raw_checksum if access else result.raw_checksum,
                fetched_at=access.fetched_at if access else None,
                cache_hit=access.cache_hit if access else None,
            ),
            result,
            True,
        )

    def _finalize(self, entries: tuple[CitationEntryResult, ...], *, not_run_reason: str | None) -> CitationAuditResult:
        examined = [entry for entry in entries if not entry.skipped]
        counts = {
            "total_entries": len(examined),
            "resolved_exact": sum(1 for entry in examined if entry.verdict == CitationVerdict.RESOLVED_EXACT),
            "resolved_misattributed": sum(1 for entry in examined if entry.verdict == CitationVerdict.RESOLVED_MISATTRIBUTED),
            "misattributed_corroborated_2plus": sum(
                1 for entry in examined
                if entry.verdict == CitationVerdict.RESOLVED_MISATTRIBUTED and entry.corroborated
            ),
            "unresolvable": sum(1 for entry in examined if entry.verdict == CitationVerdict.UNRESOLVABLE),
            "unresolvable_no_match": sum(
                1 for entry in examined
                if entry.verdict == CitationVerdict.UNRESOLVABLE and entry.subreason == CitationSubreason.NO_MATCH
            ),
            "unresolvable_no_confident_match": sum(
                1 for entry in examined
                if entry.verdict == CitationVerdict.UNRESOLVABLE and entry.subreason == CitationSubreason.NO_CONFIDENT_MATCH
            ),
            "unresolvable_unparseable": sum(
                1 for entry in examined
                if entry.verdict == CitationVerdict.UNRESOLVABLE and entry.subreason == CitationSubreason.UNPARSEABLE
            ),
            "doi_title_conflicts": sum(
                1 for entry in examined if entry.subreason == CitationSubreason.DOI_TITLE_CONFLICT
            ),
            "skipped_source_error": len(entries) - len(examined),
        }
        error_counts: dict[str, int] = {}
        for entry in entries:
            for query in entry.queries:
                if query.error:
                    error_counts[query.source] = error_counts.get(query.source, 0) + 1
        status = CitationStatus.NOT_RUN if not_run_reason else CitationStatus.RUN_COMPLETE
        return CitationAuditResult(
            status=status, not_run_reason=not_run_reason, entries=entries,
            counts=counts, source_error_counts=error_counts,
        )


def build_lookup_sources(
    cache: FetchCache,
    *,
    inner: httpx.AsyncBaseTransport | None = None,
    requests_per_second: float = 2.0,
    timeout: float = 30.0,
) -> tuple[SemanticScholarSource, OpenAlexSource, CrossrefSource]:
    """Construct the three resolution sources wired through the fetch cache.

    ``inner`` is the transport used for live fetches (production:
    ``httpx.AsyncHTTPTransport()``; tests: ``httpx.MockTransport``). In replay
    mode (``cache.replay``) pass ``inner=None`` so cache misses fail closed
    instead of touching the network.
    """
    return (
        SemanticScholarSource(
            requests_per_second=requests_per_second, timeout=timeout,
            transport=FetchCacheTransport(cache, SemanticScholarSource.name, inner),
        ),
        OpenAlexSource(
            requests_per_second=requests_per_second, timeout=timeout,
            transport=FetchCacheTransport(cache, OpenAlexSource.name, inner),
        ),
        CrossrefSource(
            requests_per_second=requests_per_second, timeout=timeout,
            transport=FetchCacheTransport(cache, CrossrefSource.name, inner),
        ),
    )


def _entry_identifier(bib: BibEntryRecord) -> str | None:
    return normalize_doi(bib.doi) or (normalize_arxiv_id(bib.arxiv_id) if bib.arxiv_id else None)


def _title_tokens(value: str) -> frozenset[str]:
    return frozenset(normalized_title(value).split())


def _title_similarity(left: str, right: str) -> float:
    left_tokens, right_tokens = _title_tokens(left), _title_tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _best_candidate(title: str, result: SourceSearchResult) -> tuple[float, MetadataPaper] | None:
    best: tuple[float, MetadataPaper] | None = None
    for paper in result.papers:
        similarity = _title_similarity(title, paper.title)
        if best is None or similarity > best[0]:
            best = (similarity, paper)
    return best


def _family_names(authors: Iterable[str]) -> frozenset[str]:
    families: set[str] = set()
    for name in authors:
        text = " ".join(str(name or "").split())
        if not text:
            continue
        family = text.split(",")[0].strip() if "," in text else text.split()[-1]
        family = normalized_title(family)
        if family:
            families.add(family)
    return frozenset(families)


def _venue_canonical(value: str) -> str:
    normalized = normalized_title(value)
    return _VENUE_ALIASES.get(normalized, normalized)


def _venues_compatible(claimed: str, found: str) -> bool:
    left, right = _venue_canonical(claimed), _venue_canonical(found)
    if left == right:
        return True
    left_tokens, right_tokens = set(left.split()), set(right.split())
    return bool(left_tokens and right_tokens and (left_tokens <= right_tokens or right_tokens <= left_tokens))


def _compare_fields(bib: BibEntryRecord, match: _SourceMatch, fields: dict[str, _MismatchDraft]) -> None:
    paper = match.paper
    if bib.year is not None and paper.year is not None and abs(bib.year - paper.year) > YEAR_TOLERANCE:
        _add_mismatch(fields, "year", bib.year, paper.year, match.source)
    claimed_families = _family_names(bib.authors)
    found_families = _family_names(paper.authors)
    if claimed_families and found_families:
        overlap = len(claimed_families & found_families) / min(len(claimed_families), len(found_families))
        if overlap < AUTHOR_FAMILY_OVERLAP_MIN:
            _add_mismatch(fields, "authors", tuple(sorted(claimed_families)), tuple(sorted(found_families)), match.source)
    if bib.venue and paper.venue and not _venues_compatible(bib.venue, paper.venue):
        _add_mismatch(fields, "venue", bib.venue, paper.venue, match.source)


def _add_mismatch(fields: dict[str, _MismatchDraft], field: str, claimed: Any, found: Any, source: str) -> None:
    draft = fields.setdefault(field, _MismatchDraft(claimed=claimed, found={}))
    draft.found.setdefault(found, []).append(source)


def _entry_detail(entry: CitationEntryResult) -> dict[str, Any]:
    return {
        "bib_key": entry.bib_key,
        "raw": {
            "title": entry.raw.title,
            "authors": list(entry.raw.authors),
            "year": entry.raw.year,
            "venue": entry.raw.venue,
            "doi": entry.raw.doi,
            "arxiv_id": entry.raw.arxiv_id,
            "entry_type": entry.raw.entry_type,
        },
        "verdict": entry.verdict.value,
        "subreason": entry.subreason.value,
        "queries": [
            {
                "source": query.source,
                "kind": query.kind,
                "query": query.query,
                "candidates": query.candidates,
                "fetched_at": query.fetched_at,
                "raw_checksum": query.raw_checksum,
                "cache_hit": query.cache_hit,
                "error": query.error,
            }
            for query in entry.queries
        ],
        "matched": (
            {
                "source": entry.matched.source,
                "external_id": entry.matched.external_id,
                "doi": entry.matched.doi,
                "title": entry.matched.title,
                "title_similarity": entry.matched.title_similarity,
                "via": entry.matched.via,
            }
            if entry.matched is not None
            else None
        ),
        "mismatches": [
            {
                "field": mismatch.field,
                "claimed": _json_value(mismatch.claimed),
                "found": _json_value(mismatch.found[0]) if len(mismatch.found) == 1 else [_json_value(value) for value in mismatch.found],
                "corroborating_sources": list(mismatch.corroborating_sources),
            }
            for mismatch in entry.mismatches
        ],
        "corroborated": entry.corroborated,
        "skipped": entry.skipped,
    }


def _json_value(value: Any) -> Any:
    return list(value) if isinstance(value, tuple) else value
