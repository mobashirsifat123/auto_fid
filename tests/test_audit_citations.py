"""Hermetic citation-resolution tests: real sources + fake transport, no network.

Fixture bib entries cover: exact match (DOI present / absent), venue alias
compatibility, year tolerance, corroborated misattribution (year / venue /
authors), DOI-title conflict, arXiv fast path, ambiguous and unresolvable
titles, author-refined queries, unparseable entries, one source 500ing
(NOT_RUN-for-source semantics), all sources failing (metric NOT_RUN), and
replay determinism through the fetch cache.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from autoresearch.audit.citations import (
    BibEntryRecord,
    BundleHandle,
    CitationAuditor,
    CitationStatus,
    CitationSubreason,
    CitationVerdict,
    build_lookup_sources,
)
from autoresearch.audit.fetchcache import FetchCache
from autoresearch.domain.models import content_checksum
from autoresearch.papers.deduplication import normalized_title

S2_HOST = "api.semanticscholar.org"
OA_HOST = "api.openalex.org"
CR_HOST = "api.crossref.org"

PAPER_A = {
    "title": "Adaptive Fusion for Robust Multimodal Learning",
    "authors": ("Alice Chen", "Bob Diaz", "Carla Eng"),
    "year": 2020,
    "venue": "Advances in Neural Information Processing Systems",
    "doi": "10.1000/adaptive-fusion",
    "arxiv": "2401.00001",
}
PAPER_B = {
    "title": "Graph Contrastive Pretraining at Scale",
    "authors": ("Dana Flores",),
    "year": 2022,
    "venue": "International Conference on Learning Representations",
    "doi": "10.1000/graph-contrastive",
    "arxiv": None,
}
PAPER_C = {
    "title": "Sparse Mixture Routing Revisited Empirically",
    "authors": ("Grace Ho",),
    "year": 2019,
    "venue": "arXiv",
    "doi": None,
    "arxiv": "2305.04321",
}
PAPERS = (PAPER_A, PAPER_B, PAPER_C)
GENERIC_PILE = tuple(
    {"title": f"Generic Report {index}", "authors": (f"Author {index}",), "year": 2021, "venue": "Generic Venue", "doi": None, "arxiv": None}
    for index in range(25)
)

EXACT_DOI = BibEntryRecord(
    key="chen2020", title=PAPER_A["title"], authors=PAPER_A["authors"], year=2020,
    venue=PAPER_A["venue"], doi=PAPER_A["doi"],
)
EXACT_NO_DOI = BibEntryRecord(
    key="flores2022", title=PAPER_B["title"], authors=PAPER_B["authors"], year=2022,
    venue="ICLR",
)
YEAR_TOLERATED = BibEntryRecord(
    key="chen2020tol", title=PAPER_A["title"], authors=PAPER_A["authors"], year=2021,
    venue=PAPER_A["venue"], doi=PAPER_A["doi"],
)
MISATTR_YEAR = BibEntryRecord(
    key="chen2020wrong", title=PAPER_A["title"], authors=PAPER_A["authors"], year=2023,
    venue=PAPER_A["venue"], doi=PAPER_A["doi"],
)
WRONG_VENUE = BibEntryRecord(
    key="flores2022wrong", title=PAPER_B["title"], authors=PAPER_B["authors"], year=2022,
    venue="Nature Communications",
)
WRONG_AUTHORS = BibEntryRecord(
    key="flores2022auth", title=PAPER_B["title"], authors=("Eve Gartner",), year=2022, venue="ICLR",
)
FABRICATED = BibEntryRecord(
    key="ghost2024", title="Quantum Bayesian Hypergraph Fusion Dynamics",
    authors=("Mallory Fox",), year=2024,
)
AMBIGUOUS = BibEntryRecord(
    key="ho2019", title="Sparse Mixture Routing Revisited", authors=("Grace Ho",),
    year=2019, venue="CoRR",
)
DOI_CONFLICT = BibEntryRecord(
    key="chen2020conflict", title=PAPER_A["title"], authors=PAPER_A["authors"], year=2020,
    venue=PAPER_A["venue"], doi=PAPER_B["doi"],
)
ARXIV_FAST = BibEntryRecord(
    key="chen2020arx", title=PAPER_A["title"], authors=PAPER_A["authors"], year=2020,
    venue=PAPER_A["venue"], arxiv_id=PAPER_A["arxiv"],
)
VAGUE = BibEntryRecord(
    key="vague2021", title="Vague Topic", authors=("Zhang Wei",), year=2021,
)
UNPARSEABLE = BibEntryRecord(key="broken", unparseable=True)


def _jaccard(left: str, right: str) -> float:
    left_tokens, right_tokens = set(normalized_title(left).split()), set(normalized_title(right).split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens)


def _search_matches(query: str) -> tuple[dict, ...]:
    if "vague topic" in normalized_title(query):
        return GENERIC_PILE
    return tuple(paper for paper in PAPERS if _jaccard(paper["title"], query) >= 0.5)


def _s2_row(paper: dict) -> dict:
    return {
        "paperId": f"s2:{paper['doi'] or paper['arxiv'] or paper['title']}",
        "title": paper["title"],
        "abstract": "Abstract.",
        "authors": [{"name": name} for name in paper["authors"]],
        "year": paper["year"],
        "publicationDate": f"{paper['year']}-05-01",
        "externalIds": {"DOI": paper["doi"], "ArXiv": paper["arxiv"]},
        "url": "https://example.org/paper",
        "openAccessPdf": {"url": None},
        "isOpenAccess": True,
        "citationCount": 7,
        "venue": paper["venue"],
    }


def _oa_row(paper: dict) -> dict:
    return {
        "id": f"https://openalex.org/W{paper['title'][:4]}",
        "display_name": paper["title"],
        "authorships": [{"author": {"display_name": name}} for name in paper["authors"]],
        "publication_year": paper["year"],
        "publication_date": f"{paper['year']}-05-01",
        "ids": {"doi": f"https://doi.org/{paper['doi']}" if paper["doi"] else None, "arxiv": paper["arxiv"]},
        "primary_location": {"source": {"display_name": paper["venue"]}, "landing_page_url": "https://example.org/work", "pdf_url": None},
        "open_access": {"is_oa": True, "oa_url": None},
        "cited_by_count": 7,
        "abstract_inverted_index": None,
    }


def _cr_row(paper: dict) -> dict:
    return {
        "DOI": paper["doi"],
        "title": [paper["title"]],
        "author": [{"given": name.split()[0], "family": name.split()[-1]} for name in paper["authors"]],
        "published": {"date-parts": [[paper["year"], 5, 1]]},
        "URL": "https://example.org/work",
        "container-title": [paper["venue"]],
        "is-referenced-by-count": 7,
        "abstract": None,
        "link": [],
        "license": [],
    }


def _paper_by_doi(doi: str) -> dict | None:
    return next((paper for paper in PAPERS if paper["doi"] == doi), None)


def _paper_by_arxiv(arxiv: str) -> dict | None:
    return next((paper for paper in PAPERS if paper["arxiv"] == arxiv), None)


def _mock_transport(fail_hosts: frozenset[str] = frozenset()) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host or ""
        if host in fail_hosts:
            return httpx.Response(500, text="source exploded")
        params = dict(request.url.params)
        path = request.url.path
        if host == S2_HOST:
            path = path.removeprefix("/graph/v1")
        if host == S2_HOST:
            if path.startswith("/paper/DOI:"):
                paper = _paper_by_doi(path.removeprefix("/paper/DOI:"))
                return httpx.Response(200, json=_s2_row(paper)) if paper else httpx.Response(404, json={"error": "Paper not found"})
            if path.startswith("/paper/arXiv:"):
                paper = _paper_by_arxiv(path.removeprefix("/paper/arXiv:"))
                return httpx.Response(200, json=_s2_row(paper)) if paper else httpx.Response(404, json={"error": "Paper not found"})
            if path == "/paper/search":
                papers = [_s2_row(paper) for paper in _search_matches(params["query"])]
                return httpx.Response(200, json={"data": papers, "total": len(papers)})
        if host == OA_HOST:
            if path.startswith("/works/doi:"):
                paper = _paper_by_doi(path.removeprefix("/works/doi:"))
                return httpx.Response(200, json=_oa_row(paper)) if paper else httpx.Response(404, json={"error": "Not found"})
            if path == "/works":
                return httpx.Response(200, json={"results": [_oa_row(paper) for paper in _search_matches(params["search"])]})
        if host == CR_HOST:
            if path.startswith("/works/") and path != "/works":
                paper = _paper_by_doi(path.removeprefix("/works/"))
                return httpx.Response(200, json={"message": _cr_row(paper)}) if paper else httpx.Response(404, json={"status": "error"})
            if path == "/works":
                return httpx.Response(200, json={"message": {"items": [_cr_row(paper) for paper in _search_matches(params["query.bibliographic"])]}})
        return httpx.Response(404, json={"error": f"unexpected path {path}"})

    return httpx.MockTransport(handler)


async def _audit(
    entries: tuple[BibEntryRecord, ...],
    cache: FetchCache,
    *,
    fail_hosts: frozenset[str] = frozenset(),
):
    sources = build_lookup_sources(cache, inner=_mock_transport(fail_hosts), requests_per_second=1000.0)
    auditor = CitationAuditor(sources, cache)
    try:
        result = await auditor.audit(BundleHandle(system_id="fixture", bibliography=entries))
    finally:
        for source in sources:
            await source.close()
    return result


@pytest.fixture(autouse=True)
def _clean_source_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("SEMANTIC_SCHOLAR_API_KEY", "OPENALEX_MAILTO", "CROSSREF_MAILTO"):
        monkeypatch.delenv(name, raising=False)


async def test_exact_match_with_doi_uses_identifier_fast_path(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fetches.jsonl")
    result = await _audit((EXACT_DOI,), cache)
    assert result.status == CitationStatus.RUN_COMPLETE
    assert result.not_run_reason is None
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.RESOLVED_EXACT
    assert entry.subreason == CitationSubreason.NONE
    assert entry.mismatches == ()
    assert entry.matched is not None and entry.matched.via == "identifier"
    assert entry.matched.title == PAPER_A["title"]
    assert [query.kind for query in entry.queries] == ["identifier_lookup"] * 3
    assert all(query.candidates == 1 for query in entry.queries)
    assert all(query.error is None for query in entry.queries)
    assert all(query.raw_checksum for query in entry.queries)
    assert result.counts == {
        "total_entries": 1, "resolved_exact": 1, "resolved_misattributed": 0,
        "misattributed_corroborated_2plus": 0, "unresolvable": 0,
        "unresolvable_no_match": 0, "unresolvable_no_confident_match": 0,
        "unresolvable_unparseable": 0, "doi_title_conflicts": 0, "skipped_source_error": 0,
    }
    assert len((tmp_path / "fetches.jsonl").read_text(encoding="utf-8").splitlines()) == 3


async def test_exact_match_without_doi_and_venue_alias(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fetches.jsonl")
    result = await _audit((EXACT_NO_DOI,), cache)
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.RESOLVED_EXACT
    assert entry.subreason == CitationSubreason.NONE
    assert [query.kind for query in entry.queries] == ["title_search"] * 3
    assert entry.matched is not None and entry.matched.via == "title"
    assert entry.mismatches == ()


async def test_year_within_tolerance_is_exact(tmp_path: Path) -> None:
    result = await _audit((YEAR_TOLERATED,), FetchCache(None))
    assert result.entries[0].verdict == CitationVerdict.RESOLVED_EXACT


async def test_misattributed_year_requires_two_source_corroboration(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fetches.jsonl")
    result = await _audit((MISATTR_YEAR,), cache)
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.RESOLVED_MISATTRIBUTED
    assert entry.subreason == CitationSubreason.NONE
    assert entry.corroborated is True
    (mismatch,) = entry.mismatches
    assert mismatch.field == "year"
    assert mismatch.claimed == 2023
    assert mismatch.found == (2020,)
    assert len(mismatch.corroborating_sources) == 3
    assert result.counts["resolved_misattributed"] == 1
    assert result.counts["misattributed_corroborated_2plus"] == 1


async def test_single_source_mismatch_is_not_an_accusation(tmp_path: Path) -> None:
    result = await _audit(
        (MISATTR_YEAR,), FetchCache(None),
        fail_hosts=frozenset({OA_HOST, CR_HOST}),
    )
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.UNRESOLVABLE
    assert entry.subreason == CitationSubreason.NO_CONFIDENT_MATCH
    assert result.source_error_counts == {"openalex": 2, "crossref": 2}


async def test_wrong_venue_is_misattributed(tmp_path: Path) -> None:
    result = await _audit((WRONG_VENUE,), FetchCache(None))
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.RESOLVED_MISATTRIBUTED
    (mismatch,) = entry.mismatches
    assert mismatch.field == "venue"
    assert mismatch.claimed == "Nature Communications"
    assert mismatch.found == (PAPER_B["venue"],)
    assert len(mismatch.corroborating_sources) >= 2


async def test_wrong_authors_are_misattributed(tmp_path: Path) -> None:
    result = await _audit((WRONG_AUTHORS,), FetchCache(None))
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.RESOLVED_MISATTRIBUTED
    (mismatch,) = entry.mismatches
    assert mismatch.field == "authors"
    assert mismatch.claimed == ("gartner",)
    assert mismatch.found == (("flores",),)
    assert len(mismatch.corroborating_sources) >= 2


async def test_fabricated_title_is_unresolvable(tmp_path: Path) -> None:
    result = await _audit((FABRICATED,), FetchCache(None))
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.UNRESOLVABLE
    assert entry.subreason == CitationSubreason.NO_MATCH
    assert entry.matched is None
    assert all(query.candidates == 0 for query in entry.queries)
    assert len(entry.queries) == 3


async def test_ambiguous_title_is_unresolvable_no_confident_match(tmp_path: Path) -> None:
    result = await _audit((AMBIGUOUS,), FetchCache(None))
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.UNRESOLVABLE
    assert entry.subreason == CitationSubreason.NO_CONFIDENT_MATCH


async def test_doi_title_conflict_is_misattributed_single_source(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fetches.jsonl")
    result = await _audit((DOI_CONFLICT,), cache)
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.RESOLVED_MISATTRIBUTED
    assert entry.subreason == CitationSubreason.DOI_TITLE_CONFLICT
    (mismatch,) = entry.mismatches
    assert mismatch.field == "title"
    assert mismatch.claimed == PAPER_A["title"]
    assert mismatch.found == (PAPER_B["title"],)
    assert result.counts["doi_title_conflicts"] == 1


async def test_arxiv_identifier_fast_path(tmp_path: Path) -> None:
    result = await _audit((ARXIV_FAST,), FetchCache(None))
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.RESOLVED_EXACT
    kinds = [(query.source, query.kind) for query in entry.queries]
    assert ("semantic_scholar", "identifier_lookup") in kinds
    assert ("openalex", "identifier_lookup") in kinds and ("crossref", "identifier_lookup") in kinds
    assert ("openalex", "title_search") in kinds and ("crossref", "title_search") in kinds
    s2_lookup = next(query for query in entry.queries if query.source == "semantic_scholar" and query.kind == "identifier_lookup")
    assert s2_lookup.candidates == 1
    oa_lookup = next(query for query in entry.queries if query.source == "openalex" and query.kind == "identifier_lookup")
    assert oa_lookup.candidates == 0 and oa_lookup.raw_checksum is None


async def test_author_refined_query_when_title_is_vague(tmp_path: Path) -> None:
    result = await _audit((VAGUE,), FetchCache(None))
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.UNRESOLVABLE
    assert entry.subreason == CitationSubreason.NO_MATCH
    refined = [query for query in entry.queries if query.kind == "author_refined_search"]
    assert len(refined) == 3
    assert refined[0].query == "Vague Topic Zhang Wei"


async def test_unparseable_entry_counts_as_artifact_failure(tmp_path: Path) -> None:
    result = await _audit((UNPARSEABLE,), FetchCache(None))
    entry = result.entries[0]
    assert entry.verdict == CitationVerdict.UNRESOLVABLE
    assert entry.subreason == CitationSubreason.UNPARSEABLE
    assert entry.queries == ()
    assert entry.skipped is False
    assert result.counts["total_entries"] == 1
    assert result.counts["unresolvable_unparseable"] == 1


async def test_one_source_erroring_is_recorded_not_fatal(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fetches.jsonl")
    result = await _audit((EXACT_NO_DOI, MISATTR_YEAR), cache, fail_hosts=frozenset({CR_HOST}))
    assert result.status == CitationStatus.RUN_COMPLETE
    # Crossref errors once per entry (title search; the DOI entry reaches
    # two confident identifier matches, so the title phase is skipped).
    assert result.source_error_counts == {"crossref": 2}
    verdicts = [entry.verdict for entry in result.entries]
    assert verdicts == [CitationVerdict.RESOLVED_EXACT, CitationVerdict.RESOLVED_MISATTRIBUTED]
    crossref_queries = [query for entry in result.entries for query in entry.queries if query.source == "crossref"]
    assert all(query.error == "HTTPStatusError" for query in crossref_queries)
    assert all(query.raw_checksum is None for query in crossref_queries)
    misattributed = result.entries[1]
    assert sorted(misattributed.mismatches[0].corroborating_sources) == ["openalex", "semantic_scholar"]


async def test_all_sources_erroring_is_not_run(tmp_path: Path) -> None:
    result = await _audit(
        (EXACT_DOI, FABRICATED), FetchCache(None),
        fail_hosts=frozenset({S2_HOST, OA_HOST, CR_HOST}),
    )
    assert result.status == CitationStatus.NOT_RUN
    assert result.not_run_reason == "all_source_calls_failed"
    assert all(entry.skipped for entry in result.entries)
    assert result.coverage() == {"items_examined": 0, "items_skipped": 2, "skip_reasons": {"source_error": 2}}
    assert result.counts["total_entries"] == 0
    assert result.counts["skipped_source_error"] == 2
    assert set(result.source_error_counts) == {"semantic_scholar", "openalex", "crossref"}


async def test_replay_cache_miss_is_not_run(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fetches.jsonl", replay=True)
    sources = build_lookup_sources(cache, inner=None, requests_per_second=1000.0)
    auditor = CitationAuditor(sources, cache)
    result = await auditor.audit(BundleHandle(system_id="fixture", bibliography=(EXACT_DOI,)))
    for source in sources:
        await source.close()
    assert result.status == CitationStatus.NOT_RUN
    assert result.not_run_reason is not None and "fetch_cache_miss" in result.not_run_reason


async def test_replay_run_reproduces_live_verdicts_and_checksums(tmp_path: Path) -> None:
    path = tmp_path / "fetches.jsonl"
    entries = (EXACT_DOI, EXACT_NO_DOI, MISATTR_YEAR, DOI_CONFLICT, FABRICATED, UNPARSEABLE)
    live = await _audit(entries, FetchCache(path))
    replay_cache = FetchCache(path, replay=True)
    sources = build_lookup_sources(replay_cache, inner=None, requests_per_second=1000.0)
    replay = await CitationAuditor(sources, replay_cache).audit(BundleHandle(system_id="fixture", bibliography=entries))
    for source in sources:
        await source.close()

    assert replay.status == live.status == CitationStatus.RUN_COMPLETE
    assert [entry.verdict for entry in replay.entries] == [entry.verdict for entry in live.entries]
    assert [entry.subreason for entry in replay.entries] == [entry.subreason for entry in live.entries]
    assert [entry.mismatches for entry in replay.entries] == [entry.mismatches for entry in live.entries]
    assert replay.counts == live.counts
    live_checksums = [
        (query.source, query.kind, query.raw_checksum)
        for entry in live.entries for query in entry.queries if query.raw_checksum
    ]
    replay_checksums = [
        (query.source, query.kind, query.raw_checksum)
        for entry in replay.entries for query in entry.queries if query.raw_checksum
    ]
    assert live_checksums == replay_checksums
    assert all(query.cache_hit is True for entry in replay.entries for query in entry.queries if query.raw_checksum)


async def test_detail_records_are_json_safe_and_schema_shaped(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fetches.jsonl")
    result = await _audit((MISATTR_YEAR, FABRICATED), cache)
    details = result.detail_records()
    json.dumps(details)
    misattributed, unresolvable = details
    assert misattributed["verdict"] == "RESOLVED_MISATTRIBUTED"
    assert misattributed["subreason"] == "none"
    assert misattributed["raw"]["year"] == 2023
    assert misattributed["queries"][0]["raw_checksum"]
    assert misattributed["mismatches"][0] == {
        "field": "year",
        "claimed": 2023,
        "found": 2020,
        "corroborating_sources": ["crossref", "openalex", "semantic_scholar"],
    }
    assert misattributed["matched"]["source"] in {"semantic_scholar", "openalex", "crossref"}
    assert unresolvable["verdict"] == "UNRESOLVABLE"
    assert unresolvable["matched"] is None
    assert unresolvable["queries"][0]["candidates"] == 0


async def test_manifest_feeds_report_checksum(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fetches.jsonl")
    await _audit((EXACT_DOI,), cache)
    manifest = cache.manifest()
    assert len(manifest["entries"]) == 3
    assert manifest["checksum"] == content_checksum(manifest["entries"])
    assert all(entry["source"] in {"semantic_scholar", "openalex", "crossref"} for entry in manifest["entries"])
