"""Quality-width upgrades: parallel expansion must widen coverage without
growing the critical path, and must degrade gracefully under partial failure.

Covers: multi-perspective discovery strategies, OpenAlex citation chaining,
dual-leg review extraction ensemble, and the widened prior-art fan-out.
Offline only (deterministic fakes, no network, no paid API).
"""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import pytest

from autoresearch.application.services import SearchStrategyGenerator
from autoresearch.domain.models import (
    ArtifactProvenance,
    CandidateStatus,
    NoveltyCandidate,
    PaperRecord,
    PipelineKind,
    SeedPaper,
    content_checksum,
)
from autoresearch.ingestion.chunking import DocumentChunk
from autoresearch.ingestion.extraction import SeedAnalysis
from autoresearch.novelty.prior_art_checker import EvidenceBackedPriorArtChecker
from autoresearch.papers.deduplication import normalized_title
from autoresearch.papers.metadata_sources import MetadataPaper, SourceSearchResult
from autoresearch.papers.search_pipeline import SearchStrategy, StudySearchPipeline
from autoresearch.providers.budget import ProviderBudgetGuard
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.config import ModelAssignment
from autoresearch.providers.registry import ProviderRegistry
from autoresearch.providers.router import ModelRouter
from autoresearch.providers.testing import FakeLLMProvider
from autoresearch.review.extraction import PaperReviewExtractor
from autoresearch.storage import ResearchRepository


def _router(responder) -> ModelRouter:
    registry = ProviderRegistry()
    registry.register_llm("fake", FakeLLMProvider(responder=responder))
    assignments = {
        role: ModelAssignment(role, "fake", "fake-model")
        for role in (
            ModelRole.LITERATURE_EXTRACTION,
            ModelRole.RESEARCH_REASONING,
            ModelRole.SCIENTIFIC_CRITIC,
        )
    }
    return ModelRouter(registry, assignments)


def _repository(tmp_path: Path) -> ResearchRepository:
    return ResearchRepository(tmp_path / "state.sqlite3", tmp_path / "artifacts")


def _seed_analysis() -> SeedAnalysis:
    return SeedAnalysis(
        "Seed Adaptive Fusion", "robust adaptive fusion",
        "Can adaptive fusion improve cross-domain robustness?",
        ("adaptive fusion",), ("Dataset-A",), (), ("single-domain evaluation",),
        ("robustness",),
    )


class UnavailablePdfResolver:
    async def resolve(self, paper: MetadataPaper):
        from autoresearch.papers.pdf_resolver import PdfResolution

        return PdfResolution(False, None, "none", None, None, "offline test")

    async def download(self, resolution, target: Path):
        raise AssertionError("unavailable resolution must not download")

    async def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# 1. Multi-perspective search-strategy generation
# ---------------------------------------------------------------------------

def _strategy_responder(messages):
    content = str(messages[-1]["content"])
    perspective = re.search(r"Perspective: ([^\n]+)", content).group(1)
    if perspective == "limitation-counter":
        raise RuntimeError("simulated perspective failure")
    return {
        "strategies": [
            {"name": f"{perspective}-a", "query": f"{perspective} query alpha"},
            {"name": f"{perspective}-b", "query": f"{perspective} query beta"},
            {"name": "shared", "query": "shared query"},
        ]
    }


@pytest.mark.anyio
async def test_strategy_generation_parallel_perspectives_union_and_partial_failure(tmp_path):
    generator = SearchStrategyGenerator(_router(_strategy_responder), perspectives=3)
    analysis = _seed_analysis()
    strategies = await generator.generate(
        analysis, project_id="proj", run_id="run"
    )
    queries = {s.query for s in strategies}
    # Both surviving perspectives contributed, the failed one did not, the
    # shared query was deduplicated, and no duplicate queries remain.
    assert {"topic-method query alpha", "application-domain query alpha"} <= queries
    assert not any(q.startswith("limitation-counter") for q in queries)
    assert "shared query" in queries
    assert len(strategies) == len(queries) == 5
    assert all(" " in s.query for s in strategies)


@pytest.mark.anyio
async def test_strategy_generation_total_failure_still_fails_closed(tmp_path):
    def failing(messages):
        raise RuntimeError("every perspective down")

    generator = SearchStrategyGenerator(_router(failing), perspectives=3)
    with pytest.raises(ValueError, match="at least three distinct"):
        await generator.generate(_seed_analysis(), project_id="proj", run_id="run")


# ---------------------------------------------------------------------------
# 2. OpenAlex citation chaining
# ---------------------------------------------------------------------------

class ChainingOpenAlexSource:
    name = "openalex"

    async def search(self, query: str, *, limit: int, since_year: int | None = None):
        papers = tuple(
            MetadataPaper(
                title=f"Anchor Study {index}",
                abstract="An unrelated abstract without the seed terms.",
                authors=("Anchor Author",), year=2020, doi=f"10.5555/anchor.{index}",
                url="https://example.org/anchor", is_open_access=False,
                citation_count=1, venue="Venue", source=self.name,
                external_id=f"https://openalex.org/W{index}",
            )
            for index in range(3)
        )
        return SourceSearchResult(self.name, query, papers[:limit], content_checksum({"q": query}))

    async def related(self, work_id: str, direction: str, *, limit: int):
        papers = tuple(
            MetadataPaper(
                title=f"Neighbor {direction} {work_id.rsplit('/', 1)[-1]} {index}",
                abstract="Adaptive fusion improves cross-domain robustness on Dataset-A. The robustness gain is matched across domains.",
                authors=("Neighbor Author",), year=2026, doi=f"10.5555/n.{work_id.rsplit('/', 1)[-1]}.{direction}.{index}",
                url="https://example.org/neighbor", is_open_access=True,
                citation_count=500, venue="Top Venue", source=self.name,
                external_id=f"https://openalex.org/N{index}",
            )
            for index in range(4)
        )
        return SourceSearchResult(self.name, f"{direction}:{work_id}", papers[:limit], content_checksum({"w": work_id, "d": direction}))

    async def lookup(self, identifier: str):
        return MetadataPaper(
            title="Resolved Anchor", abstract="", authors=(), year=2020,
            doi=identifier, url=None, is_open_access=False, citation_count=0,
            venue="", source=self.name, external_id="https://openalex.org/W99",
        )

    async def close(self) -> None:
        return None


def _seed(project_id: str, run_id: str) -> SeedPaper:
    return SeedPaper(
        uuid4().hex, project_id, run_id, "pdf", "seed.pdf",
        "Seed Adaptive Fusion",
        "Adaptive fusion robustness domain shift Dataset-A cross-domain",
        "robust adaptive fusion", "question", content_checksum("seed"),
        ArtifactProvenance.create(
            project_id=project_id, run_id=run_id, source_evidence=("seed",),
            provider="fake", model="fake", prompt_version="v1", content={},
        ),
        methods=("adaptive fusion",), datasets=("Dataset-A",),
    )


@pytest.mark.anyio
async def test_citation_chaining_enriches_pool_and_checkpoints(tmp_path):
    repository = _repository(tmp_path)
    project = repository.create_project("chaining")
    run = repository.create_run(project.id, PipelineKind.DISCOVERY, {})
    seed = _seed(project.id, run.id)
    repository.save_seed(seed)
    pipeline = StudySearchPipeline(
        repository, [ChainingOpenAlexSource()], UnavailablePdfResolver(),
        tmp_path / "downloads", citation_expansion=True, citation_anchors=2,
        citation_neighbors=4,
    )
    result = await pipeline.discover(
        project_id=project.id, run_id=run.id, seed=seed,
        strategies=(SearchStrategy("topic", "adaptive fusion"),), limit=5,
    )
    titles = [paper.title for paper in result.papers]
    # Neighbors outrank the anchors (seed-term relevance + influence) and are
    # selected; the pool genuinely widened from 3 to 19 unique candidates.
    assert len(result.papers) == 5
    assert sum(title.startswith("Neighbor") for title in titles) >= 4
    assert result.raw_candidate_count == 3
    assert result.unique_candidate_count == 19
    assert result.duplicate_count == 0  # no overlaps between anchors and neighbors
    # Checkpointed per anchor × direction for resume; failures recorded as
    # search events would also appear here on real sources.
    for anchor in ("W0", "W1"):
        for direction in ("forward", "backward"):
            checkpoint = repository.checkpoint(run.id, "citation_expansion", f"https://openalex.org/{anchor}:{direction}")
            assert checkpoint and checkpoint["status"] == "completed"
    events = repository.search_events(run.id)
    assert any("citation-forward" in event["strategy"] for event in events)
    assert any("citation-backward" in event["strategy"] for event in events)


class PlainSource:
    name = "plain"

    async def search(self, query: str, *, limit: int, since_year: int | None = None):
        return SourceSearchResult(self.name, query, tuple(
            MetadataPaper(title=f"Plain {index}", abstract="Adaptive fusion robustness.", authors=(), year=2024, doi=f"10.5555/p.{index}", url=None, is_open_access=True, citation_count=0, venue="", source=self.name, external_id="")
            for index in range(5)
        ), content_checksum({"q": query}))

    async def close(self) -> None:
        return None


@pytest.mark.anyio
async def test_citation_chaining_skips_gracefully_without_supporting_source(tmp_path):
    repository = _repository(tmp_path)
    project = repository.create_project("no-chain")
    run = repository.create_run(project.id, PipelineKind.DISCOVERY, {})
    seed = _seed(project.id, run.id)
    repository.save_seed(seed)
    pipeline = StudySearchPipeline(
        repository, [PlainSource()], UnavailablePdfResolver(), tmp_path / "downloads",
        citation_expansion=True, citation_anchors=5, citation_neighbors=10,
    )
    result = await pipeline.discover(
        project_id=project.id, run_id=run.id, seed=seed,
        strategies=(SearchStrategy("topic", "adaptive fusion"),), limit=3,
    )
    assert len(result.papers) == 3
    assert repository.checkpoints(run.id) and all(
        item["stage"] != "citation_expansion" for item in repository.checkpoints(run.id)
    )


# ---------------------------------------------------------------------------
# 3. Dual-leg review extraction ensemble
# ---------------------------------------------------------------------------

def _paper() -> PaperRecord:
    return PaperRecord(
        "p1", "proj", "run", "Ensemble Paper", "abstract", (), 2026, None,
        None, None, None, ("fake",), None, False, 0, "Venue",
        content_checksum("paper"),
    )


def _chunks() -> tuple[DocumentChunk, ...]:
    return (
        DocumentChunk("c1", 1, 1, "intro", "First sentence about fusion. Second sentence about domain shift."),
        DocumentChunk("c2", 2, 2, "results", "Third sentence about robustness. Fourth sentence about calibration."),
    )


def _review_responder(messages):
    content = str(messages[-1]["content"])
    source = content.split("SOURCE:\n", 1)[1].split("\n\nReturn one JSON object", 1)[0]
    sentences = [s.strip() for s in re.findall(r"[^.!?]+[.!?]", source)][:2]
    return {
        "problem": "P", "research_question": "Q", "method": "M",
        "datasets": ["D"], "experimental_protocol": "E", "results": ["R"],
        "ablations": [], "limitations": ["L"], "failure_modes": [],
        "confidence": 0.8,
        "evidence_spans": [
            {"claim_type": "finding", "statement": s, "excerpt": s, "page": None, "section": "abstract", "confidence": 0.9}
            for s in sentences
        ],
    }


def _review_responder_second_leg_fails(messages):
    content = str(messages[-1]["content"])
    source = content.split("SOURCE:\n", 1)[1].split("\n\nReturn one JSON object", 1)[0]
    if source.startswith("[PAGE 2"):
        raise RuntimeError("simulated second-leg failure")
    return _review_responder(messages)


@pytest.mark.anyio
async def test_review_ensemble_merges_two_legs():
    extractor = PaperReviewExtractor(_router(_review_responder))
    bundle = await extractor.extract_ensemble(
        _paper(), run_id="run", chunks=_chunks(), fallback_text="",
    )
    # Leg 1 sees pages in order (claims from page 1), leg 2 reversed (claims
    # from page 2): the merge covers both — wider span coverage.
    statements = {claim.statement for claim in bundle.claims}
    assert len(statements) == 4
    assert any("First sentence" in s for s in statements)
    assert any("Fourth sentence" in s for s in statements)
    assert "ensemble" in bundle.review.provenance.prompt_version


@pytest.mark.anyio
async def test_review_ensemble_degrades_to_single_surviving_leg():
    extractor = PaperReviewExtractor(_router(_review_responder_second_leg_fails))
    bundle = await extractor.extract_ensemble(
        _paper(), run_id="run", chunks=_chunks(), fallback_text="",
    )
    # The reversed leg starts with PAGE 2 and fails; the survivor's claims
    # (page-1 sentences) still come back — ensemble never makes a paper
    # harder to review than a single pass.
    statements = {claim.statement for claim in bundle.claims}
    assert len(statements) == 2
    assert all("First sentence" in s or "Second sentence" in s for s in statements)


# ---------------------------------------------------------------------------
# 4. Widened prior-art fan-out
# ---------------------------------------------------------------------------

class RecordingSource:
    name = "recording_source"

    def __init__(self, suffix: str) -> None:
        self.suffix = suffix
        self.queries: list[str] = []

    async def search(self, query: str, *, limit: int, since_year: int | None = None):
        self.queries.append(query)
        return SourceSearchResult(self.name, query, tuple(
            MetadataPaper(
                title=f"{self.suffix} paper {index}", abstract="Prior art abstract.",
                authors=(), year=2025, doi=f"10.5555/{self.suffix}.{index}",
                url=None, is_open_access=True, citation_count=0, venue="",
                source=self.name, external_id="",
            )
            for index in range(2)
        ), content_checksum({"q": query}))

    async def close(self) -> None:
        return None


def _prior_art_responder(messages):
    return {
        "overlap_status": "potentially_novel",
        "mechanism_comparison": "Comparison",
        "supporting_excerpts": [],
        "unresolved_overlap": "None",
    }


@pytest.mark.anyio
async def test_prior_art_fanout_covers_all_sources_and_cross_domain_queries(tmp_path):
    repository = _repository(tmp_path)
    project = repository.create_project("prior-art")
    run = repository.create_run(project.id, PipelineKind.NOVELTY, {})
    seed = _seed(project.id, run.id)
    repository.save_seed(seed)
    sources = [RecordingSource("a"), RecordingSource("b")]
    pipeline = StudySearchPipeline(
        repository, sources, UnavailablePdfResolver(), tmp_path / "downloads",
    )
    checker = EvidenceBackedPriorArtChecker(_router(_prior_art_responder), pipeline)
    candidate = NoveltyCandidate(
        uuid4().hex, project.id, run.id, "gap",
        "limitation root cause elimination",
        "spectral residue amplification", "frequency domain artifacts",
        "missed manipulations", "adds residue normalization",
        "improved detection", "fails if no artifacts", "ablation test",
        (), "Low", "Low", (), CandidateStatus.GENERATED,
        ArtifactProvenance.create(
            project_id=project.id, run_id=run.id, source_evidence=(),
            provider="fake", model="fake", prompt_version="v1", content={},
        ),
    )
    assessment = await checker.check(candidate)
    # ALL sources were searched (not just the first two)…
    all_queries = [q for source in sources for q in source.queries]
    assert len(all_queries) >= 8
    assert all(source.queries for source in sources)
    # …with cross-domain probes grounded in the seed's topic and datasets.
    assert any("applied to robust adaptive fusion" in q for q in all_queries)
    assert any("Dataset-A" in q for q in all_queries)
    # External retrieval genuinely happened: with no external results the
    # checker hard-downgrades `potentially_novel` to `expert_review_required`
    # (corpus-bounded assessment). Here the fake LLM's status survives, which
    # proves the widened fan-out actually retrieved evidence.
    assert assessment.overlap_status.value == "potentially_novel"
    assert "CORPUS_BOUNDED_ASSESSMENT" not in assessment.unresolved_overlap


def test_candidate_queries_deduplicates_and_drops_empty():
    from autoresearch.novelty.prior_art_checker import _candidate_queries

    candidate = NoveltyCandidate(
        uuid4().hex, "p", "r", "g", "s", "method X", "mechanism Y", "lim",
        "method X", "effect", "fals", "exp", (), "Low", "Low", (),
        CandidateStatus.GENERATED,
        ArtifactProvenance.create(
            project_id="p", run_id="r", source_evidence=(), provider="f",
            model="m", prompt_version="v", content={},
        ),
    )
    seed = SeedPaper(
        "s", "p", "r", "pdf", "v", "title", "text", "image forensics", "q",
        "c", ArtifactProvenance.create(
            project_id="p", run_id="r", source_evidence=(), provider="f",
            model="m", prompt_version="v", content={},
        ),
        datasets=("DeepFakeDetect",),
    )
    queries = _candidate_queries(candidate, seed)
    assert len(queries) == len(set(q.lower() for q in queries))
    assert any("applied to image forensics" in q for q in queries)
    assert any("DeepFakeDetect" in q for q in queries)
    assert all(q.strip() for q in queries)


def test_candidate_queries_no_seed_no_cross_domain():
    from autoresearch.novelty.prior_art_checker import _candidate_queries

    candidate = NoveltyCandidate(
        uuid4().hex, "p", "r", "g", "s", "method X", "mechanism Y", "lim",
        "diff", "effect", "fals", "exp", (), "Low", "Low", (),
        CandidateStatus.GENERATED,
        ArtifactProvenance.create(
            project_id="p", run_id="r", source_evidence=(), provider="f",
            model="m", prompt_version="v", content={},
        ),
    )
    queries = _candidate_queries(candidate, None)
    assert len(queries) == 3  # method, mechanism+ML, difference — no seed terms
    assert all("applied to" not in q for q in queries)


class FailingSource:
    name = "failing_source"

    async def search(self, query: str, *, limit: int, since_year: int | None = None):
        raise RuntimeError("source down")

    async def close(self) -> None:
        return None


class FailingChainingOpenAlexSource(ChainingOpenAlexSource):
    async def related(self, work_id: str, direction: str, *, limit: int):
        raise RuntimeError("citation endpoint down")


@pytest.mark.anyio
async def test_prior_art_corpus_bounded_downgrade_when_all_external_sources_fail(tmp_path):
    """The novelty guard's safety rail: `potentially_novel` must hard-downgrade
    to `expert_review_required` when every external source fails, so an LLM
    absence-of-evidence judgement can never become a novelty claim."""
    repository = _repository(tmp_path)
    project = repository.create_project("prior-art-fail")
    run = repository.create_run(project.id, PipelineKind.NOVELTY, {})
    repository.save_seed(_seed(project.id, run.id))
    pipeline = StudySearchPipeline(
        repository, [FailingSource()], UnavailablePdfResolver(), tmp_path / "downloads",
    )
    checker = EvidenceBackedPriorArtChecker(_router(_prior_art_responder), pipeline)
    candidate = NoveltyCandidate(
        uuid4().hex, project.id, run.id, "gap", "limitation root cause elimination",
        "method X", "mechanism Y", "lim", "diff", "effect", "fals", "exp",
        (), "Low", "Low", (), CandidateStatus.GENERATED,
        ArtifactProvenance.create(
            project_id=project.id, run_id=run.id, source_evidence=(),
            provider="fake", model="fake", prompt_version="v1", content={},
        ),
    )
    assessment = await checker.check(candidate)
    assert assessment.overlap_status.value == "expert_review_required"
    assert "CORPUS_BOUNDED_ASSESSMENT" in assessment.unresolved_overlap


@pytest.mark.anyio
async def test_citation_chaining_failure_degrades_to_plain_pool(tmp_path):
    """When every citation-neighbor request fails, discovery still completes
    with the plain search pool, the failure is checkpointed, and the search
    event records the error — expansion must never fail the discovery."""
    repository = _repository(tmp_path)
    project = repository.create_project("chaining-fail")
    run = repository.create_run(project.id, PipelineKind.DISCOVERY, {})
    seed = _seed(project.id, run.id)
    repository.save_seed(seed)
    pipeline = StudySearchPipeline(
        repository, [FailingChainingOpenAlexSource()], UnavailablePdfResolver(),
        tmp_path / "downloads", citation_expansion=True, citation_anchors=2,
        citation_neighbors=4,
    )
    result = await pipeline.discover(
        project_id=project.id, run_id=run.id, seed=seed,
        strategies=(SearchStrategy("topic", "adaptive fusion"),), limit=5,
    )
    assert len(result.papers) == 3  # the three anchor studies, nothing lost
    checkpoint = repository.checkpoint(run.id, "citation_expansion", "https://openalex.org/W0:forward")
    assert checkpoint and checkpoint["status"] == "failed"
    events = repository.search_events(run.id)
    citation_events = [e for e in events if "citation-" in e["strategy"]]
    assert citation_events and all(e["error"] for e in citation_events)
    assert all(e["result_count"] == 0 for e in citation_events)


def test_normalized_title_helper_used_for_seed_exclusion():
    assert normalized_title("  Adaptive   FUSION!! ") == "adaptive fusion"
