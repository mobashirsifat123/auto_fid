from __future__ import annotations

import json
import re
import asyncio
from pathlib import Path

import os
import pytest
from pypdf import PdfWriter

from autoresearch.application.services import DiscoveryService, LiteratureReviewService, NoveltyService, ExperimentService
from autoresearch.domain.models import DownloadStatus, GapClassification, content_checksum
from autoresearch.papers.metadata_sources import MetadataPaper, SourceSearchResult
from autoresearch.papers.pdf_resolver import DownloadedPdf, PdfResolution
from autoresearch.papers.search_pipeline import StudySearchPipeline
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.config import ModelAssignment
from autoresearch.providers.registry import ProviderRegistry
from autoresearch.providers.router import ModelRouter
from autoresearch.providers.testing import FakeEmbeddingProvider, FakeLLMProvider
from autoresearch.review.similarity import ReviewRanker
from autoresearch.storage import ResearchRepository


class DeterministicMetadataSource:
    name = "deterministic_source"

    async def search(self, query: str, *, limit: int, since_year: int | None = None) -> SourceSearchResult:
        papers = tuple(
            MetadataPaper(
                title="Seed Adaptive Fusion" if index == 0 else f"Adaptive Fusion Study {index}",
                abstract=f"Study {index} evaluates adaptive fusion on Dataset-A. Adaptive fusion improves robustness. Evaluation is limited to one dataset.",
                authors=(f"Researcher {index}",), year=2026 - (index % 4), publication_date=f"{2026 - (index % 4)}-01-01",
                doi=f"10.5555/study.{index}", url=f"https://example.org/study/{index}",
                open_access_pdf_url=f"https://oa.example.org/study-{index}.pdf", is_open_access=True,
                citation_count=index * 2, venue="Test Venue", source=self.name, external_id=str(index),
            )
            for index in range(25)
        )
        return SourceSearchResult(self.name, query, papers[:limit], content_checksum({"query": query, "count": len(papers[:limit])}))

    async def close(self) -> None:
        return None


class DeterministicPdfResolver:
    async def resolve(self, paper: MetadataPaper) -> PdfResolution:
        index = int((paper.doi or "0").rsplit(".", 1)[-1])
        if index % 2:
            return PdfResolution(False, None, "test_oa_registry", None, None, "No legal open-access PDF location")
        return PdfResolution(True, paper.open_access_pdf_url, "test_oa_registry", "Test fixture open-access declaration", "cc-by", None)

    async def download(self, resolution: PdfResolution, target: Path) -> DownloadedPdf:
        payload = b"%PDF-1.4\n% deterministic fixture\n%%EOF"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return DownloadedPdf(target, content_checksum(payload), len(payload))

    async def close(self) -> None:
        return None


def _responder(messages):
    prompt = str(messages[-1]["content"])
    if "seed-analysis-v1" in prompt:
        return {"title": "Seed Adaptive Fusion", "topic": "robust adaptive fusion", "research_question": "Can adaptive fusion improve cross-domain robustness?", "methods": ["adaptive fusion"], "datasets": ["Dataset-A"], "citations": [], "limitations": ["single-domain evaluation"], "keywords": ["robustness", "domain shift", "fusion"]}
    if "search-strategies-v1" in prompt:
        return {"strategies": [{"name": "topic", "query": "adaptive fusion robustness"}, {"name": "method", "query": "adaptive fusion domain shift"}, {"name": "question", "query": "cross-domain robust fusion"}, {"name": "limitation", "query": "single-domain evaluation limitation"}]}
    if "paper-review-v2" in prompt:
        return {
            "problem": "Cross-domain robustness", "research_question": "Does adaptive fusion improve robustness?", "method": "Adaptive fusion",
            "datasets": ["Dataset-A"], "experimental_protocol": "Single-dataset controlled evaluation", "results": ["Improved robustness"],
            "ablations": ["Fusion removed"], "limitations": ["Evaluation uses one dataset"], "failure_modes": ["Unverified domain transfer"], "confidence": 0.86,
            "evidence_spans": [
                {"claim_type": "result", "statement": "Adaptive fusion improves robustness.", "excerpt": "Adaptive fusion improves robustness.", "page": None, "section": "abstract", "confidence": 0.9},
                {"claim_type": "limitation", "statement": "Evaluation is limited to one dataset.", "excerpt": "Evaluation is limited to one dataset.", "page": None, "section": "abstract", "confidence": 0.95},
            ],
        }
    if "limitation-cluster-v1" in prompt:
        paper_ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([a-f0-9]{24})"', prompt)))
        return {"clusters": [{"theme": "single-domain evaluation", "frequency": 5, "paper_ids": paper_ids[:2], "description": "Most studies evaluate on a single dataset, limiting generalization."}]}
    if "evidence-gap-v2" in prompt:
        claim_ids = re.findall(r'"claim_id":\s*"([a-f0-9]{32})"', prompt)
        paper_ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([a-f0-9]{24})"', prompt)))
        return {"gap_statement": "Adaptive fusion lacks matched cross-domain validation.", "supporting_paper_ids": paper_ids[:2], "supporting_claim_ids": claim_ids[:2], "why_existing_work_does_not_solve_it": "The highest-priority studies evaluate only one dataset.", "missing_experiment": "A parameter-matched evaluation across three held-out domains.", "falsification_test": "Reject the gap if existing matched multi-domain evidence shows consistent gains across three domains.", "confidence": 0.88, "counter_evidence_claim_ids": claim_ids[2:3]}
    if "counter-evidence-v1" in prompt:
        return {"counter_evidence_claim_ids": [], "analysis": "No contradicting claims found in the supplied evidence corpus."}
    if "gap-selection-v1" in prompt:
        return {"selected_index": 0, "selection_rationale": "Candidate 0 is precisely stated and most strongly supported by paper evidence."}
    if "Research Gap:" in prompt or "limitation root cause elimination" in prompt or "candidate-gen-v1" in prompt:
        return {
            "strategy": "limitation root cause elimination", "method": "New Method", "causal_mechanism": "Cause",
            "target_limitation": "Limitation", "difference_from_existing": "Diff", "expected_effect": "Effect",
            "falsification_condition": "Falsification", "minimum_validating_experiment": "Experiment",
            "required_ablations": [], "implementation_complexity": "Low", "expected_compute": "Low", "risks_and_objections": []
        }
    if "Prior Art Abstracts" in prompt or "prior-art-v1" in prompt:
        return {"overlap_status": "potentially_novel", "mechanism_comparison": "Comparison", "supporting_excerpts": [], "unresolved_overlap": "None"}
    if "Seeds required:" in prompt:
        return {"selected_index": 0, "selection_rationale": "Design 0 is maximally falsifiable with adequate statistical power."}
    if "Candidate Method:" in prompt and "Falsification Condition:" in prompt:
        return {"what_would_disprove": "No consistent improvement over baseline.", "confounding_factors": ["Seed variance"], "minimal_viable_experiment": "One dataset with three seeds.", "expected_null_outcome": "No consistent gain.", "failure_modes": ["Implementation error"]}
    if "Method:" in prompt or "novelty-critic-v1" in prompt:
        return {
            "novelty_score": 0.9, "significance_score": 0.9, "causal_plausibility_score": 0.9, "falsifiability_score": 0.9,
            "experimental_discriminability_score": 0.9, "feasibility_score": 0.9, "implementation_risk_score": 0.1,
            "expected_information_gain_score": 0.9, "compute_efficiency_score": 0.9, "publication_potential_score": 0.9,
            "objections": [], "failure_scenarios": []
        }
    if "Candidate Method:" in prompt or "experiment-planner-v1" in prompt:
        return {
            "hypothesis": "H1", "causal_mechanism": "C1", "falsification_condition": "F1", "repository": "repo",
            "baseline_commit": "main", "dataset_identity": "data", "dataset_checksum": "checksum", "split_definition": "split",
            "baseline_configuration": "base", "proposed_change": "change", "independent_variables": [], "controlled_variables": [],
            "metrics": [], "random_seeds": [42], "ablations": [], "robustness_tests": [], "statistical_plan": "plan",
            "compute_budget": "budget", "stopping_conditions": [], "expected_outputs": [], "artifact_contract": "contract"
        }
    if "Baseline Metrics:" in prompt or "result-analyzer-v1" in prompt:
        return {"failure_case_analysis": "None", "hypothesis_support_assessment": "Supported"}
    if "Statistical Plan:" in prompt and "Baseline:" in prompt:
        return {"construct_validity": "The metrics measure the claimed construct.", "internal_validity": "Confounds are controlled via matched baselines.", "external_validity": "Single-dataset results limit generalization.", "statistical_conclusion_validity": "Appropriate given the seed count."}
    if "Assessment:" in prompt or "report-writer-v1" in prompt:
        return {
            "literature_motivation": "Motiv", "research_gap": "Gap", "implementation_diff": "Diff", "uncertainty": "Uncert",
            "ablations": "Ablations", "robustness_tests": "Robust", "limitations": "Limit", "reproducibility_instructions": "Repro",
            "decision": "supported", "decision_evidence": "Evidence"
        }

    raise AssertionError(f"Unexpected provider prompt: {prompt[:120]}")


def _router(repository):
    registry = ProviderRegistry()
    registry.register_llm("fake", FakeLLMProvider(responder=_responder))
    registry.register_embeddings("fake_embeddings", FakeEmbeddingProvider())
    assignments = {
        ModelRole.LITERATURE_EXTRACTION: ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "fake", "fake-model"),
        ModelRole.RESEARCH_REASONING: ModelAssignment(ModelRole.RESEARCH_REASONING, "fake", "fake-model"),
        ModelRole.SCIENTIFIC_CRITIC: ModelAssignment(ModelRole.SCIENTIFIC_CRITIC, "fake", "fake-model"),
        ModelRole.EMBEDDINGS: ModelAssignment(ModelRole.EMBEDDINGS, "fake_embeddings", "fake-embedding-v1"),
    }
    return ModelRouter(registry, assignments, usage_sink=repository.record_provider_usage)


def _seed_pdf(path: Path) -> None:
    writer = PdfWriter(); writer.add_blank_page(width=200, height=200); writer.add_metadata({"/Title": "Seed Adaptive Fusion"})
    with path.open("wb") as handle:
        writer.write(handle)


@pytest.mark.anyio
async def test_complete_first_milestone_without_paid_api_keys(tmp_path):
    repository = ResearchRepository(tmp_path / "autoresearch.sqlite3", tmp_path / "artifacts")
    router = _router(repository)
    pipeline = StudySearchPipeline(repository, [DeterministicMetadataSource()], DeterministicPdfResolver(), tmp_path / "downloads")
    discovery_service = DiscoveryService(repository, router, pipeline)
    review_service = LiteratureReviewService(repository, router)
    seed = tmp_path / "seed.pdf"; _seed_pdf(seed)

    discovery = await discovery_service.discover(seed, limit=20, since_year=2023)
    assert discovery.retrieved == 20
    assert discovery.downloaded > 0
    assert discovery.downloaded + discovery.unavailable == 20
    assert repository.get_run(discovery.run_id).status.value == "completed"
    assert len(repository.search_events(discovery.run_id)) == 4
    assert len(repository.discovery_scores(discovery.run_id)) == 20
    discovery_report = json.loads(Path(discovery.artifact_path).read_text())
    assert discovery_report["timestamp"]
    assert discovery_report["source_evidence"]
    assert all(paper.title != "Seed Adaptive Fusion" for paper in repository.list_papers(discovery.project_id))

    review = await review_service.review(discovery.project_id, top_k=5)
    assert review.reviewed == 20
    assert len(review.top_paper_ids) == 5
    assert review.gap.classification is GapClassification.EVIDENCE_SUPPORTED
    assert review.gap.supporting_claim_ids
    assert Path(review.json_report).exists() and Path(review.markdown_report).exists()
    report = json.loads(Path(review.json_report).read_text())
    assert report["project_id"] == discovery.project_id
    assert report["run_id"] == review.run_id
    assert len(report["ranked_papers"]) == 5
    assert report["gap"]["supporting_claims"][0]["excerpt"]
    assert report["provenance"]["provider"] == "fake"
    assert len(report["provenance"]["checksum"]) == 64
    markdown = Path(review.markdown_report).read_text()
    assert "## Provenance" in markdown
    assert discovery.project_id in markdown and review.run_id in markdown
    assert repository.integrity_check() == "ok"
    assert repository.row("SELECT COUNT(*) AS n FROM provider_usage")["n"] >= 24
    assert repository.review_failures(review.run_id) == []
    
    novelty_service = NoveltyService(repository, router, pipeline)
    portfolio_id = await novelty_service.generate_portfolio(discovery.project_id)
    assert portfolio_id is not None
    portfolio = repository.row("SELECT * FROM novelty_portfolios WHERE id=?", (portfolio_id,))
    assert portfolio["low_risk_candidate_id"]

    workspace = tmp_path / "workspace"
    workspace.mkdir(exist_ok=True)
    experiment_service = ExperimentService(repository, router, workspace)
    spec_id = await experiment_service.execute_candidate(portfolio["low_risk_candidate_id"], mode="plan")
    assert spec_id is not None
    spec = repository.row("SELECT * FROM experiment_specs WHERE id=?", (spec_id,))
    assert spec["approval_required"] == 1
    assert repository.row("SELECT * FROM execution_jobs LIMIT 1") is None

    await pipeline.close(); await router.close()


class TrackingMetadataSource(DeterministicMetadataSource):
    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, *, limit: int, since_year: int | None = None) -> SourceSearchResult:
        self.calls += 1
        return await super().search(query, limit=limit, since_year=since_year)


class TrackingPdfResolver(DeterministicPdfResolver):
    def __init__(self) -> None:
        self.resolutions = 0
        self.downloads = 0

    async def resolve(self, paper: MetadataPaper) -> PdfResolution:
        self.resolutions += 1
        return await super().resolve(paper)

    async def download(self, resolution: PdfResolution, target: Path) -> DownloadedPdf:
        self.downloads += 1
        return await super().download(resolution, target)


class InterruptAfterDiscoveryPipeline(StudySearchPipeline):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.interrupt_once = True

    async def discover(self, **kwargs):
        result = await super().discover(**kwargs)
        if self.interrupt_once:
            self.interrupt_once = False
            raise asyncio.CancelledError("simulated interruption after durable downloads")
        return result


class CountingEmbeddingProvider(FakeEmbeddingProvider):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    async def embed_batch(self, *, model: str, texts):
        self.calls += 1
        return await super().embed_batch(model=model, texts=texts)


class InterruptAfterEmbeddingRanker(ReviewRanker):
    def __init__(self, router) -> None:
        super().__init__(router)
        self.interrupt_once = True

    async def rank(self, **kwargs):
        scores = await super().rank(**kwargs)
        if self.interrupt_once:
            self.interrupt_once = False
            raise asyncio.CancelledError("simulated interruption after cached embeddings")
        return scores


def _tracking_router(repository):
    llm = FakeLLMProvider(responder=_responder)
    embeddings = CountingEmbeddingProvider()
    registry = ProviderRegistry()
    registry.register_llm("fake", llm)
    registry.register_embeddings("fake_embeddings", embeddings)
    assignments = {
        ModelRole.LITERATURE_EXTRACTION: ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "fake", "fake-model"),
        ModelRole.RESEARCH_REASONING: ModelAssignment(ModelRole.RESEARCH_REASONING, "fake", "fake-model"),
        ModelRole.SCIENTIFIC_CRITIC: ModelAssignment(ModelRole.SCIENTIFIC_CRITIC, "fake", "fake-model"),
        ModelRole.EMBEDDINGS: ModelAssignment(ModelRole.EMBEDDINGS, "fake_embeddings", "fake-embedding-v1"),
    }
    return ModelRouter(registry, assignments, usage_sink=repository.record_provider_usage), llm, embeddings


@pytest.mark.asyncio
async def test_interrupted_discovery_resumes_without_repeating_searches_or_downloads(tmp_path):
    repository = ResearchRepository(tmp_path / "autoresearch.sqlite3", tmp_path / "artifacts")
    router, llm, _ = _tracking_router(repository)
    source = TrackingMetadataSource()
    resolver = TrackingPdfResolver()
    pipeline = InterruptAfterDiscoveryPipeline(repository, [source], resolver, tmp_path / "downloads")
    service = DiscoveryService(repository, router, pipeline)
    seed = tmp_path / "seed.pdf"; _seed_pdf(seed)

    with pytest.raises(asyncio.CancelledError):
        await service.discover(seed, limit=20, since_year=2023)
    run_id = repository.row("SELECT id FROM pipeline_runs WHERE kind='study_search'")["id"]
    first_counts = (source.calls, resolver.resolutions, resolver.downloads, len(llm.calls))

    resumed = await service.resume(run_id)
    assert resumed.retrieved == 20
    assert (source.calls, resolver.resolutions, resolver.downloads, len(llm.calls)) == first_counts
    assert all(item["status"] == "completed" for item in repository.checkpoints(run_id))

    repeated = await service.resume(run_id)
    assert repeated.artifact_path == resumed.artifact_path
    assert (source.calls, resolver.resolutions, resolver.downloads, len(llm.calls)) == first_counts
    await pipeline.close(); await router.close()


@pytest.mark.asyncio
async def test_interrupted_review_reuses_reviews_and_cached_embeddings(tmp_path):
    repository = ResearchRepository(tmp_path / "autoresearch.sqlite3", tmp_path / "artifacts")
    router, llm, embeddings = _tracking_router(repository)
    pipeline = StudySearchPipeline(repository, [DeterministicMetadataSource()], DeterministicPdfResolver(), tmp_path / "downloads")
    discovery_service = DiscoveryService(repository, router, pipeline)
    review_service = LiteratureReviewService(repository, router)
    review_service.ranker = InterruptAfterEmbeddingRanker(router)
    seed = tmp_path / "seed.pdf"; _seed_pdf(seed)
    discovery = await discovery_service.discover(seed, limit=20, since_year=2023)

    with pytest.raises(asyncio.CancelledError):
        await review_service.review(discovery.project_id, top_k=5)
    run_id = repository.row("SELECT id FROM pipeline_runs WHERE kind='deep_literature_review'")["id"]
    paper_calls = sum("paper-review" in str(call["messages"][-1]["content"]) for call in llm.calls)
    assert paper_calls == 20 * 3  # multi-perspective review: 3 perspectives per paper
    assert len(repository.reviews(discovery.project_id, run_id)) == 20
    assert embeddings.calls == 1

    resumed = await review_service.resume(run_id)
    assert resumed.reviewed == 20
    assert embeddings.calls == 1
    assert sum("paper-review" in str(call["messages"][-1]["content"]) for call in llm.calls) == 20 * 3

    llm_count = len(llm.calls)
    repeated = await review_service.resume(run_id)
    assert repeated.json_report == resumed.json_report
    assert embeddings.calls == 1 and len(llm.calls) == llm_count
    await pipeline.close(); await router.close()


class RecoveringMetadataSource(DeterministicMetadataSource):
    name = "recovering_source"

    def __init__(self) -> None:
        self.attempts: dict[str, int] = {}

    async def search(self, query: str, *, limit: int, since_year: int | None = None) -> SourceSearchResult:
        self.attempts[query] = self.attempts.get(query, 0) + 1
        if self.attempts[query] <= 3:
            raise RuntimeError("temporary source outage")
        result = await super().search(query, limit=limit, since_year=since_year)
        papers = tuple(MetadataPaper(**{**paper.__dict__, "source": self.name}) for paper in result.papers)
        return SourceSearchResult(self.name, result.query, papers, result.raw_checksum)


@pytest.mark.asyncio
async def test_partial_source_failure_recovers_without_repeating_successful_work(tmp_path):
    repository = ResearchRepository(tmp_path / "autoresearch.sqlite3", tmp_path / "artifacts")
    router, _, _ = _tracking_router(repository)
    stable = TrackingMetadataSource()
    recovering = RecoveringMetadataSource()
    resolver = TrackingPdfResolver()
    pipeline = StudySearchPipeline(repository, [stable, recovering], resolver, tmp_path / "downloads")
    service = DiscoveryService(repository, router, pipeline)
    seed = tmp_path / "seed.pdf"; _seed_pdf(seed)

    first = await service.discover(seed, limit=20, since_year=2023)
    assert first.retrieved == 20
    assert len([item for item in repository.checkpoints(first.run_id) if item["status"] == "failed"]) == 4
    stable_calls = stable.calls
    acquisition_counts = (resolver.resolutions, resolver.downloads)

    resumed = await service.resume(first.run_id)
    assert resumed.retrieved == 20
    assert stable.calls == stable_calls
    assert (resolver.resolutions, resolver.downloads) == acquisition_counts
    assert sum(attempts == 4 for attempts in recovering.attempts.values()) == 4
    assert not [item for item in repository.checkpoints(first.run_id) if item["status"] == "failed"]
    await pipeline.close(); await router.close()
