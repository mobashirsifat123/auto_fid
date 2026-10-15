from autoresearch.domain.models import ArtifactProvenance, PaperRecord, PaperReview, SeedPaper
from autoresearch.papers.deduplication import DeduplicatedCandidate, SearchProvenance
from autoresearch.papers.metadata_sources import MetadataPaper
from autoresearch.papers.ranking import rank_candidates
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.config import ModelAssignment
from autoresearch.providers.registry import ProviderRegistry
from autoresearch.providers.router import ModelRouter
from autoresearch.providers.testing import FakeLLMProvider
from autoresearch.review.similarity import ReviewRanker


def test_candidate_ranking_exposes_components_and_is_deterministic():
    provenance = (SearchProvenance("source", "topic", "adaptive fusion", 1, "c"),)
    related = DeduplicatedCandidate(MetadataPaper("Adaptive fusion for robust vision", "adaptive fusion improves robust vision", year=2025, citation_count=5, is_open_access=True), provenance)
    unrelated = DeduplicatedCandidate(MetadataPaper("Marine biology survey", "ocean ecosystems", year=2025, citation_count=100), provenance)
    ranked = rank_candidates([unrelated, related], seed_text="adaptive fusion robust vision", seed_methods=("adaptive fusion",), current_year=2026)
    assert ranked[0][0].paper.title.startswith("Adaptive")
    components = ranked[0][1].components()
    assert set(components) == {"relevance", "method_overlap", "recency", "influence", "source_agreement", "open_access", "total"}
    assert 0 <= components["total"] <= 1


def _lexical_router() -> ModelRouter:
    """Router with LLM roles but NO embeddings assignment — must not block review ranking."""
    registry = ProviderRegistry()
    registry.register_llm("fake", FakeLLMProvider(responses=[]))
    assignments = {
        ModelRole.LITERATURE_EXTRACTION: ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "fake", "fake-model"),
        ModelRole.RESEARCH_REASONING: ModelAssignment(ModelRole.RESEARCH_REASONING, "fake", "fake-model"),
        ModelRole.SCIENTIFIC_CRITIC: ModelAssignment(ModelRole.SCIENTIFIC_CRITIC, "fake", "fake-model"),
    }
    return ModelRouter(registry, assignments)


def _provenance(project_id: str, run_id: str) -> ArtifactProvenance:
    return ArtifactProvenance.create(project_id=project_id, run_id=run_id, source_evidence=(), provider="fake", model="fake-model", prompt_version="test", content={})


def test_review_ranking_falls_back_to_lexical_similarity_without_embeddings():
    project_id, run_id = "project-1", "run-1"
    seed = SeedPaper(id="seed-1", project_id=project_id, run_id=run_id, input_type="pdf", input_value="/seed.pdf", title="Adaptive fusion for robust vision", text="adaptive fusion improves robust vision", topic="robust vision", research_question="Does adaptive fusion improve robustness?", checksum="seed", methods=("adaptive fusion", "gradient projection"), provenance=_provenance(project_id, run_id))
    paper = PaperRecord(id="paper-1", project_id=project_id, run_id=run_id, title="Adaptive fusion for robust vision", abstract="adaptive fusion for robust vision classification", authors=("A",), year=2025, publication_date="2025-01-01", doi="10.1/2", arxiv_id=None, url=None, source_names=("openalex",), citation_count=5)
    related_review = PaperReview(id="review-1", project_id=project_id, run_id=run_id, paper_id="paper-1", problem="robust vision under distribution shift", research_question="Does adaptive fusion improve robustness?", method="adaptive fusion with gradient projection", datasets=(), experimental_protocol="", results=("accuracy gains",), ablations=(), limitations=(), failure_modes=(), evidence_claim_ids=("claim-1",), confidence=0.8, provenance=_provenance(project_id, run_id))
    unrelated = PaperRecord(id="paper-2", project_id=project_id, run_id=run_id, title="Marine biology survey", abstract="ocean ecosystems and fish populations", authors=("B",), year=2025, publication_date="2025-01-01", doi="10.1/3", arxiv_id=None, url=None, source_names=("openalex",), citation_count=100)
    unrelated_review = PaperReview(id="review-2", project_id=project_id, run_id=run_id, paper_id="paper-2", problem="ocean ecosystem dynamics", research_question="How do fish populations change?", method="statistical survey of ocean samples", datasets=(), experimental_protocol="", results=("population curves",), ablations=(), limitations=(), failure_modes=(), evidence_claim_ids=("claim-2",), confidence=0.8, provenance=_provenance(project_id, run_id))

    scores = _run_rank(seed, [paper, unrelated], [related_review, unrelated_review])

    assert len(scores) == 2
    assert scores[0].paper_id == "paper-1", "lexical similarity should still rank the topically-related paper first"
    assert scores[0].explanation["similarity_basis"] == "lexical"
    assert scores[0].provenance.provider == "deterministic"
    assert scores[0].provenance.model == "lexical-ranking"
    assert "lexical" in scores[0].provenance.prompt_version
    assert 0.0 <= scores[0].semantic_similarity <= 1.0


def test_review_ranking_lexical_fallback_is_deterministic():
    project_id, run_id = "project-2", "run-2"
    seed = SeedPaper(id="seed-1", project_id=project_id, run_id=run_id, input_type="pdf", input_value="/seed.pdf", title="Graph neural networks", text="graph neural network message passing", topic="graph learning", research_question="How do GNNs generalize?", checksum="seed", methods=("message passing",), provenance=_provenance(project_id, run_id))
    paper = PaperRecord(id="paper-1", project_id=project_id, run_id=run_id, title="Graph neural networks", abstract="message passing graph networks", authors=("A",), year=2025, publication_date="2025-01-01", doi="10.1/2", arxiv_id=None, url=None, source_names=("openalex",), citation_count=3)
    review = PaperReview(id="review-1", project_id=project_id, run_id=run_id, paper_id="paper-1", problem="graph representation", research_question="How do GNNs generalize?", method="message passing", datasets=(), experimental_protocol="", results=("good",), ablations=(), limitations=(), failure_modes=(), evidence_claim_ids=("claim-1",), confidence=0.9, provenance=_provenance(project_id, run_id))

    first = _run_rank(seed, [paper], [review])
    second = _run_rank(seed, [paper], [review])

    assert first[0].total == second[0].total
    assert first[0].provenance.checksum == second[0].provenance.checksum


def _run_rank(seed, papers, reviews):
    import asyncio

    ranker = ReviewRanker(_lexical_router())
    return asyncio.run(ranker.rank(seed=seed, papers=papers, reviews=reviews, run_id=seed.run_id))
