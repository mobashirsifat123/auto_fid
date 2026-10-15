"""Tests for the novelty generation and selection pipeline."""

import pytest
from autoresearch.domain.models import GapClassification
from autoresearch.novelty.candidate_generator import LLMCandidateGenerator
from autoresearch.novelty.novelty_critic import ScientificCritic
from autoresearch.novelty.pipeline import NoveltyPipeline
from autoresearch.novelty.prior_art_checker import EvidenceBackedPriorArtChecker
from autoresearch.novelty.selection import PortfolioSelector
from autoresearch.providers.router import ModelRouter
from autoresearch.providers.testing import FakeLLMProvider
from autoresearch.storage.repository import ResearchRepository


@pytest.mark.anyio
async def test_novelty_pipeline(tmp_path):
    repo = ResearchRepository(tmp_path / "state.db", tmp_path / "artifacts")
    repo.migrate()
    
    project = repo.create_project("Novelty Test Project")
    repo.execute("INSERT INTO pipeline_runs(id,project_id,kind,status,input_json,started_at) VALUES(?,?,?,?,?,?)", ("run1", project.id, "novelty", "completed", "{}", "2026-01-01"))
    repo.execute(
        "INSERT INTO research_gaps(id,project_id,run_id,statement,supporting_paper_ids_json,supporting_claim_ids_json,why_unsolved,missing_experiment,falsification_test,confidence,counter_evidence_claim_ids_json,classification,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("gap1", project.id, "run1", "No one has scaled this.", '["paper1"]', '["claim1"]', "Cost.", "Scale test.", "Fails if cost > X.", 0.9, "[]", GapClassification.EVIDENCE_SUPPORTED.value, '{"project_id": "test", "run_id": "test", "source_evidence": [], "provider": "fake", "model": "fake", "prompt_version": "v1", "created_at": "2026-01-01T00:00:00Z", "checksum": "sum"}')
    )
    
    
    def _responder(messages):
        prompt = str(messages[-1]["content"])
        if "Research Gap:" in prompt or "limitation root cause elimination" in prompt or "candidate-gen-v1" in prompt:
            return {
                "strategy": "limitation root cause elimination",
                "method": "New method",
                "causal_mechanism": "X causes Y",
                "target_limitation": "OOM errors",
                "difference_from_existing": "No OOM.",
                "expected_effect": "Faster training",
                "falsification_condition": "OOM happens.",
                "minimum_validating_experiment": "Train on 1 GPU.",
                "required_ablations": ["No norm"],
                "implementation_complexity": "Low",
                "expected_compute": "1 GPU hour",
                "risks_and_objections": ["Might overfit."]
            }
        if "Prior Art Abstracts" in prompt or "prior-art-v1" in prompt:
            return {
                "overlap_status": "potentially_novel",
                "mechanism_comparison": "Ours is better.",
                "supporting_excerpts": [],
                "unresolved_overlap": "None."
            }
        if "Method:" in prompt or "novelty-critic-v1" in prompt:
            return {
                "novelty_score": 0.8,
                "significance_score": 0.8,
                "causal_plausibility_score": 0.8,
                "falsifiability_score": 0.8,
                "experimental_discriminability_score": 0.8,
                "feasibility_score": 0.8,
                "implementation_risk_score": 0.2,
                "expected_information_gain_score": 0.8,
                "compute_efficiency_score": 0.9,
                "publication_potential_score": 0.9,
                "objections": [],
                "failure_scenarios": []
            }
        
        raise AssertionError(f"Unexpected prompt: {prompt[:100]}")
        
    provider = FakeLLMProvider(responder=_responder)
    
    from autoresearch.providers.registry import ProviderRegistry
    from autoresearch.providers.capabilities import ModelRole
    from autoresearch.providers.config import ModelAssignment
    
    registry = ProviderRegistry()
    registry.register_llm("fake", provider)
    
    assignments = {
        ModelRole.RESEARCH_REASONING: ModelAssignment(ModelRole.RESEARCH_REASONING, "fake", "fake-model"),
        ModelRole.SCIENTIFIC_CRITIC: ModelAssignment(ModelRole.SCIENTIFIC_CRITIC, "fake", "fake-model"),
    }
    router = ModelRouter(registry, assignments)
    
    class MockSearchPipeline:
        pass
        
    pipeline = NoveltyPipeline(
        repository=repo,
        generator=LLMCandidateGenerator(router),
        prior_art_checker=EvidenceBackedPriorArtChecker(router, MockSearchPipeline()),
        critic=ScientificCritic(router),
        selector=PortfolioSelector(repo)
    )
    
    gap = repo.row("SELECT * FROM research_gaps WHERE id=?", ("gap1",))
    from autoresearch.application.services import NoveltyService
    
    service = NoveltyService(repo, router, MockSearchPipeline())
    portfolio_id = await service.generate_portfolio(project.id)
    
    assert portfolio_id is not None
    portfolio = repo.row("SELECT * FROM novelty_portfolios WHERE id=?", (portfolio_id,))
    assert portfolio is not None
    assert portfolio["low_risk_candidate_id"]
    assert portfolio["balanced_candidate_id"]
    assert portfolio["high_risk_candidate_id"]
