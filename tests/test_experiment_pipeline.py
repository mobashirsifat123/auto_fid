"""Tests for the experiment planning and execution pipeline."""

import pytest
from autoresearch.domain.models import CandidateStatus
from autoresearch.experiments.environment import IsolatedEnvironmentBuilder
from autoresearch.experiments.executor import DurableExperimentExecutor
from autoresearch.experiments.pipeline import ExperimentPipeline
from autoresearch.experiments.planner import ExperimentSpecGenerator
from autoresearch.experiments.report_writer import ScientificReportWriter
from autoresearch.experiments.result_analysis import StatisticalAnalyzer
from autoresearch.providers.router import ModelRouter
from autoresearch.providers.testing import FakeLLMProvider
from autoresearch.storage.repository import ResearchRepository


@pytest.mark.anyio
async def test_experiment_pipeline(tmp_path):
    repo = ResearchRepository(tmp_path / "state.db", tmp_path / "artifacts")
    repo.migrate()
    
    project = repo.create_project("Experiment Test Project")
    repo.execute("INSERT INTO pipeline_runs(id,project_id,kind,status,input_json,started_at) VALUES(?,?,?,?,?,?)", ("run1", project.id, "novelty", "completed", "{}", "2026-01-01"))
    from autoresearch.domain.models import GapClassification
    repo.execute(
        "INSERT INTO research_gaps(id,project_id,run_id,statement,supporting_paper_ids_json,supporting_claim_ids_json,why_unsolved,missing_experiment,falsification_test,confidence,counter_evidence_claim_ids_json,classification,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("gap1", project.id, "run1", "No one has scaled this.", '["paper1"]', '["claim1"]', "Cost.", "Scale test.", "Fails if cost > X.", 0.9, "[]", GapClassification.EVIDENCE_SUPPORTED.value, '{"project_id": "test", "run_id": "test", "source_evidence": [], "provider": "fake", "model": "fake", "prompt_version": "v1", "created_at": "2026-01-01T00:00:00Z", "checksum": "sum"}')
    )
    repo.execute(
        "INSERT INTO novelty_candidates(id,project_id,run_id,gap_id,strategy,method,causal_mechanism,target_limitation,difference_from_existing,expected_effect,falsification_condition,minimum_validating_experiment,required_ablations_json,implementation_complexity,expected_compute,risks_and_objections_json,status,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("cand1", project.id, "run1", "gap1", "strategy1", "method1", "cause1", "limit1", "diff1", "effect1", "fals1", "min_exp1", "[]", "low", "1 gpu", "[]", CandidateStatus.GENERATED.value, '{"project_id": "test", "run_id": "test", "source_evidence": [], "provider": "fake", "model": "fake", "prompt_version": "v1", "created_at": "2026-01-01T00:00:00Z", "checksum": "sum"}')
    )
    
    def _responder(messages):
        prompt = str(messages[-1]["content"])
        if "Seeds required:" in prompt:
            return {"selected_index": 0, "selection_rationale": "Design 0 is maximally falsifiable with adequate power."}
        if "Candidate Method:" in prompt and "Falsification Condition:" in prompt:
            return {
                "what_would_disprove": "No consistent improvement.",
                "confounding_factors": ["Seed variance"],
                "minimal_viable_experiment": "One dataset, three seeds.",
                "expected_null_outcome": "No consistent gain.",
                "failure_modes": ["Implementation error"]
            }
        if "Candidate Method:" in prompt or "experiment-planner-v1" in prompt:
            return {
                "hypothesis": "H1",
                "causal_mechanism": "cause",
                "falsification_condition": "cond",
                "repository": "repo",
                "baseline_commit": "main",
                "dataset_identity": "data",
                "dataset_checksum": "sum",
                "split_definition": "split",
                "baseline_configuration": "base",
                "proposed_change": "change",
                "independent_variables": [],
                "controlled_variables": [],
                "metrics": [],
                "random_seeds": [42],
                "ablations": [],
                "robustness_tests": [],
                "statistical_plan": "plan",
                "compute_budget": "budget",
                "stopping_conditions": [],
                "expected_outputs": [],
                "artifact_contract": "contract"
            }
        if "Baseline Metrics:" in prompt or "result-analyzer-v1" in prompt:
            return {
                "failure_case_analysis": "None",
                "hypothesis_support_assessment": "Supported"
            }
        if "Assessment:" in prompt or "report-writer-v1" in prompt:
            return {
                "literature_motivation": "Lit",
                "research_gap": "Gap",
                "implementation_diff": "Diff",
                "uncertainty": "Unc",
                "ablations": "Abl",
                "robustness_tests": "Rob",
                "limitations": "Lim",
                "reproducibility_instructions": "Rep",
                "decision": "supported",
                "decision_evidence": "Evid"
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
    
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    
    from autoresearch.application.services import ExperimentService
    service = ExperimentService(repo, router, workspace)
    
    spec_id = await service.execute_candidate("cand1", mode="plan")
    
    assert spec_id is not None
    spec = repo.row("SELECT * FROM experiment_specs WHERE id=?", (spec_id,))
    assert spec is not None
    assert repo.row("SELECT * FROM execution_jobs LIMIT 1") is None
