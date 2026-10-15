"""Tests for the scientific reinvention pipeline and service."""

import pytest
from autoresearch.application.services import ReinventionService
from autoresearch.domain.models import CandidateStatus, ExperimentDecision, GapClassification, ReinventionAction
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.config import ModelAssignment
from autoresearch.providers.router import ModelRouter
from autoresearch.providers.testing import FakeLLMProvider
from autoresearch.reinvention.pipeline import ReinventionPipeline
from autoresearch.storage.repository import ResearchRepository


@pytest.mark.anyio
async def test_reinvention_pipeline_refine_flow(tmp_path):
    repo = ResearchRepository(tmp_path / "state.db", tmp_path / "artifacts")
    repo.migrate()

    project = repo.create_project("Reinvention Test Project")
    repo.execute("INSERT INTO pipeline_runs(id,project_id,kind,status,input_json,started_at) VALUES(?,?,?,?,?,?)", ("run1", project.id, "experiment", "completed", "{}", "2026-01-01"))

    # Seed research gap
    repo.execute(
        "INSERT INTO research_gaps(id,project_id,run_id,statement,supporting_paper_ids_json,supporting_claim_ids_json,why_unsolved,missing_experiment,falsification_test,confidence,counter_evidence_claim_ids_json,classification,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("gap1", project.id, "run1", "No one has scaled this.", '["paper1"]', '["claim1"]', "Cost.", "Scale test.", "Fails if cost > X.", 0.9, "[]", GapClassification.EVIDENCE_SUPPORTED.value, '{"project_id": "test", "run_id": "test", "source_evidence": [], "provider": "fake", "model": "fake", "prompt_version": "v1", "created_at": "2026-01-01T00:00:00Z", "checksum": "sum"}')
    )

    # Seed parent candidate
    repo.execute(
        "INSERT INTO novelty_candidates(id,project_id,run_id,gap_id,strategy,method,causal_mechanism,target_limitation,difference_from_existing,expected_effect,falsification_condition,minimum_validating_experiment,required_ablations_json,implementation_complexity,expected_compute,risks_and_objections_json,status,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("cand1", project.id, "run1", "gap1", "strategy1", "Adaptive fusion v1", "Attention routing", "OOM errors", "No OOM", "+15% accuracy", "Fails if acc < 0.8", "Run on GPU", "[]", "low", "1 gpu hour", "[]", CandidateStatus.GENERATED.value, '{"project_id": "test", "run_id": "test", "source_evidence": [], "provider": "fake", "model": "fake", "prompt_version": "v1", "created_at": "2026-01-01T00:00:00Z", "checksum": "sum"}')
    )

    # Seed experiment spec
    repo.execute(
        "INSERT INTO experiment_specs(id,project_id,run_id,candidate_id,hypothesis,causal_mechanism,falsification_condition,repository,baseline_commit,dataset_identity,dataset_checksum,split_definition,baseline_configuration,proposed_change,independent_variables_json,controlled_variables_json,metrics_json,random_seeds_json,ablations_json,robustness_tests_json,statistical_plan,compute_budget,stopping_conditions_json,expected_outputs_json,artifact_contract,provenance_json,metric_direction_json,minimum_meaningful_effect_json,required_seed_count,network_policy,target_worker,approval_required) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("spec1", project.id, "run1", "cand1", "H1: Adaptive fusion improves acc", "Attention routing", "Fails if acc < 0.8", "/tmp", "main", "cifar", "sum", "train/val", "base", "change", "[]", "[]", '["accuracy"]', '[42, 43]', "[]", "[]", "t-test", "1 hr", "[]", "[]", "contract", '{"project_id": "test", "run_id": "test", "source_evidence": [], "provider": "fake", "model": "fake", "prompt_version": "v1", "created_at": "2026-01-01T00:00:00Z", "checksum": "sum"}', "{}", "{}", 2, "disabled", "local", 1)
    )

    # Seed statistical analysis
    repo.execute(
        "INSERT INTO statistical_analyses(id,project_id,run_id,spec_id,baseline_metrics_json,proposed_metrics_json,effect_size_json,confidence_intervals_json,seed_level_results_json,ablation_contributions_json,parameter_count_diff,flops_latency_diff,training_inference_cost,failure_case_analysis,hypothesis_support_assessment,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("analysis1", project.id, "run1", "spec1", '{"accuracy": 0.70}', '{"accuracy": 0.85}', '{"accuracy": 0.15}', '{"accuracy": [0.10, 0.20]}', '[]', '{}', 0, 0.0, 1.0, "None", "ESTIMATE", '{"project_id": "test", "run_id": "test", "source_evidence": [], "provider": "fake", "model": "fake", "prompt_version": "v1", "created_at": "2026-01-01T00:00:00Z", "checksum": "sum"}')
    )

    # Seed experiment report
    repo.execute(
        "INSERT INTO experiment_reports(id,project_id,run_id,analysis_id,literature_motivation,research_gap,hypothesis,proposed_method,implementation_diff,lineage,dataset_split,exact_commands_json,baseline_results_json,aggregated_results_json,uncertainty,ablations,robustness_tests,failed_runs,cost,limitations,reproducibility_instructions,decision,decision_evidence,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("report1", project.id, "run1", "analysis1", "Lit", "Gap", "H1", "Adaptive fusion v1", "diff", "lineage", "split", '["train"]', '{"accuracy": 0.70}', '{"accuracy": 0.85}', "Low", "None", "None", 0, 1.0, "Slight memory overhead at batch 128", "Run script", ExperimentDecision.SUPPORTED.value, "Accuracy improved from 70% to 85%", '{"project_id": "test", "run_id": "test", "source_evidence": [], "provider": "fake", "model": "fake", "prompt_version": "v1", "created_at": "2026-01-01T00:00:00Z", "checksum": "sum"}')
    )

    def _responder(messages):
        return {
            "action": "refine",
            "root_cause_diagnosis": "Attention routing successfully concentrated gradient flow on informative features.",
            "empirical_lessons": ["Accuracy increased by +15%", "Memory overhead scales quadratically with sequence length"],
            "next_generation_hypothesis": "H2: Sparse attention routing will maintain +15% accuracy while eliminating memory overhead.",
            "revised_mechanism": "Sparse attention routing with top-k gating",
            "actionable_next_steps": ["Implement sparse top-k mask", "Evaluate at sequence length 2048"],
            "revised_method_name": "Sparse Adaptive Fusion v2",
            "revised_difference_statement": "Introduces top-k sparsity to the parent adaptive attention mechanism.",
            "expected_effect": "+15% accuracy with 4x memory savings"
        }

    provider = FakeLLMProvider(responder=_responder)

    from autoresearch.providers.registry import ProviderRegistry

    registry = ProviderRegistry()
    registry.register_llm("fake", provider)

    assignments = {
        ModelRole.RESEARCH_REASONING: ModelAssignment(ModelRole.RESEARCH_REASONING, "fake", "fake-model"),
        ModelRole.SCIENTIFIC_CRITIC: ModelAssignment(ModelRole.SCIENTIFIC_CRITIC, "fake", "fake-model"),
    }
    router = ModelRouter(registry, assignments)

    service = ReinventionService(repo, router)
    result = await service.reinvent("cand1")

    assert result["project_id"] == project.id
    assert result["action"] == "refine"
    assert result["next_candidate_id"] is not None
    assert result["next_candidate_method"] == "Sparse Adaptive Fusion v2"

    # Verify database persistence
    db_reinvention = repo.get_reinvention_analysis_by_id(result["reinvention_id"])
    assert db_reinvention is not None
    assert db_reinvention.action == ReinventionAction.REFINE
    assert db_reinvention.next_candidate_id == result["next_candidate_id"]

    next_candidate = repo.get_novelty_candidate(result["next_candidate_id"])
    assert next_candidate is not None
    assert next_candidate.method == "Sparse Adaptive Fusion v2"
    assert next_candidate.causal_mechanism == "Sparse attention routing with top-k gating"
