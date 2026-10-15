from pathlib import Path

import pytest

from autoresearch.domain.models import ArtifactProvenance, ExecutionResult, ExperimentSpec
from autoresearch.experiments.integrity import IntegrityFailure, PredictedMetric, ProvenanceType
from autoresearch.experiments.result_analysis import StatisticalAnalyzer, exact_wilcoxon_two_sided


def _spec(tmp_path: Path, *, seeds: int = 5) -> ExperimentSpec:
    provenance = ArtifactProvenance.create(project_id="p", run_id="r", source_evidence=["candidate"], provider="test", model="test", prompt_version="test", content={})
    return ExperimentSpec("spec", "p", "r", "candidate", "hypothesis", "mechanism", "falsify", str(tmp_path), "commit", "dataset", "dataset-sum", "split-sum", "base", "change", (), (), ("map",), tuple(range(seeds)), (), (), "two-sided Wilcoxon", "0", (), ("train.py",), "strict", provenance, {"map": "maximize"}, {"map": .05}, seeds)


def _identity(seed: int | None = None) -> dict[str, object]:
    value: dict[str, object] = {"candidate_id": "candidate", "method_id": "example_method", "repository_commit": "commit", "implementation_hash": "impl", "config_hash": "config", "dataset_checksum": "dataset-sum", "split_checksum": "split-sum", "metric_version": "v1"}
    if seed is not None:
        value["seed"] = seed
    return value


def _artifacts() -> list[dict[str, object]]:
    return [{"id": kind, "artifact_type": kind, "verified": True, "provenance_type": "MEASURED"} for kind in (
        "implementation_diff", "training_log", "checkpoint", "seed_metrics", "parameter_profile", "flop_profile", "latency_profile", "evaluation_report")]


def _result(spec: ExperimentSpec, *, count: int = 5, candidate_id: str = "candidate", fixture: bool = False) -> ExecutionResult:
    identity = _identity(); identity["candidate_id"] = candidate_id
    rows = [{"seed": i, "identity": _identity(i), "baseline": {"map": 100.0}, "candidate": {"map": 101.83}, "provenance_type": "FIXTURE" if fixture else "MEASURED"} for i in range(count)]
    return ExecutionResult("result", "p", "r", "job", 0, "", "", {"identity": identity, "seeds": rows, "artifacts": _artifacts()}, {}, (), None, spec.provenance)


@pytest.mark.anyio
async def test_pytest_validation_is_not_a_scientific_result(tmp_path: Path):
    from autoresearch.application.services import ExperimentService
    from dataclasses import replace
    spec = replace(_spec(tmp_path), expected_outputs=())
    service = object.__new__(ExperimentService)
    with pytest.raises(ValueError, match="experiment_action_missing"):
        service.action_plan(spec)


@pytest.mark.anyio
async def test_incompatible_method_identity_is_rejected(tmp_path: Path):
    spec = _spec(tmp_path)
    with pytest.raises(IntegrityFailure, match="CACHE_INCOMPATIBLE"):
        await StatisticalAnalyzer().analyze(spec, _result(spec, candidate_id="f8-fastkan"))


@pytest.mark.anyio
async def test_three_seeds_cannot_satisfy_five_seed_spec(tmp_path: Path):
    spec = _spec(tmp_path, seeds=5)
    with pytest.raises(IntegrityFailure, match="INCOMPLETE_EXPERIMENT"):
        await StatisticalAnalyzer().analyze(spec, _result(spec, count=3))


@pytest.mark.anyio
async def test_fixture_values_are_rejected_from_production_analysis(tmp_path: Path):
    spec = _spec(tmp_path)
    with pytest.raises(IntegrityFailure, match="FIXTURE_PROHIBITED"):
        await StatisticalAnalyzer().analyze(spec, _result(spec, fixture=True))


@pytest.mark.anyio
async def test_audited_gates_refute_not_support(tmp_path: Path):
    spec = _spec(tmp_path)
    # +1.83% mAP does not clear a +5% gate.  The old system reported this as supported.
    analysis = await StatisticalAnalyzer().analyze(spec, _result(spec))
    assert analysis.hypothesis_support_assessment == "REFUTED"
    assert analysis.decision_gates == {"map_gate": False}


def test_model_prediction_cannot_be_observed_metric():
    prediction = PredictedMetric(40, ProvenanceType.MODEL_HYPOTHESIS, "model", None, "now", False)
    assert prediction.provenance_type is ProvenanceType.MODEL_HYPOTHESIS


def test_wilcoxon_is_computed_not_claimed():
    result = exact_wilcoxon_two_sided([1, 1, 1, 1, 1], [1.0183] * 5)
    assert result["test"] == "wilcoxon_signed_rank"
    assert result["p_value"] >= 0.05
