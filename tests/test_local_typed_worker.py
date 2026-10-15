import asyncio
from pathlib import Path

import pytest

from autoresearch.domain.models import ArtifactProvenance, CandidateStatus, ExperimentSpec, GapClassification
from autoresearch.experiments.policy import ActionType, ExecutionAction, specification_checksum
from autoresearch.storage.repository import ResearchRepository
from autoresearch.workers.local_typed import LocalTypedWorker


def spec(tmp_path: Path) -> ExperimentSpec:
    provenance = ArtifactProvenance.create(project_id="p", run_id="r", source_evidence=["c"], provider="test", model="test", prompt_version="test", content={})
    return ExperimentSpec("s", "p", "r", "c", "h", "m", "f", str(tmp_path), "baseline", "dataset", "checksum", "split", "base", "change", (), (), ("score",), (1, 2, 3), (), (), "paired", "1", (), (), "metrics.json", provenance, {"score": "maximize"}, {"score": 0.1}, 3, "disabled", "local", True)


def persist_dependencies(repo: ResearchRepository, project_id: str, run_id: str) -> None:
    repo.execute("INSERT INTO research_gaps(id,project_id,run_id,statement,supporting_paper_ids_json,supporting_claim_ids_json,why_unsolved,missing_experiment,falsification_test,confidence,counter_evidence_claim_ids_json,classification,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", ("gap", project_id, run_id, "gap", "[]", "[]", "why", "exp", "test", 0.5, "[]", GapClassification.HYPOTHESIS.value, "{}"))
    repo.execute("INSERT INTO novelty_candidates(id,project_id,run_id,gap_id,strategy,method,causal_mechanism,target_limitation,difference_from_existing,expected_effect,falsification_condition,minimum_validating_experiment,required_ablations_json,implementation_complexity,expected_compute,risks_and_objections_json,status,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ("c", project_id, run_id, "gap", "s", "m", "cause", "limit", "diff", "effect", "fals", "exp", "[]", "low", "small", "[]", CandidateStatus.GENERATED.value, "{}"))


@pytest.mark.asyncio
async def test_local_worker_executes_allowlisted_action_and_deduplicates(tmp_path: Path):
    repo = ResearchRepository(tmp_path / "db.sqlite", tmp_path / "artifacts")
    project = repo.create_project("worker")
    run = repo.create_run(project.id, __import__("autoresearch.domain.models", fromlist=["PipelineKind"]).PipelineKind.EXPERIMENT, {})
    persist_dependencies(repo, project.id, run.id)
    target = tmp_path / "workspace"
    target.mkdir()
    (target / "fixture.py").write_text("print('offline-ok')\n", encoding="utf-8")
    experiment = spec(target)
    experiment = ExperimentSpec(**{**experiment.__dict__, "project_id": project.id, "run_id": run.id})
    repo.save_experiment_spec(experiment)
    action = ExecutionAction(ActionType.RUN_PYTHON_SCRIPT, "fixture.py")
    worker = LocalTypedWorker(repo, target)
    checksum = specification_checksum(experiment, (action,), "local")
    first = await worker.run(experiment, (action,), approved_checksum=checksum, timeout_seconds=30)
    second = await worker.run(experiment, (action,), approved_checksum=checksum, timeout_seconds=30)
    assert first["state"] == "COMPLETED"
    assert "offline-ok" in str(first["stdout"])
    assert second["job_id"] == first["job_id"]
    assert len(repo.worker_jobs_for_run(run.id)) == 1


def test_local_worker_rejects_checksum_and_escape(tmp_path: Path):
    repo = ResearchRepository(tmp_path / "db.sqlite", tmp_path / "artifacts")
    project = repo.create_project("worker")
    run = repo.create_run(project.id, __import__("autoresearch.domain.models", fromlist=["PipelineKind"]).PipelineKind.EXPERIMENT, {})
    persist_dependencies(repo, project.id, run.id)
    target = tmp_path / "workspace"
    target.mkdir()
    experiment = spec(target)
    experiment = ExperimentSpec(**{**experiment.__dict__, "project_id": project.id, "run_id": run.id})
    repo.save_experiment_spec(experiment)
    worker = LocalTypedWorker(repo, target)
    action = ExecutionAction(ActionType.RUN_PYTHON_SCRIPT, "../escape.py")
    with pytest.raises(PermissionError, match="checksum_mismatch"):
        worker.submit(experiment, (action,), approved_checksum="0" * 64)
    safe = ExecutionAction(ActionType.RUN_PYTHON_SCRIPT, "missing.py")
    checksum = specification_checksum(experiment, (safe,), "local")
    with pytest.raises(ValueError, match="path"):
        worker.submit(experiment, (safe,), approved_checksum=checksum)
