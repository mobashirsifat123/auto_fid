"""Regression: worker results must survive process restarts.

Live bug (2026-10-16, FIRE-Bench execution): LocalTypedWorker.run returned the
worker_jobs DB row when a job was already COMPLETED — but that row stores
bookkeeping only, never stdout. The second execution therefore handed the
measurement ingester an empty stdout ("MEASUREMENT_DOCUMENT_MISSING") after the
first process's in-memory result died with it. Results are now persisted under
run_outputs/ and reloaded on the reuse path; a completed job with no stored
result (legacy/in-memory-lost) re-executes the same approved deterministic
action instead of returning an unusable row.
"""

from __future__ import annotations

import asyncio
from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, ExperimentSpec, PipelineKind
from autoresearch.experiments.policy import ActionType, ExecutionAction, specification_checksum
from autoresearch.storage.repository import ResearchRepository
from autoresearch.workers.local_typed import LocalTypedWorker


def _setup(tmp_path):
    workspace = tmp_path / "workspace"
    (workspace / "scripts").mkdir(parents=True)
    (workspace / "scripts" / "echo_experiment.py").write_text(
        "import json\n"
        "print(json.dumps({'identity': {'seed': 1}, 'seeds': ["
        "{'seed': 1, 'baseline': {'m': 1.0}, 'candidate': {'m': 2.0}}]}))\n",
        encoding="utf-8",
    )
    repository = ResearchRepository(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    project = repository.create_project("worker result persistence")
    run = repository.create_run(project.id, PipelineKind.EXPERIMENT, {})
    from autoresearch.domain.models import CandidateStatus, NoveltyCandidate
    candidate = NoveltyCandidate(
        id=uuid4().hex, project_id=project.id, run_id=run.id, gap_id="g",
        strategy="test", method="m", causal_mechanism="cm", target_limitation="tl",
        difference_from_existing="d", expected_effect="e", falsification_condition="f",
        minimum_validating_experiment="mv", required_ablations=(),
        implementation_complexity="low", expected_compute="low", risks_and_objections=(),
        status=CandidateStatus.GENERATED,
        provenance=ArtifactProvenance.create(
            project_id=project.id, run_id=run.id, source_evidence=(),
            provider="deterministic", model="test", prompt_version="test", content="c",
        ),
    )
    repository.save_novelty_candidate(candidate)
    spec = ExperimentSpec(
        id=uuid4().hex, project_id=project.id, run_id=run.id, candidate_id=candidate.id,
        hypothesis="h", causal_mechanism="cm", falsification_condition="f",
        repository=str(workspace.resolve()), baseline_commit="none",
        dataset_identity="synthetic", dataset_checksum="d", split_definition="s",
        baseline_configuration="b", proposed_change="p",
        independent_variables=(), controlled_variables=(), metrics=("m",),
        random_seeds=(1,), ablations=(), robustness_tests=(),
        statistical_plan="paired", compute_budget="trivial",
        stopping_conditions=(), expected_outputs=(), artifact_contract="a",
        provenance=ArtifactProvenance.create(
            project_id=project.id, run_id=run.id, source_evidence=(),
            provider="deterministic", model="test", prompt_version="test", content="x",
        ),
    )
    repository.save_experiment_spec(spec)
    action = ExecutionAction(ActionType.RUN_PYTHON_SCRIPT, "scripts/echo_experiment.py", ())
    approved = specification_checksum(spec, (action,), "local")
    return repository, workspace, spec, action, approved


def test_completed_job_result_survives_process_restart(tmp_path):
    repository, workspace, spec, action, approved = _setup(tmp_path)
    worker1 = LocalTypedWorker(repository, workspace)
    first = asyncio.run(worker1.run(spec, (action,), approved_checksum=approved))
    assert first["state"] == "COMPLETED"
    assert '"seeds"' in str(first["stdout"])

    # A fresh worker (new process) must reuse the PERSISTED result, not the
    # stdout-less worker_jobs row.
    worker2 = LocalTypedWorker(repository, workspace)
    second = asyncio.run(worker2.run(spec, (action,), approved_checksum=approved))
    assert second["state"] == "COMPLETED"
    assert second["stdout"] == first["stdout"]
    assert '"seeds"' in str(second["stdout"])


def test_completed_job_without_stored_result_reexecutes(tmp_path):
    repository, workspace, spec, action, approved = _setup(tmp_path)
    worker1 = LocalTypedWorker(repository, workspace)
    first = asyncio.run(worker1.run(spec, (action,), approved_checksum=approved))
    assert first["state"] == "COMPLETED"

    # Simulate a legacy/lost result payload: the COMPLETED row exists but no
    # stored result file. The worker must re-execute, not return an empty row.
    from autoresearch.workers.local_typed import LocalTypedWorker as _W
    store = workspace / "run_outputs" / f"worker_result_{first['job_id']}.json"
    assert store.is_file()
    store.unlink()

    worker2 = LocalTypedWorker(repository, workspace)
    second = asyncio.run(worker2.run(spec, (action,), approved_checksum=approved))
    assert second["state"] == "COMPLETED"
    assert '"seeds"' in str(second["stdout"])
    assert int(second.get("attempt") or 0) >= 1  # re-executed, not cached