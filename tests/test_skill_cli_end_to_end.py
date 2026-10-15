"""End-to-end CLI verification of the four agent skills (review, novelty,
experiment, workflow): the exact command sequences documented in
src/autoresearch/skills/*/SKILL.md are driven through the real Typer CLI with
deterministic providers, asserting exit codes and the JSON contract at every
step. No network, no paid API, real worker execution of a fixture script.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

import test_workflow_coordination as fixtures
from autoresearch.cli.skills import skill_app

runner = CliRunner()


class _FakeContainer:
    """Duck-typed ApplicationContainer over the deterministic harness."""

    def __init__(self, tmp_path: Path):
        repository, workflow, worker, seed, workspace = fixtures._harness(tmp_path)
        self.repository = repository
        self.workflow = workflow
        self.worker = worker
        self.discovery = workflow.discovery
        self.review = workflow.review
        self.novelty = workflow.novelty
        self.experiment = workflow.experiment
        self.reinvention = workflow.reinvention

    async def close(self) -> None:
        return None


@pytest.fixture()
def container(tmp_path, monkeypatch):
    fake = _FakeContainer(tmp_path)
    import autoresearch.cli.skills as skills_module

    monkeypatch.setattr(skills_module, "ApplicationContainer", lambda _settings: fake)
    return fake


def _json(result):
    return json.loads(result.output)


def test_workflow_skill_full_cycle_and_experiment_skill_lifecycle(tmp_path, container):
    catalog = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "catalog_example.json"

    # ---- workflow skill: seed → design ----
    result = runner.invoke(skill_app, ["workflow", "--from-catalog", str(catalog), "--until", "design"])
    assert result.exit_code == 0, result.output
    payload = _json(result)
    assert payload["ok"] and payload["stopped_at"] == "design"
    stages = {s["stage"]: s for s in payload["stages"]}
    assert stages["review"]["status"] == "completed"
    assert stages["methodology"]["methodology_freeze_id"]
    spec_id = stages["design"]["spec_id"]
    candidate_id = stages["methodology"]["candidate_id"]
    project_id = payload["project_id"]

    # ---- workflow skill: resume → experiment without approval ----
    result = runner.invoke(skill_app, ["workflow", "--project", project_id, "--until", "experiment"])
    assert result.exit_code == 0, result.output
    payload = _json(result)
    experiment_stage = next(s for s in payload["stages"] if s["stage"] == "experiment")
    assert experiment_stage["status"] == "awaiting_approval"
    assert experiment_stage["checksum"]
    assert container.repository.rows("SELECT * FROM worker_jobs") == []

    # ---- experiment skill: execute before approval refuses cleanly (exit 3) ----
    result = runner.invoke(skill_app, ["experiment", "execute", "--candidate-id", candidate_id])
    assert result.exit_code == 3, result.output
    assert _json(result)["error"]["code"] == "approval_required"

    # ---- experiment skill: plan (idempotent spec) ----
    result = runner.invoke(skill_app, ["experiment", "plan", "--candidate-id", candidate_id])
    assert result.exit_code == 0, result.output
    payload = _json(result)
    assert payload["status"] == "planned" and payload["executed"] is False
    assert payload["approval"]["checksum"]
    checksum = payload["approval"]["checksum"]

    # ---- experiment skill: approve preview (no side effect) then accept ----
    result = runner.invoke(skill_app, ["experiment", "approve", "--candidate-id", candidate_id])
    assert result.exit_code == 0 and _json(result)["approved"] is False

    result = runner.invoke(skill_app, ["experiment", "approve", "--candidate-id", candidate_id, "--accept"])
    assert result.exit_code == 0, result.output
    payload = _json(result)
    assert payload["approved"] is True and payload["checksum"] == checksum

    # ---- experiment skill: execute (approved) completes a real worker job ----
    result = runner.invoke(skill_app, ["experiment", "execute", "--candidate-id", candidate_id])
    assert result.exit_code == 0, result.output
    payload = _json(result)
    jobs = payload["jobs"]
    assert jobs and jobs[0]["state"] == "COMPLETED"
    job_id = jobs[0]["id"]

    # duplicate dispatch: identical approved execution returns the same job
    result = runner.invoke(skill_app, ["experiment", "execute", "--candidate-id", candidate_id])
    assert result.exit_code == 0, result.output
    assert _json(result)["jobs"][0]["id"] == job_id
    assert len(container.repository.rows("SELECT id FROM worker_jobs")) == 1

    # ---- experiment skill: honest report boundary (exit 5, clean JSON) ----
    result = runner.invoke(skill_app, ["experiment", "report", "--candidate-id", candidate_id])
    assert result.exit_code == 5, result.output
    assert _json(result)["error"]["code"] == "insufficient_evidence"


def test_review_and_novelty_skills_on_a_second_project(tmp_path, container):
    catalog = Path(__file__).resolve().parents[1] / "evals" / "fixtures" / "catalog_example.json"

    # workflow to review produces the gap the review/novelty skills consume
    result = runner.invoke(skill_app, ["workflow", "--from-catalog", str(catalog), "--until", "review"])
    assert result.exit_code == 0, result.output
    project_id = _json(result)["project_id"]

    # ---- review skill: re-review an existing corpus ----
    result = runner.invoke(skill_app, ["review", "--project", project_id, "--top-k", "3"])
    assert result.exit_code == 0, result.output
    payload = _json(result)
    assert payload["gap"]["id"]
    gap_id = payload["gap"]["id"]

    # ---- novelty skill: portfolio for a specific gap ----
    result = runner.invoke(skill_app, ["novelty", "--project", project_id, "--gap-id", gap_id])
    assert result.exit_code == 0, result.output
    payload = _json(result)
    assert payload["portfolio_id"] and payload["gap_id"] == gap_id
    assert payload["low_risk_candidate_id"] in payload["candidate_ids"]

    # ---- status skill: projects visible ----
    result = runner.invoke(skill_app, ["status"])
    assert result.exit_code == 0, result.output
    assert any(p["project_id"] == project_id for p in _json(result)["projects"])


def test_skill_cli_argument_contracts(tmp_path, container):
    # exactly one of seed/catalog/project
    result = runner.invoke(skill_app, ["workflow", "--from-catalog", "x", "--project", "y"])
    assert result.exit_code == 2
    # unknown stop stage
    result = runner.invoke(skill_app, ["workflow", "--project", "p", "--until", "bogus"])
    assert result.exit_code != 0
    # unknown candidate
    result = runner.invoke(skill_app, ["experiment", "plan", "--candidate-id", "missing"])
    assert result.exit_code == 4
    assert _json(result)["error"]["code"] == "novelty_candidate_not_found"
