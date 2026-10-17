"""Progress reporting: live percent + remaining tasks while skills run.

The contract: one JSON line per event on stderr —
{"progress": {"stage", "percent", "completed", "total", "detail", "remaining"}}
— while stdout remains the single machine-readable result. Percentages are
STEP counts (completed steps / total steps), never time estimates.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autoresearch.application.progress import report, reset_sink, set_sink


def test_report_is_noop_without_sink():
    report("stage", 1, 2)  # must never raise


def test_sink_receives_structured_events():
    events = []
    token = set_sink(events.append)
    try:
        report("review.papers", 8, 20, detail="8/20 papers reviewed", remaining=["novelty"])
        report("workflow.novelty", 3, 7, remaining=["experiment", "manuscript"])
    finally:
        reset_sink(token)
    assert events[0]["stage"] == "review.papers"
    assert events[0]["percent"] == 40
    assert events[0]["completed"] == 8 and events[0]["total"] == 20
    assert events[0]["remaining"] == ["novelty"]
    assert events[1]["percent"] == 43


def test_report_never_raises_when_sink_fails():
    token = set_sink(lambda event: (_ for _ in ()).throw(RuntimeError("boom")))
    try:
        report("stage", 1, 2)
    finally:
        reset_sink(token)


@pytest.mark.anyio
async def test_five_skill_loop_emits_live_progress_events(tmp_path, monkeypatch):
    """Full offline loop with a sink installed: the stage sequence must be
    monotonic per counter and every event must carry percent + remaining."""
    from test_smoke_e2e import _build_loop  # shared fixture harness

    events: list[dict] = []
    token = set_sink(events.append)
    try:
        repository, workflow, seed = _build_loop(tmp_path)
        first = await workflow.run(seed=str(seed), until="design")
        from uuid import uuid4

        candidate_id = next(
            s for s in first["stages"] if s["stage"] == "methodology"
        )["candidate_id"]
        preview = workflow.experiment.approval_preview_for_candidate(candidate_id)
        saved = repository.save_exact_approval({
            "id": uuid4().hex, "run_id": preview["run_id"],
            "specification_checksum": str(preview["checksum"]),
            "approver": "progress-test", "approved_scope": "workspace",
            "approved_target": str(preview["worker"]), "approved_budget": 60.0,
            "expires_at": "2099-01-01T00:00:00+00:00", "status": "pending",
        })
        repository.approve_exact(saved["id"], approver="progress-test")
        await workflow.run(project_id=first["project_id"], until="manuscript")
    finally:
        reset_sink(token)

    assert events, "the loop must emit progress events"
    for event in events:
        assert set(event) >= {"stage", "percent", "completed", "total", "detail", "remaining"}
        assert 0 <= event["percent"] <= 100

    workflow_stages = [event["stage"] for event in events if event["stage"].startswith("workflow.")]
    assert workflow_stages, "workflow stage events must be emitted"
    percents = [event["percent"] for event in events if event["stage"] == "manuscript.steps"]
    assert percents == sorted(percents), "manuscript progress must be monotonic"
    assert percents[-1] == 100
    review_events = [event for event in events if event["stage"] == "review.papers"]
    assert review_events and review_events[-1]["percent"] == 100
    experiment_events = [event for event in events if event["stage"] == "experiment.worker"]
    assert experiment_events and experiment_events[-1]["percent"] == 100
    assert experiment_events[-1]["remaining"] == []


def test_cli_stdout_stays_pure_result_json(tmp_path, monkeypatch):
    """Progress lines go to stderr; stdout remains exactly one JSON result."""
    monkeypatch.setenv("AUTORESEARCH_DATA_ROOT", str(tmp_path / "data"))
    runner = CliRunner()
    from autoresearch.cli.app import app

    result = runner.invoke(app, ["skill", "list"])
    assert result.exit_code == 0
    json.loads(result.stdout)  # stdout is parseable as the final result alone
    for line in result.stderr.splitlines():
        if line.strip():
            payload = json.loads(line)
            assert "progress" in payload


def test_codex_target_installs_the_five_skills(tmp_path, monkeypatch):
    monkeypatch.setenv("CODEX_SKILLS_DIR", str(tmp_path / "codex-skills"))
    runner = CliRunner()
    from autoresearch.cli.app import app

    result = runner.invoke(app, ["skill", "install", "--target", "codex"])
    assert result.exit_code == 0, result.output
    installed = sorted(p.name for p in (tmp_path / "codex-skills").iterdir() if p.is_dir())
    assert installed == ["experiment", "manuscript", "novelty", "review", "workflow"]
    listing = json.loads(runner.invoke(app, ["skill", "list"]).output)
    assert all("codex" in skill["targets"] for skill in listing["skills"])
