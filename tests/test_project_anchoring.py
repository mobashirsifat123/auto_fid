"""Project state anchoring: each project remembers the directory its data
lives in; resume works from any working directory and projects never bleed
into each other's state (or into the source checkout)."""

from __future__ import annotations

from pathlib import Path

import pytest

import test_smoke_e2e as smoke


@pytest.fixture
def registry(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_REGISTRY_PATH", str(tmp_path / "registry.json"))
    return tmp_path / "registry.json"


@pytest.mark.anyio
async def test_project_records_data_root_and_resolve_anchors(tmp_path, monkeypatch, registry):
    dir_a = tmp_path / "project_a"
    dir_a.mkdir()
    repository, workflow, seed = smoke._build_loop(dir_a)
    result = await workflow.run(seed=str(seed), until="review")
    project_id = result["project_id"]

    recorded = repository.row("SELECT state_db_path FROM projects WHERE id=?", (project_id,))
    assert recorded and Path(recorded["state_db_path"]).resolve() == (dir_a / "state.sqlite3").resolve(), \
        "the project must remember the directory its state lives in"

    # Resume from a different directory: the default settings would point at
    # dir_b's fresh database, but the project anchor must override that.
    dir_b = tmp_path / "project_b"
    dir_b.mkdir()
    monkeypatch.setenv("AUTORESEARCH_DATA_ROOT", str(dir_b))
    from autoresearch.application.config import resolve_settings

    settings = resolve_settings(project_id)
    assert settings.database_path.parent.resolve() == dir_a.resolve(), \
        "resume must anchor to the project's own directory, not the current one"


@pytest.mark.anyio
async def test_two_projects_in_two_directories_stay_isolated(tmp_path, monkeypatch, registry):
    dir_a, dir_b = tmp_path / "a", tmp_path / "b"
    dir_a.mkdir(); dir_b.mkdir()
    repo_a, workflow_a, seed_a = smoke._build_loop(dir_a)
    repo_b, workflow_b, seed_b = smoke._build_loop(dir_b)

    result_a = await workflow_a.run(seed=str(seed_a), until="review")
    result_b = await workflow_b.run(seed=str(seed_b), until="review")

    # Each directory's database contains exactly its own project.
    assert repo_a.row("SELECT COUNT(*) AS n FROM projects WHERE id=?", (result_b["project_id"],))["n"] == 0
    assert repo_b.row("SELECT COUNT(*) AS n FROM projects WHERE id=?", (result_a["project_id"],))["n"] == 0

    # Resume of A from anywhere keeps using A's gap (not B's).
    monkeypatch.setenv("AUTORESEARCH_DATA_ROOT", str(dir_b))
    from autoresearch.application.config import resolve_settings
    from autoresearch.storage.repository import ResearchRepository

    anchored = resolve_settings(result_a["project_id"])
    anchored_repo = ResearchRepository(anchored.database_path, anchored.artifact_root)
    gap_a = anchored_repo.row("SELECT id FROM research_gaps WHERE project_id=?", (result_a["project_id"],))
    assert gap_a and gap_a["id"] == repo_a.latest_gap(result_a["project_id"]).id
