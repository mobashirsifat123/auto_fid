from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from autoresearch.api.app import create_app
from autoresearch.application.config import Settings
from autoresearch.domain.models import PipelineKind


def test_api_contract_capabilities_and_idempotent_mutation(tmp_path: Path):
    config = Path(__file__).parents[1] / "docs" / "live_config.example.toml"
    settings = Settings(
        data_root=tmp_path,
        database_path=tmp_path / "api.sqlite3",
        artifact_root=tmp_path / "artifacts",
        download_root=tmp_path / "downloads",
        provider_config=config,
        search_sources=("crossref",),
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/health").json()["status"] == "ready"
        capabilities = client.get("/v1/capabilities").json()["pipelines"]
        assert {item["key"] for item in capabilities if item["enabled"]} == {"study_search", "deep_literature_review", "novelty_generation", "experiment_design_execution", "result_conditioned_reinvention"}
        assert all(item["reason"] for item in capabilities if not item["enabled"])

        project = client.post("/v1/projects", json={"title": "API contract"}).json()
        project_id = project["id"]
        same_project = client.post("/v1/projects", headers={"Idempotency-Key": "project-contract-key"}, json={"title": "same project"}).json()
        same_project_retry = client.post("/v1/projects", headers={"Idempotency-Key": "project-contract-key"}, json={"title": "different title on retry"}).json()
        assert same_project["id"] == same_project_retry["id"]
        async def fake_discovery(project_id: str, run_id: str, parsed, *, limit: int, since_year: int | None, max_runtime_seconds: float | None):
            repository = client.app.state.container.repository
            repository.begin_checkpoint(run_id, "fake")
            repository.complete_checkpoint(run_id, "fake", payload={"records": limit})
            repository.finish_run(run_id)

        client.app.state.container.discovery._execute = fake_discovery
        headers = {"Idempotency-Key": "discover-contract-key"}
        first = client.post(
            f"/v1/projects/{project_id}/discover",
            headers=headers,
            json={"seed_paper": "10.48550/arXiv.2005.11401", "limit": 20},
        )
        second = client.post(
            f"/v1/projects/{project_id}/discover",
            headers=headers,
            json={"seed_paper": "10.48550/arXiv.2005.11401", "limit": 20},
        )
        assert first.status_code == 202
        assert first.json()["run_id"] == second.json()["run_id"]
        run_id = first.json()["run_id"]
        assert client.get(f"/v1/runs/{run_id}").status_code == 200
        events = client.get(f"/v1/runs/{run_id}/events").text
        assert "event: run.created" in events
        assert "event: run.progress" in events
        assert "event: run.completed" in events
        reinvent = client.post(f"/v1/projects/{project_id}/reinvent", json={})
        assert reinvent.status_code == 409
        assert reinvent.json()["detail"]["code"] == "capability_unavailable"
        assert "not implemented" in reinvent.json()["detail"]["message"]


def test_api_resumes_discovery_with_service_compatible_input(tmp_path: Path):
    config = Path(__file__).parents[1] / "docs" / "live_config.example.toml"
    settings = Settings(tmp_path, tmp_path / "api.sqlite3", tmp_path / "artifacts", tmp_path / "downloads", config, ("crossref",))
    with TestClient(create_app(settings)) as client:
        repository = client.app.state.container.repository
        project = repository.create_project("Resume contract")
        run = repository.create_run(
            project.id,
            PipelineKind.DISCOVERY,
            {"seed_input": "10.48550/arXiv.2005.11401", "limit": 20, "since_year": None, "max_cost_usd": None, "max_runtime_seconds": None},
        )
        repository.finish_run(run.id, error="simulated interruption")

        async def fake_resume(run_id: str):
            assert repository.get_run(run_id).input_data["seed_input"] == "10.48550/arXiv.2005.11401"
            repository.finish_run(run_id)

        client.app.state.container.discovery.resume = fake_resume
        response = client.post(f"/v1/runs/{run.id}/resume", headers={"Idempotency-Key": "resume-contract-key"})
        retry = client.post(f"/v1/runs/{run.id}/resume", headers={"Idempotency-Key": "resume-contract-key"})
        assert response.status_code == 202
        assert response.json()["run_id"] == retry.json()["run_id"] == run.id
        assert "event: run.resumed" in client.get(f"/v1/runs/{run.id}/events").text


def test_api_authentication_and_secret_free_provider_status(tmp_path: Path, monkeypatch):
    config = Path(__file__).parents[1] / "docs" / "live_config.example.toml"
    settings = Settings(tmp_path, tmp_path / "api.sqlite3", tmp_path / "artifacts", tmp_path / "downloads", config, ("crossref",))
    monkeypatch.setenv("AUTORESEARCH_API_AUTH_TOKEN_ENV", "TEST_AUTORESEARCH_TOKEN")
    monkeypatch.setenv("TEST_AUTORESEARCH_TOKEN", "secret-token")
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/health").status_code == 401
        response = client.get("/v1/providers/status", headers={"Authorization": "Bearer secret-token"})
        assert response.status_code == 200
        body = response.json()
        assert all("api_key" not in str(item).lower() for item in body["providers"])
        assert all("secret-token" not in str(item) for item in body["providers"])


def test_api_keeps_empty_provider_installation_usable(tmp_path: Path):
    settings = Settings(
        data_root=tmp_path,
        database_path=tmp_path / "api.sqlite3",
        artifact_root=tmp_path / "artifacts",
        download_root=tmp_path / "downloads",
        provider_config=tmp_path / "missing-provider-config.toml",
        search_sources=("crossref",),
        agent_roots=(tmp_path,),
    )
    with TestClient(create_app(settings)) as client:
        assert client.get("/v1/health").json()["status"] == "ready"
        assert client.get("/v1/providers/status").json() == {"providers": [], "roles": []}
        project = client.post("/v1/projects", json={"title": "No provider yet"}).json()
        session = client.post("/v1/agent/sessions", json={"project_id": project["id"], "active_repository": str(tmp_path), "working_directory": str(tmp_path)}).json()
        assert session["session"]["id"]
        turn = client.post(f"/v1/agent/sessions/{session['session']['id']}/turns", json={"message": "Explain this repository."})
        assert turn.status_code == 202
        import time
        for _ in range(20):
            status = client.get(f"/v1/agent/sessions/{session['session']['id']}").json()["turns"][-1]["status"]
            if status in {"completed", "failed"}:
                break
            time.sleep(0.02)
        payload = client.get(f"/v1/agent/sessions/{session['session']['id']}").json()
        assert payload["turns"][-1]["status"] == "completed"
        assert "No chat" in payload["messages"][-1]["content"]


def test_run_sse_rejects_malformed_last_event_id(tmp_path: Path):
    config = Path(__file__).parents[1] / "docs" / "live_config.example.toml"
    settings = Settings(tmp_path, tmp_path / "api.sqlite3", tmp_path / "artifacts", tmp_path / "downloads", config, ("crossref",))
    with TestClient(create_app(settings)) as client:
        repository = client.app.state.container.repository
        project = repository.create_project("SSE cursor")
        run = repository.create_run(project.id, PipelineKind.REVIEW, {"top_k": 5})
        response = client.get(f"/v1/runs/{run.id}/events", headers={"Last-Event-ID": "not-a-cursor"})
        assert response.status_code == 400
        assert response.json()["detail"]["code"] == "invalid_event_cursor"


def test_run_sse_drains_large_completed_event_history(tmp_path: Path):
    config = Path(__file__).parents[1] / "docs" / "live_config.example.toml"
    settings = Settings(tmp_path, tmp_path / "api.sqlite3", tmp_path / "artifacts", tmp_path / "downloads", config, ("crossref",))
    with TestClient(create_app(settings)) as client:
        repository = client.app.state.container.repository
        project = repository.create_project("Large event history")
        run = repository.create_run(project.id, PipelineKind.REVIEW, {"top_k": 15})
        for index in range(450):
            repository.append_api_event(run.id, "run.progress", {"index": index})
        repository.finish_run(run.id)
        response = client.get(f"/v1/runs/{run.id}/events")
        assert response.status_code == 200
        assert response.text.count("event: run.progress") == 450


def test_cancel_and_approval_mutations_are_idempotent(tmp_path: Path):
    config = Path(__file__).parents[1] / "docs" / "live_config.example.toml"
    settings = Settings(tmp_path, tmp_path / "api.sqlite3", tmp_path / "artifacts", tmp_path / "downloads", config, ("crossref",))
    with TestClient(create_app(settings)) as client:
        repository = client.app.state.container.repository
        project = repository.create_project("Mutation contract")
        run = repository.create_run(project.id, PipelineKind.REVIEW, {"top_k": 5})
        headers = {"Idempotency-Key": "cancel-once"}
        first = client.post(f"/v1/runs/{run.id}/cancel", headers=headers)
        second = client.post(f"/v1/runs/{run.id}/cancel", headers=headers)
        assert first.json() == second.json() == {"run_id": run.id, "status": "cancelled"}
        assert len([item for item in repository.api_events(run.id) if item["event_type"] == "run.cancelled"]) == 1

        approval_headers = {"Idempotency-Key": "approval-once"}
        approved = client.post(f"/v1/runs/{run.id}/approve", headers=approval_headers, json={"approved": True, "reason": "reviewed"})
        approved_retry = client.post(f"/v1/runs/{run.id}/approve", headers=approval_headers, json={"approved": False, "reason": "changed retry"})
        assert approved.json() == approved_retry.json()
        assert repository.approval(run.id)["status"] == "approved"
        assert len([item for item in repository.api_events(run.id) if item["event_type"] == "run.approval"]) == 1
