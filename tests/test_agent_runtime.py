from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from autoresearch.agent.runtime import AgentRuntime
from autoresearch.application.config import Settings
from autoresearch.api.app import create_app
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.llm.base import CompletionResult, ToolCallResult
from autoresearch.storage.repository import ResearchRepository


class DeterministicRouter:
    assignments = {ModelRole.RESEARCH_REASONING: object()}

    def __init__(self, calls: list[ToolCallResult]) -> None:
        self.calls = calls

    async def tool_call(self, *_args, **_kwargs):
        return self.calls.pop(0)

    async def complete(self, *_args, **_kwargs):
        return CompletionResult("A concise backend-backed answer.", "fake-model", 3, 5, "fake-request", 0.0, "USD", {})


def test_default_agent_turn_stays_conversational(tmp_path: Path):
    repository = ResearchRepository(tmp_path / "agent.sqlite3", tmp_path / "artifacts")
    project = repository.create_project("Agent fixture")
    router = DeterministicRouter([ToolCallResult((), CompletionResult("The repository is a Python service.", "fake-model", 4, 7, "fake-request", 0.0, "USD", {}))])
    container = SimpleNamespace(repository=repository, router=router, worker=None)
    runtime = AgentRuntime(container, allowed_roots=(tmp_path,))
    session = runtime.create_session(project.id, active_repository=str(tmp_path))
    turn = runtime.start_turn(session.id, "Explain this repository.")
    asyncio.run(runtime.execute_turn(turn.id))
    payload = runtime.session_payload(session.id)
    assert payload["turns"][0]["status"] == "completed"
    assert payload["messages"][-1]["content"] == "The repository is a Python service."
    assert repository.list_runs(project_id=project.id) == []
    assert {event["event_type"] for event in runtime.store.events(session.id)} >= {"agent.routing_decided", "agent.response_delta", "agent.turn_completed"}


def test_mutating_generic_tool_requires_scoped_permission_and_resumes(tmp_path: Path):
    repository = ResearchRepository(tmp_path / "agent.sqlite3", tmp_path / "artifacts")
    project = repository.create_project("Permission fixture")
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "write_file", "arguments": '{"path":"answer.txt","content":"safe"}'}}
    router = DeterministicRouter([
        ToolCallResult((tool_call,), CompletionResult("", "fake-model", 3, 1, "fake-request", 0.0, "USD", {})),
        ToolCallResult((), CompletionResult("The file is updated.", "fake-model", 3, 4, "fake-request", 0.0, "USD", {})),
    ])
    runtime = AgentRuntime(SimpleNamespace(repository=repository, router=router, worker=None), allowed_roots=(tmp_path,))
    session = runtime.create_session(project.id, active_repository=str(tmp_path))
    turn = runtime.start_turn(session.id, "Write the answer file.")
    asyncio.run(runtime.execute_turn(turn.id))
    pending = runtime.store.pending_tool_calls(turn.id)
    assert pending and pending[0]["status"] == "awaiting_permission"
    permission_id = pending[0]["permission_id"]
    asyncio.run(runtime.decide_permission(permission_id, approved=True))
    asyncio.run(runtime.continue_after_permission(turn.id))
    assert (tmp_path / "answer.txt").read_text() == "safe"
    assert runtime.store.require_turn(turn.id).status.value == "completed"


def test_cancelled_session_cannot_execute_late_permission(tmp_path: Path):
    repository = ResearchRepository(tmp_path / "agent.sqlite3", tmp_path / "artifacts")
    project = repository.create_project("Cancellation fixture")
    tool_call = {"id": "call-1", "type": "function", "function": {"name": "write_file", "arguments": '{"path":"answer.txt","content":"must not write"}'}}
    router = DeterministicRouter([ToolCallResult((tool_call,), CompletionResult("", "fake-model", 3, 1, "fake-request", 0.0, "USD", {}))])
    runtime = AgentRuntime(SimpleNamespace(repository=repository, router=router, worker=None), allowed_roots=(tmp_path,))
    session = runtime.create_session(project.id, active_repository=str(tmp_path))
    turn = runtime.start_turn(session.id, "Write the answer file.")
    asyncio.run(runtime.execute_turn(turn.id))
    pending = runtime.store.pending_tool_calls(turn.id)[0]
    permission_id = pending["permission_id"]
    runtime.cancel(session.id)
    asyncio.run(runtime.decide_permission(permission_id, approved=True))
    assert not (tmp_path / "answer.txt").exists()
    assert runtime.store.require_turn(turn.id).status.value == "cancelled"
    assert runtime.store.tool_call(pending["id"])["status"] == "cancelled"


def test_agent_api_session_mode_sse_and_reinvention_fallback(tmp_path: Path):
    config = Path(__file__).parents[1] / "docs" / "live_config.example.toml"
    settings = Settings(tmp_path, tmp_path / "api.sqlite3", tmp_path / "artifacts", tmp_path / "downloads", config, ("crossref",), (tmp_path,))
    with TestClient(create_app(settings)) as client:
        project = client.post("/v1/projects", json={"title": "Agent API"}).json()
        session = client.post("/v1/agent/sessions", json={"project_id": project["id"], "active_repository": str(tmp_path), "working_directory": str(tmp_path), "project_title": "Agent API", "token_budget": 32000, "cost_budget": None}).json()
        session_id = session["session"]["id"]
        assert session["session"]["selected_mode"] == "default"
        assert client.post(f"/v1/agent/sessions/{session_id}/mode", json={"mode": "default", "auto_routing": True}).status_code == 200
        turn = client.post(f"/v1/agent/sessions/{session_id}/turns", headers={"Idempotency-Key": "agent-turn-once"}, json={"message": "Explain this class", "time_budget_seconds": 30}).json()
        assert client.post(f"/v1/agent/sessions/{session_id}/turns", headers={"Idempotency-Key": "agent-turn-once"}, json={"message": "different retry", "time_budget_seconds": 30}).json() == turn
        for _ in range(20):
            if client.get(f"/v1/agent/sessions/{session_id}").json()["turns"][-1]["status"] in {"completed", "failed"}:
                break
            import time
            time.sleep(0.02)
        assert "agent.turn_started" in client.get(f"/v1/agent/sessions/{session_id}/events").text
        invalid_cursor = client.get(f"/v1/agent/sessions/{session_id}/events", headers={"Last-Event-ID": "not-a-cursor"})
        assert invalid_cursor.status_code == 400
        assert invalid_cursor.json()["detail"]["code"] == "invalid_event_cursor"
        response = client.post(f"/v1/agent/sessions/{session_id}/turns", json={"message": "Reinvent a replacement after the failed experiment", "time_budget_seconds": 30})
        assert response.status_code == 202
        for _ in range(20):
            if client.get(f"/v1/agent/sessions/{session_id}").json()["turns"][-1]["status"] in {"completed", "failed"}:
                break
            import time
            time.sleep(0.05)
        turn_data = client.get(f"/v1/agent/sessions/{session_id}").json()["turns"][-1]
        assert turn_data["status"] in {"awaiting_confirmation", "completed", "failed"}
