"""Durable SQLite-backed state store for agent sessions and turns."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from autoresearch.domain.models import utc_now
from autoresearch.providers.redaction import redact
from autoresearch.storage.repository import ResearchRepository

from .models import AgentIntent, AgentSession, AgentTurn, RoutingDecision, SessionMode, TurnStatus


class AgentStore:
    def __init__(self, repository: ResearchRepository) -> None:
        self.repository = repository

    def create_session(self, project_id: str, repository: str, working_directory: str, *, token_budget: int = 32_000, cost_budget: float | None = None) -> AgentSession:
        now = utc_now()
        session = AgentSession(uuid4().hex, project_id, repository, working_directory, token_budget=token_budget, cost_budget=cost_budget, created_at=now, updated_at=now)
        self.repository.execute(
            "INSERT INTO agent_sessions(id,project_id,active_repository,working_directory,selected_mode,auto_routing,provider_role,token_budget,cost_budget,active_run_id,cancellation_requested,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (session.id, session.project_id, session.active_repository, session.working_directory, session.selected_mode.value, 0, session.provider_role, session.token_budget, session.cost_budget, None, 0, now, now),
        )
        return session

    def session(self, session_id: str) -> AgentSession | None:
        row = self.repository.row("SELECT * FROM agent_sessions WHERE id=?", (session_id,))
        return _session(row) if row else None

    def list_sessions(self, *, project_id: str | None = None, repository: str | None = None) -> list[AgentSession]:
        clauses: list[str] = []
        params: list[Any] = []
        if project_id:
            clauses.append("project_id=?")
            params.append(project_id)
        if repository:
            clauses.append("active_repository=?")
            params.append(repository)
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        return [_session(row) for row in self.repository.rows(f"SELECT * FROM agent_sessions{where} ORDER BY updated_at DESC", tuple(params))]

    def update_session(self, session_id: str, *, selected_mode: SessionMode | None = None, auto_routing: bool | None = None, active_run_id: str | None | object = ..., cancellation_requested: bool | None = None) -> AgentSession:
        session = self.require_session(session_id)
        mode = selected_mode or session.selected_mode
        auto = session.auto_routing if auto_routing is None else auto_routing
        run_id = session.active_run_id if active_run_id is ... else active_run_id
        cancelled = session.cancellation_requested if cancellation_requested is None else cancellation_requested
        self.repository.execute(
            "UPDATE agent_sessions SET selected_mode=?,auto_routing=?,active_run_id=?,cancellation_requested=?,updated_at=? WHERE id=?",
            (mode.value, int(auto), run_id, int(cancelled), utc_now(), session_id),
        )
        return self.require_session(session_id)

    def require_session(self, session_id: str) -> AgentSession:
        value = self.session(session_id)
        if value is None:
            raise ValueError(f"Agent session '{session_id}' was not found.")
        return value

    def append_message(self, session_id: str, role: str, content: str, *, turn_id: str | None = None, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        message = {"id": uuid4().hex, "session_id": session_id, "turn_id": turn_id, "role": role, "content": content, "metadata": metadata or {}, "created_at": utc_now()}
        self.repository.execute(
            "INSERT INTO agent_messages(id,session_id,turn_id,role,content,metadata_json,created_at) VALUES(?,?,?,?,?,?,?)",
            (message["id"], session_id, turn_id, role, redact(str(content)), _json(redact(message["metadata"])), message["created_at"]),
        )
        self.repository.execute("UPDATE agent_sessions SET updated_at=? WHERE id=?", (utc_now(), session_id))
        return message

    def messages(self, session_id: str, *, limit: int = 100) -> list[dict[str, Any]]:
        rows = self.repository.rows("SELECT * FROM agent_messages WHERE session_id=? ORDER BY created_at DESC,id DESC LIMIT ?", (session_id, limit))
        return [{**row, "metadata": _loads(row.pop("metadata_json"), {})} for row in reversed(rows)]

    def create_turn(self, session_id: str, message: str, *, override_mode: str | None, max_tool_iterations: int, max_model_calls: int, time_budget_seconds: float, token_budget: int, cost_budget: float | None) -> AgentTurn:
        turn_id = uuid4().hex
        input_message = self.append_message(session_id, "user", message, turn_id=turn_id)
        turn = AgentTurn(turn_id, session_id, message, TurnStatus.PENDING, override_mode, None, max_tool_iterations, max_model_calls, time_budget_seconds, token_budget, cost_budget)
        self.repository.execute(
            "INSERT INTO agent_turns(id,session_id,input_message_id,status,override_mode,routing_json,max_tool_iterations,max_model_calls,time_budget_seconds,token_budget,cost_budget,model_calls,tool_iterations,input_tokens,output_tokens,estimated_cost,error,created_at,completed_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (turn.id, turn.session_id, input_message["id"], turn.status.value, turn.override_mode, None, turn.max_tool_iterations, turn.max_model_calls, turn.time_budget_seconds, turn.token_budget, turn.cost_budget, 0, 0, 0, 0, 0.0, None, turn.created_at, None),
        )
        return turn

    def turn(self, turn_id: str) -> AgentTurn | None:
        row = self.repository.row("SELECT t.*,m.content AS message FROM agent_turns t JOIN agent_messages m ON m.id=t.input_message_id WHERE t.id=?", (turn_id,))
        return _turn(row) if row else None

    def require_turn(self, turn_id: str) -> AgentTurn:
        value = self.turn(turn_id)
        if value is None:
            raise ValueError(f"Agent turn '{turn_id}' was not found.")
        return value

    def update_turn(self, turn_id: str, *, status: TurnStatus | None = None, routing: RoutingDecision | None | object = ..., model_calls: int | None = None, tool_iterations: int | None = None, input_tokens: int | None = None, output_tokens: int | None = None, estimated_cost: float | None = None, error: str | None | object = ..., completed: bool = False) -> AgentTurn:
        turn = self.require_turn(turn_id)
        route = turn.routing if routing is ... else routing
        final_error = turn.error if error is ... else error
        self.repository.execute(
            "UPDATE agent_turns SET status=?,routing_json=?,model_calls=?,tool_iterations=?,input_tokens=?,output_tokens=?,estimated_cost=?,error=?,completed_at=? WHERE id=?",
            ((status or turn.status).value, _json(asdict(route)) if route else None, model_calls if model_calls is not None else turn.model_calls, tool_iterations if tool_iterations is not None else turn.tool_iterations, input_tokens if input_tokens is not None else turn.input_tokens, output_tokens if output_tokens is not None else turn.output_tokens, estimated_cost if estimated_cost is not None else turn.estimated_cost, final_error, utc_now() if completed else turn.completed_at, turn_id),
        )
        return self.require_turn(turn_id)

    def append_event(self, session_id: str, event_type: str, payload: dict[str, Any]) -> int:
        with self.repository.connect() as connection:
            cursor = connection.execute("INSERT INTO agent_events(session_id,event_type,payload_json,created_at) VALUES(?,?,?,?)", (session_id, event_type, _json(redact(payload)), utc_now()))
            return int(cursor.lastrowid)

    def events(self, session_id: str, *, after_id: int = 0, limit: int = 200) -> list[dict[str, Any]]:
        rows = self.repository.rows("SELECT * FROM agent_events WHERE session_id=? AND id>? ORDER BY id LIMIT ?", (session_id, after_id, limit))
        return [{"id": row["id"], "session_id": session_id, "event_type": row["event_type"], "payload": _loads(row["payload_json"], {}), "created_at": row["created_at"]} for row in rows]

    def create_tool_call(self, session_id: str, turn_id: str, tool_id: str, arguments: dict[str, Any]) -> dict[str, Any]:
        value = {"id": uuid4().hex, "session_id": session_id, "turn_id": turn_id, "tool_id": tool_id, "arguments": arguments, "status": "proposed", "created_at": utc_now()}
        self.repository.execute("INSERT INTO agent_tool_calls(id,session_id,turn_id,tool_id,arguments_json,status,created_at) VALUES(?,?,?,?,?,?,?)", (value["id"], session_id, turn_id, tool_id, _json(redact(arguments)), value["status"], value["created_at"]))
        return value

    def update_tool_call(self, call_id: str, *, status: str, permission_id: str | None = None, observation: dict[str, Any] | None = None, error: str | None = None) -> dict[str, Any]:
        self.repository.execute("UPDATE agent_tool_calls SET status=?,permission_id=COALESCE(?,permission_id),observation_json=?,error=?,completed_at=? WHERE id=?", (status, permission_id, _json(observation) if observation is not None else None, error, utc_now() if status in {"completed", "failed", "denied", "cancelled"} else None, call_id))
        return self.tool_call(call_id)

    def tool_call(self, call_id: str) -> dict[str, Any]:
        row = self.repository.row("SELECT * FROM agent_tool_calls WHERE id=?", (call_id,))
        if not row:
            raise ValueError(f"Tool call '{call_id}' was not found.")
        row["arguments"] = _loads(row.pop("arguments_json"), {})
        row["observation"] = _loads(row.pop("observation_json"), None)
        return row

    def pending_tool_calls(self, turn_id: str) -> list[dict[str, Any]]:
        return [self.tool_call(row["id"]) for row in self.repository.rows("SELECT id FROM agent_tool_calls WHERE turn_id=? AND status='awaiting_permission' ORDER BY created_at", (turn_id,))]

    def create_permission(self, session_id: str, turn_id: str, tool_id: str, scope: str, repository: str, action: str, request: dict[str, Any]) -> dict[str, Any]:
        now = utc_now()
        value = {"id": uuid4().hex, "session_id": session_id, "turn_id": turn_id, "tool_id": tool_id, "scope": scope, "repository": repository, "action": action, "decision": "pending", "request": request, "expires_at": None, "created_at": now, "updated_at": now}
        self.repository.execute("INSERT INTO agent_permissions(id,session_id,turn_id,tool_id,scope,repository,action,decision,request_json,expires_at,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)", (value["id"], session_id, turn_id, tool_id, scope, repository, action, value["decision"], _json(redact(request)), None, now, now))
        return value

    def permission(self, permission_id: str) -> dict[str, Any] | None:
        row = self.repository.row("SELECT * FROM agent_permissions WHERE id=?", (permission_id,))
        if row:
            row["request"] = _loads(row.pop("request_json"), {})
        return row

    def decide_permission(self, permission_id: str, *, approved: bool, expires_at: str | None = None) -> dict[str, Any]:
        permission = self.permission(permission_id)
        if not permission:
            raise ValueError(f"Permission request '{permission_id}' was not found.")
        self.repository.execute("UPDATE agent_permissions SET decision=?,expires_at=?,updated_at=? WHERE id=?", ("approved" if approved else "denied", expires_at, utc_now(), permission_id))
        return self.permission(permission_id) or permission

    def permission_allows(self, session_id: str, tool_id: str, scope: str, repository: str) -> bool:
        rows = self.repository.rows("SELECT * FROM agent_permissions WHERE session_id=? AND tool_id=? AND scope=? AND repository=? AND decision='approved' ORDER BY updated_at DESC", (session_id, tool_id, scope, repository))
        now = datetime.now(timezone.utc)
        return any(not row["expires_at"] or datetime.fromisoformat(row["expires_at"]) >= now for row in rows)

    def link_artifact(self, session_id: str, artifact_id: str, turn_id: str) -> None:
        self.repository.execute("INSERT OR IGNORE INTO agent_artifact_refs(session_id,artifact_id,turn_id,created_at) VALUES(?,?,?,?)", (session_id, artifact_id, turn_id, utc_now()))

    def artifacts(self, session_id: str) -> list[dict[str, Any]]:
        return self.repository.rows("SELECT a.*,r.turn_id AS referenced_by_turn FROM agent_artifact_refs r JOIN artifacts a ON a.id=r.artifact_id WHERE r.session_id=? ORDER BY r.created_at", (session_id,))

    def create_suggestion(self, session_id: str, turn_id: str, capability_id: str, message: str, routing: RoutingDecision) -> dict[str, Any]:
        value = {"id": uuid4().hex, "session_id": session_id, "turn_id": turn_id, "capability_id": capability_id, "message": message, "routing": asdict(routing), "status": "pending", "created_at": utc_now()}
        self.repository.execute("INSERT INTO agent_capability_suggestions(id,session_id,turn_id,capability_id,message,routing_json,status,created_at) VALUES(?,?,?,?,?,?,?,?)", (value["id"], session_id, turn_id, capability_id, message, _json(value["routing"]), value["status"], value["created_at"]))
        return value

    def suggestion(self, suggestion_id: str) -> dict[str, Any] | None:
        row = self.repository.row("SELECT * FROM agent_capability_suggestions WHERE id=?", (suggestion_id,))
        if row:
            row["routing"] = _loads(row.pop("routing_json"), {})
        return row

    def decide_suggestion(self, suggestion_id: str, approved: bool) -> dict[str, Any]:
        suggestion = self.suggestion(suggestion_id)
        if not suggestion:
            raise ValueError(f"Capability suggestion '{suggestion_id}' was not found.")
        if suggestion["status"] != "pending":
            return suggestion
        self.repository.execute("UPDATE agent_capability_suggestions SET status=?,decided_at=? WHERE id=?", ("approved" if approved else "declined", utc_now(), suggestion_id))
        return self.suggestion(suggestion_id) or suggestion

    def idempotency(self, key: str) -> dict[str, Any] | None:
        row = self.repository.row("SELECT * FROM agent_idempotency WHERE key=?", (key,))
        if row:
            row["response"] = _loads(row.pop("response_json"), {})
        return row

    def save_idempotency(self, key: str, operation: str, session_id: str, turn_id: str | None, response: dict[str, Any]) -> None:
        self.repository.execute("INSERT INTO agent_idempotency(key,operation,session_id,turn_id,response_json,created_at) VALUES(?,?,?,?,?,?)", (key, operation, session_id, turn_id, _json(response), utc_now()))


def _session(row: dict[str, Any]) -> AgentSession:
    return AgentSession(row["id"], row["project_id"], row["active_repository"], row["working_directory"], SessionMode(row["selected_mode"]), bool(row["auto_routing"]), row["provider_role"], int(row["token_budget"]), float(row["cost_budget"]) if row["cost_budget"] is not None else None, row["active_run_id"], bool(row["cancellation_requested"]), row["created_at"], row["updated_at"])


def _turn(row: dict[str, Any]) -> AgentTurn:
    raw_route = _loads(row["routing_json"], None)
    route = RoutingDecision(AgentIntent(raw_route["intent"]), raw_route.get("recommended_capability"), float(raw_route["confidence"]), raw_route["reason"], tuple(raw_route.get("required_inputs", ())), bool(raw_route.get("confirmation_required")), bool(raw_route.get("explicit"))) if raw_route else None
    return AgentTurn(row["id"], row["session_id"], row["message"], TurnStatus(row["status"]), row["override_mode"], route, int(row["max_tool_iterations"]), int(row["max_model_calls"]), float(row["time_budget_seconds"]), int(row["token_budget"]), float(row["cost_budget"]) if row["cost_budget"] is not None else None, int(row["model_calls"]), int(row["tool_iterations"]), int(row["input_tokens"]), int(row["output_tokens"]), float(row["estimated_cost"]), row["error"], row["created_at"], row["completed_at"])


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True, separators=(",", ":"))


def _loads(value: str | None, default: Any) -> Any:
    if value is None:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default
