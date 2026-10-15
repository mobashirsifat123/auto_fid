"""Thin HTTP/SSE boundary for the durable AutoResearch application services."""

from __future__ import annotations

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any, AsyncIterator, Literal
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, HTTPException, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from autoresearch.application.factory import ApplicationContainer
from autoresearch.application.services import DiscoveryOutcome, ReviewOutcome
from autoresearch.agent.models import SessionMode, TurnStatus
from autoresearch.domain.models import PipelineKind, RunStatus, dataclass_payload
from autoresearch.providers.errors import ProviderError
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.config import ModelAssignment
from autoresearch.providers.redaction import redact_text
from autoresearch.versioning import API_VERSION, DATABASE_SCHEMA_VERSION, PRODUCT_VERSION, WORKER_PROTOCOL_VERSION

from . import __version__
from ..application.config import Settings, load_settings


class DiscoverRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    seed_paper: str = Field(min_length=1)
    limit: int = Field(default=20, ge=1, le=100)
    since_year: int | None = Field(default=None, ge=1900, le=2100)
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_runtime_seconds: float | None = Field(default=None, gt=0)


class ReviewRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    top_k: int = Field(default=5, ge=1, le=15)
    hierarchical: bool = True
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_runtime_seconds: float | None = Field(default=None, gt=0)


class NoveltyRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gap_id: str | None = Field(default=None)
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_runtime_seconds: float | None = Field(default=None, gt=0)


class ExperimentRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1)
    mode: Literal["plan", "validate", "execute", "inspect", "report"] = "plan"
    max_cost_usd: float | None = Field(default=None, ge=0)
    max_runtime_seconds: float | None = Field(default=None, gt=0)


class ProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1, max_length=200)


class ProviderSelectionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: ModelRole
    provider: str = Field(min_length=1, max_length=100)
    model: str = Field(min_length=1, max_length=300)


class LinkProjectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str = Field(min_length=1)


class ApprovalRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool
    reason: str | None = Field(default=None, max_length=1000)
    specification_checksum: str | None = Field(default=None, min_length=64, max_length=64)
    approver: str = Field(default="local-user", min_length=1, max_length=200)
    approved_scope: str = Field(default="local-experiment", max_length=200)
    approved_target: str = Field(default="local", max_length=100)
    approved_budget: float = Field(default=3600.0, ge=0, le=86400)
    expires_at: str | None = None


class AgentSessionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project_id: str | None = None
    project_title: str = Field(default="AutoResearch workspace", min_length=1, max_length=200)
    active_repository: str | None = None
    working_directory: str | None = None
    token_budget: int = Field(default=32_000, ge=256, le=200_000)
    cost_budget: float | None = Field(default=None, ge=0)


class AgentTurnRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=100_000)
    capability_override: str | None = None
    token_budget: int | None = Field(default=None, ge=256, le=200_000)
    cost_budget: float | None = Field(default=None, ge=0)
    time_budget_seconds: float = Field(default=120.0, gt=0, le=600)


class AgentModeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str | None = None
    auto_routing: bool | None = None


class AgentDecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    approved: bool
    expires_at: str | None = None


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved = settings or load_settings()
    tasks: dict[str, asyncio.Task[Any]] = {}
    monitors: dict[str, asyncio.Task[Any]] = {}
    agent_tasks: dict[str, asyncio.Task[Any]] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            app.state.container = ApplicationContainer(resolved)
            app.state.startup_error = None
        except Exception as exc:
            app.state.container = None
            app.state.startup_error = redact_text(f"{type(exc).__name__}: {exc}")
        yield
        for task in [*tasks.values(), *monitors.values(), *agent_tasks.values()]:
            task.cancel()
        if tasks or monitors or agent_tasks:
            await asyncio.gather(*[*tasks.values(), *monitors.values(), *agent_tasks.values()], return_exceptions=True)
        if app.state.container:
            await app.state.container.close()

    app = FastAPI(title="AutoResearch API", version=API_VERSION, lifespan=lifespan)

    @app.exception_handler(ValueError)
    async def value_error(_request: Request, exc: ValueError):
        return JSONResponse(status_code=400, content={"error": {"code": "invalid_request", "message": redact_text(str(exc))}})

    @app.exception_handler(ProviderError)
    async def provider_error(_request: Request, exc: ProviderError):
        return JSONResponse(status_code=502, content={"error": {"code": exc.kind.value, "message": redact_text(str(exc)), "provider": exc.provider or None}})

    def require_auth(authorization: str | None = Header(default=None), x_autoresearch_token: str | None = Header(default=None)) -> None:
        env_name = os.getenv("AUTORESEARCH_API_AUTH_TOKEN_ENV", "").strip()
        if not env_name:
            return
        expected = os.getenv(env_name, "")
        presented = x_autoresearch_token or (authorization[7:] if authorization and authorization.lower().startswith("bearer ") else "")
        if not expected or presented != expected:
            raise HTTPException(status_code=401, detail={"code": "unauthorized", "message": "Backend authentication failed."})

    def container() -> ApplicationContainer:
        value = getattr(app.state, "container", None)
        if value is None:
            reason = getattr(app.state, "startup_error", None) or "Backend is not ready."
            raise HTTPException(status_code=503, detail={"code": "backend_unavailable", "message": reason})
        return value

    def project_or_404(repo, project_id: str):
        project = repo.get_project(project_id)
        if not project:
            raise HTTPException(status_code=404, detail={"code": "project_not_found", "message": f"Project '{project_id}' was not found."})
        return project

    def run_payload(repo, run_id: str) -> dict[str, Any]:
        run = repo.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail={"code": "run_not_found", "message": f"Run '{run_id}' was not found."})
        return dataclass_payload(run)

    def emit(repo, run_id: str, event_type: str, payload: dict[str, Any]) -> int:
        return repo.append_api_event(run_id, event_type, payload)

    async def monitor_run(repo, run_id: str) -> None:
        last_signature = ""
        while True:
            run = repo.get_run(run_id)
            if not run:
                return
            checkpoints = repo.checkpoints(run_id)
            usage = repo.usage(run_id=run_id)
            artifacts = repo.artifacts_for_run(run_id)
            payload = {
                "run": dataclass_payload(run),
                "checkpoints": checkpoints,
                "usage": {"requests": len(usage), "estimated_cost": round(sum(float(item.get("estimated_cost") or 0) for item in usage), 12)},
                "artifacts": [dataclass_payload(item) for item in artifacts],
            }
            signature = json.dumps(payload, sort_keys=True, default=str)
            if signature != last_signature:
                emit(repo, run_id, "run.progress", payload)
                last_signature = signature
            if run.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                emit(repo, run_id, "run.completed" if run.status is RunStatus.COMPLETED else "run.terminal", {"run": dataclass_payload(run)})
                return
            await asyncio.sleep(0.4)

    async def execute_discovery(c: ApplicationContainer, run_id: str, request: DiscoverRequest, project_id: str) -> None:
        monitor = asyncio.create_task(monitor_run(c.repository, run_id))
        monitors[run_id] = monitor
        try:
            parsed = c.discovery.parser.parse_seed(request.seed_paper)
            await c.discovery._execute(project_id, run_id, parsed, limit=request.limit, since_year=request.since_year, max_runtime_seconds=request.max_runtime_seconds)
        except asyncio.CancelledError:
            c.repository.cancel_run(run_id)
            emit(c.repository, run_id, "run.cancelled", {"run_id": run_id})
            raise
        except Exception as exc:
            if c.repository.get_run(run_id) and c.repository.get_run(run_id).status is RunStatus.RUNNING:
                c.repository.finish_run(run_id, error=redact_text(f"{type(exc).__name__}: {exc}"))
            emit(c.repository, run_id, "run.error", {"message": redact_text(f"{type(exc).__name__}: {exc}")})
        finally:
            c.repository.set_project_budget(project_id, hard_stop=request.max_cost_usd) if request.max_cost_usd is not None else None
            if not monitor.done():
                await monitor
            monitors.pop(run_id, None)

    async def execute_review(c: ApplicationContainer, run_id: str, project_id: str, request: ReviewRequest) -> None:
        monitor = asyncio.create_task(monitor_run(c.repository, run_id))
        monitors[run_id] = monitor
        try:
            await c.review._execute(run_id, project_id, top_k=request.top_k, hierarchical=request.hierarchical, max_runtime_seconds=request.max_runtime_seconds)
        except asyncio.CancelledError:
            c.repository.cancel_run(run_id)
            emit(c.repository, run_id, "run.cancelled", {"run_id": run_id})
            raise
        except Exception as exc:
            emit(c.repository, run_id, "run.error", {"message": redact_text(f"{type(exc).__name__}: {exc}")})
        finally:
            if not monitor.done():
                await monitor
            monitors.pop(run_id, None)

    async def execute_invent(c: ApplicationContainer, run_id: str, project_id: str, request: NoveltyRequest) -> None:
        monitor = asyncio.create_task(monitor_run(c.repository, run_id))
        monitors[run_id] = monitor
        try:
            await c.novelty.generate_portfolio(project_id, gap_id=request.gap_id, run_id=run_id)
            c.repository.finish_run(run_id)
        except asyncio.CancelledError:
            c.repository.cancel_run(run_id)
            emit(c.repository, run_id, "run.cancelled", {"run_id": run_id})
            raise
        except Exception as exc:
            c.repository.finish_run(run_id, error=redact_text(f"{type(exc).__name__}: {exc}"))
            emit(c.repository, run_id, "run.error", {"message": redact_text(f"{type(exc).__name__}: {exc}")})
        finally:
            if not monitor.done():
                await monitor
            monitors.pop(run_id, None)

    async def execute_experiment(c: ApplicationContainer, run_id: str, project_id: str, request: ExperimentRequest) -> None:
        monitor = asyncio.create_task(monitor_run(c.repository, run_id))
        monitors[run_id] = monitor
        try:
            await c.experiment.execute_candidate(request.candidate_id, run_id=run_id, mode=request.mode)
            c.repository.finish_run(run_id)
        except asyncio.CancelledError:
            c.repository.cancel_run(run_id)
            emit(c.repository, run_id, "run.cancelled", {"run_id": run_id})
            raise
        except Exception as exc:
            c.repository.finish_run(run_id, error=redact_text(f"{type(exc).__name__}: {exc}"))
            emit(c.repository, run_id, "run.error", {"message": redact_text(f"{type(exc).__name__}: {exc}")})
        finally:
            if not monitor.done():
                await monitor
            monitors.pop(run_id, None)

    def mutation_key(header: str | None) -> str:
        return (header or uuid4().hex).strip()

    @app.get("/v1/health", dependencies=[Depends(require_auth)])
    async def health():
        c = getattr(app.state, "container", None)
        return {"status": "ready" if c else "unavailable", "service": "autoresearch-api", "version": API_VERSION, "api_version": API_VERSION, "product_version": PRODUCT_VERSION, "database_schema_version": DATABASE_SCHEMA_VERSION, "worker_protocol_version": WORKER_PROTOCOL_VERSION, "reason": getattr(app.state, "startup_error", None)}

    @app.get("/v1/capabilities", dependencies=[Depends(require_auth)])
    async def capabilities():
        c = container()
        pipelines = [
            {"key": PipelineKind.DISCOVERY.value, "name": "Study Search", "enabled": True, "reason": None},
            {"key": PipelineKind.REVIEW.value, "name": "Deep Literature Review", "enabled": True, "reason": None},
            {"key": PipelineKind.NOVELTY.value, "name": "Novelty Generation", "enabled": True, "reason": None},
            {"key": PipelineKind.EXPERIMENT.value, "name": "Experiment Design", "enabled": True, "reason": None,
             "execution": {"state": "approval_gated", "approval_required": True, "local_execution_allowed": bool(getattr(c, "worker", None)),
             "remote_execution_allowed": False, "worker_types": ["local_typed"] if getattr(c, "worker", None) else [], "isolation_strength": "workspace-only",
             "network_enforcement": "policy-only", "max_concurrency": 1,
             "reason": "Local execution is workspace-contained and approval-gated; it is not an OS sandbox."}},
            {"key": PipelineKind.REINVENTION.value, "name": "Result-Conditioned Reinvention", "enabled": True, "reason": None},
        ]
        return {"pipelines": pipelines, "capabilities": c.agent.capabilities.payload(), "tools": [{"id": item.id, "description": item.description, "permission": item.permission, "mutating": item.mutating, "schema": item.schema} for item in c.agent.tools.definitions()]}

    @app.get("/v1/agent/sessions", dependencies=[Depends(require_auth)])
    async def agent_sessions(project_id: str | None = None, active_repository: str | None = None):
        c = container()
        return {"sessions": [dataclass_payload(item) for item in c.agent.store.list_sessions(project_id=project_id, repository=active_repository)]}

    @app.post("/v1/agent/sessions", status_code=201, dependencies=[Depends(require_auth)])
    async def create_agent_session(payload: AgentSessionRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        key = mutation_key(idempotency_key)
        prior = c.repository.row("SELECT response_json FROM agent_idempotency WHERE key=?", (key,)) if idempotency_key else None
        if prior:
            return json.loads(prior["response_json"])
        project_id = payload.project_id
        if project_id:
            project_or_404(c.repository, project_id)
        else:
            repository_path = Path(payload.active_repository or Path.cwd()).expanduser().resolve()
            project_id = c.repository.create_project(payload.project_title or repository_path.name).id
        session = c.agent.create_session(project_id, active_repository=payload.active_repository, working_directory=payload.working_directory, token_budget=payload.token_budget, cost_budget=payload.cost_budget)
        response = c.agent.session_payload(session.id)
        if idempotency_key:
            c.agent.store.save_idempotency(key, "agent_session", session.id, None, response)
        c.agent.store.append_event(session.id, "session.created", {"session_id": session.id, "project_id": project_id, "mode": session.selected_mode.value, "auto_routing": session.auto_routing})
        return response

    @app.get("/v1/agent/sessions/{session_id}", dependencies=[Depends(require_auth)])
    async def get_agent_session(session_id: str):
        c = container()
        c.agent.store.require_session(session_id)
        return c.agent.session_payload(session_id)

    @app.post("/v1/agent/sessions/{session_id}/turns", status_code=202, dependencies=[Depends(require_auth)])
    async def agent_turn(session_id: str, payload: AgentTurnRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        c.agent.store.require_session(session_id)
        key = mutation_key(idempotency_key)
        prior = c.agent.store.idempotency(key) if idempotency_key else None
        if prior:
            return prior["response"]
        turn = c.agent.start_turn(session_id, payload.message, capability_override=payload.capability_override, token_budget=payload.token_budget, cost_budget=payload.cost_budget, time_budget_seconds=payload.time_budget_seconds)
        response = {"session_id": session_id, "turn_id": turn.id, "status": turn.status.value}
        if idempotency_key:
            c.agent.store.save_idempotency(key, "agent_turn", session_id, turn.id, response)
        task = asyncio.create_task(c.agent.execute_turn(turn.id))
        agent_tasks[turn.id] = task
        task.add_done_callback(lambda _task: agent_tasks.pop(turn.id, None))
        return response

    @app.get("/v1/agent/sessions/{session_id}/events", dependencies=[Depends(require_auth)])
    async def agent_events(session_id: str, last_event_id: int = Query(default=0, ge=0), last_event_header: str | None = Header(default=None, alias="Last-Event-ID")):
        c = container()
        c.agent.store.require_session(session_id)
        try:
            header_cursor = int(last_event_header or 0)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"code": "invalid_event_cursor", "message": "Last-Event-ID must be a non-negative integer."}) from exc
        if header_cursor < 0:
            raise HTTPException(status_code=400, detail={"code": "invalid_event_cursor", "message": "Last-Event-ID must be a non-negative integer."})
        after = max(last_event_id, header_cursor)

        async def stream_agent() -> AsyncIterator[str]:
            cursor = after
            idle = 0
            while idle < 30:
                events = c.agent.store.events(session_id, after_id=cursor)
                if events:
                    idle = 0
                    for event in events:
                        cursor = int(event["id"])
                        yield f"id: {cursor}\nevent: {event['event_type']}\ndata: {json.dumps(event['payload'], ensure_ascii=False, default=str)}\n\n"
                        if event["event_type"] in {"agent.turn_completed", "agent.turn_failed", "agent.turn_canceled"}:
                            return
                else:
                    idle += 1
                await asyncio.sleep(0.4)

        return StreamingResponse(stream_agent(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.post("/v1/agent/sessions/{session_id}/mode", dependencies=[Depends(require_auth)])
    async def agent_mode(session_id: str, payload: AgentModeRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        c.agent.store.require_session(session_id)
        key = mutation_key(idempotency_key)
        prior = c.agent.store.idempotency(key) if idempotency_key else None
        if prior:
            return prior["response"]
        mode = SessionMode(payload.mode or "default")
        updated = c.agent.store.update_session(session_id, selected_mode=mode, auto_routing=payload.auto_routing)
        c.agent.store.append_event(session_id, "capability.selected", {"session_id": session_id, "mode": updated.selected_mode.value, "auto_routing": updated.auto_routing})
        response = {"session": dataclass_payload(updated)}
        if idempotency_key:
            c.agent.store.save_idempotency(key, "agent_mode", session_id, None, response)
        return response

    @app.post("/v1/agent/sessions/{session_id}/link-project", dependencies=[Depends(require_auth)])
    async def link_project(session_id: str, payload: LinkProjectRequest):
        c = container()
        session = c.agent.store.require_session(session_id)
        if not c.repository.get_project(payload.project_id):
            raise HTTPException(404, f"Project '{payload.project_id}' was not found. Run 'discover --from-catalog' or 'discover --seed-paper' first.")
        # Link the discover project's papers into the session's project
        c.repository.link_project(payload.project_id, session.project_id)
        paper_count = len(c.repository.list_papers(session.project_id))
        c.agent.store.append_event(session_id, "agent.project_linked", {"session_id": session_id, "source_project": payload.project_id, "paper_count": paper_count})
        return {"status": "linked", "session_id": session_id, "source_project": payload.project_id, "paper_count": paper_count}

    @app.post("/v1/agent/sessions/{session_id}/cancel", dependencies=[Depends(require_auth)])
    async def cancel_agent_session(session_id: str, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        c.agent.store.require_session(session_id)
        key = mutation_key(idempotency_key)
        prior = c.agent.store.idempotency(key) if idempotency_key else None
        if prior:
            return prior["response"]
        c.agent.cancel(session_id)
        for turn_id, task in list(agent_tasks.items()):
            turn = c.agent.store.turn(turn_id)
            if turn and turn.session_id == session_id and not task.done():
                task.cancel()
        c.agent.store.append_event(session_id, "agent.turn_canceled", {"session_id": session_id})
        response = {"session_id": session_id, "status": "cancel_requested"}
        if idempotency_key:
            c.agent.store.save_idempotency(key, "agent_cancel", session_id, None, response)
        return response

    @app.post("/v1/agent/suggestions/{suggestion_id}", dependencies=[Depends(require_auth)])
    async def agent_suggestion(suggestion_id: str, payload: AgentDecisionRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        suggestion = c.agent.store.suggestion(suggestion_id)
        if not suggestion:
            raise HTTPException(status_code=404, detail={"code": "suggestion_not_found", "message": "Capability suggestion was not found."})
        key = mutation_key(idempotency_key)
        prior = c.agent.store.idempotency(key) if idempotency_key else None
        if prior:
            return prior["response"]
        turn = await c.agent.approve_suggestion(suggestion_id, approved=payload.approved)
        if turn:
            task = asyncio.create_task(c.agent.execute_turn(turn.id))
            agent_tasks[turn.id] = task
            task.add_done_callback(lambda _task: agent_tasks.pop(turn.id, None))
        response = {"suggestion": c.agent.store.suggestion(suggestion_id), "turn_id": turn.id if turn else None}
        if idempotency_key:
            c.agent.store.save_idempotency(key, "agent_suggestion", str(suggestion["session_id"]), turn.id if turn else None, response)
        return response

    @app.post("/v1/agent/permissions/{permission_id}", dependencies=[Depends(require_auth)])
    async def agent_permission(permission_id: str, payload: AgentDecisionRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        permission = c.agent.store.permission(permission_id)
        if not permission:
            raise HTTPException(status_code=404, detail={"code": "permission_not_found", "message": "Permission request was not found."})
        key = mutation_key(idempotency_key)
        prior = c.agent.store.idempotency(key) if idempotency_key else None
        if prior:
            return prior["response"]
        turn = await c.agent.decide_permission(permission_id, approved=payload.approved, expires_at=payload.expires_at)
        if turn and turn.status is TurnStatus.RUNNING:
            task = asyncio.create_task(c.agent.continue_after_permission(turn.id))
            agent_tasks[turn.id] = task
            task.add_done_callback(lambda _task: agent_tasks.pop(turn.id, None))
        response = {"permission": c.agent.store.permission(permission_id), "turn_id": turn.id if turn else None}
        if idempotency_key:
            c.agent.store.save_idempotency(key, "agent_permission", str(permission["session_id"]), turn.id if turn else None, response)
        return response

    @app.get("/v1/providers/status", dependencies=[Depends(require_auth)])
    async def provider_status():
        c = container()
        entries = []
        for name in c.router.registry.names():
            provider = c.router.registry.get(name)
            config = getattr(provider, "config", None)
            key_env = getattr(config, "api_key_env", None)
            entries.append({"name": name, "kind": getattr(config, "kind", None), "base_url": redact_text(str(getattr(config, "base_url", None))), "configured": bool(os.getenv(key_env, "")) if key_env else False, "capabilities": sorted(str(item.value) for item in getattr(provider, "capabilities", ()))})
        roles = []
        for role, assignment in c.router.assignments.items():
            roles.append({"role": role.value, "provider": assignment.provider, "model": assignment.model, "fallbacks": list(assignment.fallbacks)})
        return {"providers": entries, "roles": roles}

    @app.post("/v1/providers/select", dependencies=[Depends(require_auth)])
    async def select_provider(payload: ProviderSelectionRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        key = mutation_key(idempotency_key)
        prior = c.repository.idempotency(key) if idempotency_key else None
        if prior:
            return prior["response"]
        try:
            provider = c.router.registry.get(payload.provider)
            current = c.router.assignments.get(payload.role)
            assignment = ModelAssignment(payload.role, payload.provider, payload.model, current.fallbacks if current else ())
            c.router.registry.validate_assignment(assignment)
            c.router.assignments[payload.role] = assignment
            if payload.role is ModelRole.EMBEDDINGS:
                c.router.registry.get_embeddings(payload.provider).model_name = payload.model
        except (KeyError, ValueError, ProviderError) as exc:
            raise HTTPException(status_code=422, detail={"code": "invalid_provider_selection", "message": redact_text(str(exc))}) from exc
        response = {"selected": {"role": payload.role.value, "provider": payload.provider, "model": payload.model}, "durable": False}
        if idempotency_key:
            c.repository.save_idempotency(key, "provider_selection", "provider-selection", response)
        return response

    @app.get("/v1/usage", dependencies=[Depends(require_auth)])
    async def usage(project_id: str | None = None, run_id: str | None = None):
        c = container()
        records = c.repository.usage(project_id=project_id, run_id=run_id)
        return {"records": records, "requests": len(records), "estimated_cost": round(sum(float(item.get("estimated_cost") or 0) for item in records), 12), "currency": "USD"}

    @app.get("/v1/projects", dependencies=[Depends(require_auth)])
    async def projects():
        c = container()
        return {"projects": [dataclass_payload(item) for item in c.repository.list_projects()]}

    @app.post("/v1/projects", status_code=201, dependencies=[Depends(require_auth)])
    async def create_project(payload: ProjectRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        key = mutation_key(idempotency_key)
        prior = c.repository.project_idempotency(key)
        if prior:
            return prior["response"]
        response = dataclass_payload(c.repository.create_project(payload.title))
        c.repository.save_project_idempotency(key, str(response["id"]), response)
        return response

    @app.get("/v1/projects/{project_id}", dependencies=[Depends(require_auth)])
    async def get_project(project_id: str):
        c = container()
        project = project_or_404(c.repository, project_id)
        return {"project": dataclass_payload(project), "seed": dataclass_payload(c.repository.get_seed(project_id)) if c.repository.get_seed(project_id) else None, "papers": [dataclass_payload(item) for item in c.repository.list_papers(project_id)], "artifacts": [dataclass_payload(item) for item in c.repository.artifacts(project_id)], "latest_gap": dataclass_payload(c.repository.latest_gap(project_id)) if c.repository.latest_gap(project_id) else None}

    @app.post("/v1/projects/{project_id}/discover", status_code=202, dependencies=[Depends(require_auth)])
    async def discover(project_id: str, payload: DiscoverRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        project_or_404(c.repository, project_id)
        key = mutation_key(idempotency_key)
        prior = c.repository.idempotency(key)
        if prior:
            return prior["response"]
        input_data = payload.model_dump()
        input_data["seed_input"] = input_data.pop("seed_paper")
        run = c.repository.create_run(project_id, PipelineKind.DISCOVERY, input_data)
        if payload.max_cost_usd is not None:
            c.repository.set_project_budget(project_id, hard_stop=payload.max_cost_usd)
        response = {"run_id": run.id, "project_id": project_id, "kind": run.kind.value, "status": run.status.value}
        c.repository.save_idempotency(key, "discover", run.id, response)
        emit(c.repository, run.id, "run.created", response)
        task = asyncio.create_task(execute_discovery(c, run.id, payload, project_id))
        tasks[run.id] = task
        task.add_done_callback(lambda _task: tasks.pop(run.id, None))
        return response

    @app.post("/v1/runs/{run_id}/resume", status_code=202, dependencies=[Depends(require_auth)])
    async def resume_run(run_id: str, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        run = c.repository.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail={"code": "run_not_found", "message": f"Run '{run_id}' was not found."})
        if run.kind not in {PipelineKind.DISCOVERY, PipelineKind.REVIEW}:
            raise HTTPException(status_code=409, detail={"code": "capability_unavailable", "message": f"Pipeline '{run.kind.value}' cannot be resumed."})
        if run_id in tasks and not tasks[run_id].done():
            return {"run_id": run_id, "project_id": run.project_id, "kind": run.kind.value, "status": run.status.value}
        key = mutation_key(idempotency_key)
        prior = c.repository.idempotency(key)
        if prior:
            return prior["response"]
        c.repository.resume_run(run_id)
        response = {"run_id": run_id, "project_id": run.project_id, "kind": run.kind.value, "status": RunStatus.RUNNING.value}
        c.repository.save_idempotency(key, "resume", run_id, response)
        emit(c.repository, run_id, "run.resumed", response)

        async def resume_operation() -> None:
            monitor = asyncio.create_task(monitor_run(c.repository, run_id))
            monitors[run_id] = monitor
            try:
                if run.kind is PipelineKind.DISCOVERY:
                    await c.discovery.resume(run_id)
                else:
                    await c.review.resume(run_id)
            except asyncio.CancelledError:
                c.repository.cancel_run(run_id)
                emit(c.repository, run_id, "run.cancelled", {"run_id": run_id})
                raise
            except Exception as exc:
                current = c.repository.get_run(run_id)
                if current and current.status is RunStatus.RUNNING:
                    c.repository.finish_run(run_id, error=redact_text(f"{type(exc).__name__}: {exc}"))
                emit(c.repository, run_id, "run.error", {"message": redact_text(f"{type(exc).__name__}: {exc}")})
            finally:
                if not monitor.done():
                    await monitor
                monitors.pop(run_id, None)

        task = asyncio.create_task(resume_operation())
        tasks[run_id] = task
        task.add_done_callback(lambda _task: tasks.pop(run_id, None))
        return response

    @app.post("/v1/projects/{project_id}/review", status_code=202, dependencies=[Depends(require_auth)])
    async def review(project_id: str, payload: ReviewRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        project_or_404(c.repository, project_id)
        key = mutation_key(idempotency_key)
        prior = c.repository.idempotency(key)
        if prior:
            return prior["response"]
        run = c.repository.create_run(project_id, PipelineKind.REVIEW, payload.model_dump())
        if payload.max_cost_usd is not None:
            c.repository.set_project_budget(project_id, hard_stop=payload.max_cost_usd)
        response = {"run_id": run.id, "project_id": project_id, "kind": run.kind.value, "status": run.status.value}
        c.repository.save_idempotency(key, "review", run.id, response)
        emit(c.repository, run.id, "run.created", response)
        task = asyncio.create_task(execute_review(c, run.id, project_id, payload))
        tasks[run.id] = task
        task.add_done_callback(lambda _task: tasks.pop(run.id, None))
        return response

    @app.post("/v1/projects/{project_id}/invent", status_code=202, dependencies=[Depends(require_auth)])
    async def invent(project_id: str, payload: NoveltyRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        project_or_404(c.repository, project_id)
        key = mutation_key(idempotency_key)
        prior = c.repository.idempotency(key)
        if prior:
            return prior["response"]
        run = c.repository.create_run(project_id, PipelineKind.NOVELTY, payload.model_dump())
        if payload.max_cost_usd is not None:
            c.repository.set_project_budget(project_id, hard_stop=payload.max_cost_usd)
        response = {"run_id": run.id, "project_id": project_id, "kind": run.kind.value, "status": run.status.value}
        c.repository.save_idempotency(key, "invent", run.id, response)
        emit(c.repository, run.id, "run.created", response)
        task = asyncio.create_task(execute_invent(c, run.id, project_id, payload))
        tasks[run.id] = task
        task.add_done_callback(lambda _task: tasks.pop(run.id, None))
        return response

    @app.post("/v1/projects/{project_id}/experiment", status_code=202, dependencies=[Depends(require_auth)])
    async def experiment(project_id: str, payload: ExperimentRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        project_or_404(c.repository, project_id)
        key = mutation_key(idempotency_key)
        prior = c.repository.idempotency(key)
        if prior:
            return prior["response"]
        run = c.repository.create_run(project_id, PipelineKind.EXPERIMENT, payload.model_dump())
        if payload.max_cost_usd is not None:
            c.repository.set_project_budget(project_id, hard_stop=payload.max_cost_usd)
        response = {"run_id": run.id, "project_id": project_id, "kind": run.kind.value, "status": run.status.value}
        c.repository.save_idempotency(key, "experiment", run.id, response)
        emit(c.repository, run.id, "run.created", response)
        task = asyncio.create_task(execute_experiment(c, run.id, project_id, payload))
        tasks[run.id] = task
        task.add_done_callback(lambda _task: tasks.pop(run.id, None))
        return response

    @app.post("/v1/projects/{project_id}/reinvent", status_code=409, dependencies=[Depends(require_auth)])
    async def reinvent(project_id: str):
        c = container()
        project_or_404(c.repository, project_id)
        raise HTTPException(status_code=409, detail={"code": "capability_unavailable", "message": "reinvent: Result-Conditioned Reinvention is not implemented in the Python backend."})

    @app.post("/v1/projects/{project_id}/{pipeline}", status_code=409, dependencies=[Depends(require_auth)])
    async def unavailable_pipeline(project_id: str, pipeline: str):
        if pipeline in {"reinvent"}:
            raise HTTPException(status_code=409, detail={"code": "capability_unavailable", "message": f"{pipeline} is not implemented in the Python backend."})
        raise HTTPException(status_code=404, detail={"code": "not_found", "message": "Unknown pipeline."})

    @app.get("/v1/runs", dependencies=[Depends(require_auth)])
    async def runs(project_id: str | None = None, status: RunStatus | None = None):
        c = container()
        return {"runs": [dataclass_payload(item) for item in c.repository.list_runs(project_id=project_id, status=status)]}

    @app.get("/v1/runs/{run_id}", dependencies=[Depends(require_auth)])
    async def get_run(run_id: str):
        c = container()
        run_payload_value = run_payload(c.repository, run_id)
        try:
            preview = c.experiment.approval_preview(run_id)
        except ValueError:
            preview = None
        return {"run": run_payload_value, "checkpoints": c.repository.checkpoints(run_id), "artifacts": [dataclass_payload(item) for item in c.repository.artifacts_for_run(run_id)], "usage": c.repository.usage(run_id=run_id), "worker_jobs": c.repository.worker_jobs_for_run(run_id), "state_transitions": c.repository.worker_transitions(run_id), "approval": c.repository.exact_approval(run_id), "approval_preview": preview}

    @app.get("/v1/runs/{run_id}/events", dependencies=[Depends(require_auth)])
    async def run_events(run_id: str, last_event_id: int = Query(default=0, ge=0), last_event_header: str | None = Header(default=None, alias="Last-Event-ID")):
        c = container()
        run_payload(c.repository, run_id)
        try:
            header_cursor = int(last_event_header or 0)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail={"code": "invalid_event_cursor", "message": "Last-Event-ID must be a non-negative integer."}) from exc
        if header_cursor < 0:
            raise HTTPException(status_code=400, detail={"code": "invalid_event_cursor", "message": "Last-Event-ID must be a non-negative integer."})
        after = max(last_event_id, header_cursor)
        page_size = 200

        async def stream() -> AsyncIterator[str]:
            cursor = after
            idle = 0
            while idle < 30:
                events = c.repository.api_events(run_id, after_id=cursor, limit=page_size)
                if events:
                    idle = 0
                    for event in events:
                        cursor = int(event["id"])
                        yield f"id: {cursor}\nevent: {event['event_type']}\ndata: {json.dumps(event['payload'], ensure_ascii=False, default=str)}\n\n"
                    if len(events) == page_size:
                        continue
                    terminal = c.repository.get_run(run_id)
                    if terminal and terminal.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED} and len(events) < page_size:
                        return
                else:
                    idle += 1
                    terminal = c.repository.get_run(run_id)
                    if terminal and terminal.status in {RunStatus.COMPLETED, RunStatus.FAILED, RunStatus.CANCELLED}:
                        return
                await asyncio.sleep(0.4)

        return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/v1/runs/{run_id}/artifacts", dependencies=[Depends(require_auth)])
    async def run_artifacts(run_id: str):
        c = container()
        run_payload(c.repository, run_id)
        return {"artifacts": [dataclass_payload(item) for item in c.repository.artifacts_for_run(run_id)]}

    @app.post("/v1/runs/{run_id}/cancel", dependencies=[Depends(require_auth)])
    async def cancel_run(run_id: str, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        run = c.repository.get_run(run_id)
        if not run:
            raise HTTPException(status_code=404, detail={"code": "run_not_found", "message": f"Run '{run_id}' was not found."})
        key = mutation_key(idempotency_key)
        prior = c.repository.idempotency(key)
        if prior:
            return prior["response"]
        if run.status is RunStatus.CANCELLED:
            response = {"run_id": run_id, "status": RunStatus.CANCELLED.value}
            c.repository.save_idempotency(key, "cancel", run_id, response)
            return response
        if run.status in {RunStatus.COMPLETED, RunStatus.FAILED}:
            raise HTTPException(status_code=409, detail={"code": "run_not_cancellable", "message": f"Run '{run_id}' is already {run.status.value}."})
        task = tasks.get(run_id)
        if task and not task.done():
            task.cancel()
        for job in c.repository.worker_jobs_for_run(run_id):
            if job.get("state") in {"QUEUED", "LEASED", "RUNNING"} and getattr(c, "worker", None):
                await c.worker.cancel(str(job["id"]))
        c.repository.cancel_run(run_id)
        emit(c.repository, run_id, "run.cancelled", {"run_id": run_id})
        response = {"run_id": run_id, "status": RunStatus.CANCELLED.value}
        c.repository.save_idempotency(key, "cancel", run_id, response)
        return response

    @app.post("/v1/runs/{run_id}/approve", dependencies=[Depends(require_auth)])
    async def approve_run(run_id: str, payload: ApprovalRequest, idempotency_key: str | None = Header(default=None, alias="Idempotency-Key")):
        c = container()
        run_payload(c.repository, run_id)
        key = mutation_key(idempotency_key)
        prior = c.repository.idempotency(key)
        if prior:
            return prior["response"]
        try:
            preview = c.experiment.approval_preview(run_id)
        except ValueError:
            c.repository.set_approval(run_id, "approved" if payload.approved else "rejected", payload.reason)
            response = {"run_id": run_id, "status": "approved" if payload.approved else "rejected", "reason": payload.reason}
            c.repository.save_idempotency(key, "approve", run_id, response)
            emit(c.repository, run_id, "run.approval", response)
            return response
        if payload.specification_checksum and payload.specification_checksum != preview["checksum"]:
            raise HTTPException(status_code=409, detail={"code": "checksum_mismatch", "message": "Approval checksum does not match the durable experiment specification."})
        approval = c.repository.save_exact_approval({"id": uuid4().hex, "run_id": run_id, "specification_checksum": str(preview["checksum"]), "approver": payload.approver, "approved_scope": payload.approved_scope, "approved_target": payload.approved_target, "approved_budget": payload.approved_budget, "expires_at": payload.expires_at or "9999-12-31T23:59:59+00:00", "status": "pending", "approved_actions": preview["actions"]})
        if payload.approved:
            approval = c.repository.approve_exact(str(approval["id"]), approver=payload.approver) or approval
        else:
            c.repository.execute("UPDATE experiment_approvals SET status='rejected' WHERE id=?", (approval["id"],))
        emit(c.repository, run_id, "approval.granted" if payload.approved else "approval.rejected", {"run_id": run_id, "approval_id": approval["id"], "checksum": preview["checksum"], "approved": payload.approved, "reason": payload.reason})
        response = {"run_id": run_id, "approval_id": approval["id"], "status": "approved" if payload.approved else "rejected", "checksum": preview["checksum"], "reason": payload.reason}
        c.repository.save_idempotency(key, "approve", run_id, response)
        return response

    @app.get("/v1/artifacts/{artifact_id}", dependencies=[Depends(require_auth)])
    async def artifact(artifact_id: str, offset: int = Query(default=0, ge=0), limit: int = Query(default=65536, ge=1, le=200000)):
        c = container()
        rows = c.repository.rows("SELECT * FROM artifacts WHERE id=?", (artifact_id,))
        if not rows:
            raise HTTPException(status_code=404, detail={"code": "artifact_not_found", "message": "Artifact was not found."})
        path = Path(rows[0]["path"]).resolve()
        root = c.settings.artifact_root.resolve()
        if root not in path.parents and path != root:
            raise HTTPException(status_code=403, detail={"code": "artifact_forbidden", "message": "Artifact path is outside the artifact root."})
        if not path.exists():
            raise HTTPException(status_code=404, detail={"code": "artifact_missing", "message": "Artifact file is missing."})
        size = path.stat().st_size
        with path.open("rb") as handle:
            handle.seek(offset)
            payload = handle.read(limit)
            next_offset = offset + len(payload)
        truncated = next_offset < size
        return {"artifact": rows[0], "content": payload.decode("utf-8", errors="replace"), "offset": offset, "next_offset": next_offset if truncated else None, "size": size, "truncated": truncated}

    return app


__version__ = API_VERSION


def main() -> None:
    import uvicorn

    uvicorn.run("autoresearch.api.app:create_app", factory=True, host=os.getenv("AUTORESEARCH_API_HOST", "127.0.0.1"), port=int(os.getenv("AUTORESEARCH_API_PORT", "8000")))
