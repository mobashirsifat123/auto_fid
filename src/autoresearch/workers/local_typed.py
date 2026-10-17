"""Purpose: Executes approved experiment scripts without arbitrary shell execution."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from uuid import uuid4

from autoresearch.domain.models import ExperimentSpec, utc_now
from autoresearch.experiments.policy import ActionType, ExecutionAction, resolve_action, specification_checksum
from autoresearch.storage.repository import ResearchRepository


@dataclass(frozen=True)
class WorkerCapabilities:
    worker_id: str = "local"
    actions: tuple[str, ...] = tuple(item.value for item in ActionType)
    isolation: str = "workspace-only"
    network: str = "disabled-by-policy"
    max_concurrency: int = 1
    max_runtime_seconds: int = 3600


@dataclass(frozen=True)
class WorkerSubmission:
    job_id: str
    checksum: str
    state: str


class LocalTypedWorker:
    """Execute only action vectors produced by ``resolve_action``.

    This is workspace containment and environment filtering, not an OS sandbox.
    """

    def __init__(self, repository: ResearchRepository, workspace_root: Path, *, worker_id: str = "local") -> None:
        self.repository = repository
        self.workspace_root = workspace_root.expanduser().resolve()
        self.worker_id = worker_id
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._cancelled: set[str] = set()

    def capabilities(self) -> WorkerCapabilities:
        return WorkerCapabilities(worker_id=self.worker_id)

    def _event(self, run_id: str, event_type: str, payload: dict[str, object]) -> None:
        self.repository.append_api_event(run_id, event_type, payload)

    def _workspace(self, spec: ExperimentSpec) -> Path:
        root = self.workspace_root
        repo = Path(spec.repository).expanduser().resolve(strict=True)
        if repo != root and root not in repo.parents and repo not in root.parents:
            raise ValueError("unsafe_execution_request: repository is outside worker workspace")
        return repo

    def _result_store_path(self, spec: ExperimentSpec, job_id: str) -> Path:
        return self._workspace(spec) / "run_outputs" / f"worker_result_{job_id}.json"

    def _persist_result(self, spec: ExperimentSpec, job_id: str, payload: dict[str, object]) -> None:
        """Persist the job result so a later process can reuse the measured
        stdout deterministically instead of returning an stdout-less job row
        (worker_jobs stores bookkeeping only, never outputs)."""
        try:
            path = self._result_store_path(spec, job_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        except OSError:
            pass  # bookkeeping only — the in-memory result is returned regardless

    def _stored_result(self, spec: ExperimentSpec, job_id: str) -> dict[str, object] | None:
        try:
            path = self._result_store_path(spec, job_id)
            if not path.is_file():
                return None
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        return payload if isinstance(payload, dict) and "stdout" in payload else None

    def submit(self, spec: ExperimentSpec, actions: tuple[ExecutionAction, ...], *, approved_checksum: str, timeout_seconds: int = 3600) -> WorkerSubmission:
        expected = specification_checksum(spec, actions, self.worker_id)
        if approved_checksum != expected:
            raise PermissionError("checksum_mismatch: exact experiment specification approval does not match")
        self._workspace(spec)
        if not actions:
            raise ValueError("unsafe_execution_request: at least one typed action is required")
        for action in actions:
            resolve_action(action, self._workspace(spec), sys.executable)
        existing = self.repository.row("SELECT * FROM worker_jobs WHERE spec_id=? AND checksum=? AND state IN ('QUEUED','LEASED','RUNNING','ANALYZING','COMPLETED') ORDER BY created_at DESC LIMIT 1", (spec.id, expected))
        if existing:
            return WorkerSubmission(existing["id"], expected, existing["state"])
        job_id = uuid4().hex
        now = utc_now()
        self.repository.save_worker_job({"id": job_id, "project_id": spec.project_id, "run_id": spec.run_id, "spec_id": spec.id, "worker": self.worker_id, "state": "QUEUED", "attempt": 0, "checksum": expected, "actions": [asdict(action) for action in actions], "timeout_seconds": min(timeout_seconds, self.capabilities().max_runtime_seconds), "created_at": now, "updated_at": now})
        self.repository.append_worker_transition(spec.run_id, "AWAITING_APPROVAL", "QUEUED", reason="exact approval verified", worker_id=self.worker_id)
        self._event(spec.run_id, "job.queued", {"job_id": job_id, "worker": self.worker_id, "checksum": expected})
        return WorkerSubmission(job_id, expected, "QUEUED")

    async def run(self, spec: ExperimentSpec, actions: tuple[ExecutionAction, ...], *, approved_checksum: str, timeout_seconds: int = 3600) -> dict[str, object]:
        submission = self.submit(spec, actions, approved_checksum=approved_checksum, timeout_seconds=timeout_seconds)
        job = self.repository.worker_job(submission.job_id)
        assert job is not None
        if job["state"] == "COMPLETED":
            stored = self._stored_result(spec, submission.job_id)
            if stored is not None:
                return stored
            # The result payload died with the previous process (results are
            # process-local; worker_jobs keeps bookkeeping only). Re-execute the
            # SAME approved deterministic action rather than returning an
            # stdout-less row the measurement ingester cannot use.
        lease_id = uuid4().hex
        now = utc_now()
        self.repository.save_worker_job({**job, "lease_id": lease_id, "state": "LEASED", "attempt": int(job["attempt"]) + 1, "updated_at": now, "heartbeat_at": now, "lease_expires_at": now})
        self.repository.append_worker_transition(spec.run_id, "QUEUED", "LEASED", reason="local lease acquired", worker_id=self.worker_id, attempt=int(job["attempt"]) + 1)
        self._event(spec.run_id, "job.leased", {"job_id": submission.job_id, "worker": self.worker_id, "attempt": int(job["attempt"]) + 1})
        output: list[str] = []
        errors: list[str] = []
        exit_code = 0
        for raw in job["actions"]:
            action = ExecutionAction(ActionType(raw["kind"]), str(raw["target"]), tuple(str(item) for item in raw.get("arguments", ())))
            argv = resolve_action(action, self._workspace(spec), sys.executable)
            current_job = self.repository.worker_job(submission.job_id) or job
            self.repository.save_worker_job({**current_job, "state": "RUNNING", "updated_at": utc_now(), "heartbeat_at": utc_now()})
            self.repository.append_worker_transition(spec.run_id, "LEASED", "RUNNING", reason=action.kind.value, worker_id=self.worker_id, attempt=int(job["attempt"]) + 1)
            self._event(spec.run_id, "job.started", {"job_id": submission.job_id, "action": action.kind.value})
            env = {"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1"}
            process = await asyncio.create_subprocess_exec(*argv, cwd=str(self._workspace(spec)), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True, env=env)
            self._processes[submission.job_id] = process
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=min(timeout_seconds, self.capabilities().max_runtime_seconds))
            except asyncio.TimeoutError:
                await self.cancel(submission.job_id)
                exit_code = 124
                errors.append("TIMEOUT")
                break
            finally:
                self._processes.pop(submission.job_id, None)
            output.append(stdout.decode(errors="replace")[:1_000_000])
            errors.append(stderr.decode(errors="replace")[:1_000_000])
            exit_code = process.returncode or 0
            if submission.job_id in self._cancelled:
                exit_code = 130
                break
            if exit_code != 0:
                break
        final_state = "CANCELED" if submission.job_id in self._cancelled else ("COMPLETED" if exit_code == 0 else "FAILED")
        current = self.repository.worker_job(submission.job_id) or job
        self.repository.save_worker_job({**current, "state": final_state, "updated_at": utc_now(), "heartbeat_at": utc_now(), "error": {"category": errors[-1]} if final_state == "FAILED" and errors else None})
        self.repository.append_worker_transition(spec.run_id, "RUNNING", final_state, reason="process group finished", worker_id=self.worker_id, attempt=int(current["attempt"]))
        self._event(spec.run_id, "job.completed" if final_state == "COMPLETED" else "job.canceled" if final_state == "CANCELED" else "run.failed", {"job_id": submission.job_id, "state": final_state, "exit_code": exit_code})
        payload: dict[str, object] = {"job_id": submission.job_id, "state": final_state, "exit_code": exit_code, "attempt": int(current.get("attempt") or 0), "stdout": "".join(output), "stderr": "".join(errors), "checksum": approved_checksum}
        if final_state == "COMPLETED":
            self._persist_result(spec, submission.job_id, payload)
        return payload

    async def heartbeat(self, job_id: str) -> dict[str, object] | None:
        job = self.repository.worker_job(job_id)
        if not job:
            return None
        job["heartbeat_at"] = utc_now()
        job["updated_at"] = job["heartbeat_at"]
        self.repository.save_worker_job(job)
        return job

    async def cancel(self, job_id: str) -> dict[str, object] | None:
        job = self.repository.worker_job(job_id)
        if not job:
            return None
        self._cancelled.add(job_id)
        process = self._processes.get(job_id)
        if process and process.returncode is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(process.wait(), timeout=5)
            except asyncio.TimeoutError:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        previous_state = str(job.get("state"))
        job["state"] = "CANCELED"
        job["updated_at"] = utc_now()
        self.repository.save_worker_job(job)
        self.repository.append_worker_transition(job["run_id"], previous_state, "CANCELED", reason="cancellation requested", worker_id=self.worker_id, attempt=int(job["attempt"]))
        self._event(job["run_id"], "job.canceled", {"job_id": job_id})
        return job

    def recover_stale(self, *, stale_after_seconds: int = 120) -> list[dict[str, object]]:
        cutoff = time.time() - stale_after_seconds
        recovered: list[dict[str, object]] = []
        for job in self.repository.rows("SELECT id FROM worker_jobs WHERE state IN ('LEASED','RUNNING')"):
            current = self.repository.worker_job(job["id"])
            if current and current.get("heartbeat_at"):
                from datetime import datetime
                if datetime.fromisoformat(str(current["heartbeat_at"])).timestamp() < cutoff:
                    current["state"] = "STALE"
                    current["updated_at"] = utc_now()
                    self.repository.save_worker_job(current)
                    self.repository.append_worker_transition(current["run_id"], "RUNNING", "STALE", reason="heartbeat expired", worker_id=self.worker_id, attempt=int(current["attempt"]))
                    recovered.append(current)
        return recovered
