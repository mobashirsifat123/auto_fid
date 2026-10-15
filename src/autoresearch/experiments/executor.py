"""Policy-controlled experiment execution.

The backend does not expose this executor unless a worker adapter supplies a
typed action and an approval checksum matching the immutable specification.
"""
from __future__ import annotations

import asyncio
import sys
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, EnvironmentManifest, ExecutionJob, ExecutionResult, ExperimentSpec, RunStatus
from autoresearch.experiments.policy import ExecutionAction, resolve_action, specification_checksum
from autoresearch.storage.repository import ResearchRepository


class DurableExperimentExecutor:
    def __init__(self, repository: ResearchRepository, workspace_root: Path | None = None) -> None:
        self.repository = repository
        self.workspace_root = workspace_root

    async def run(self, spec: ExperimentSpec, manifest: EnvironmentManifest, *, action: ExecutionAction,
                  approved_checksum: str, timeout_seconds: int = 3600) -> ExecutionResult:
        if self.workspace_root is None:
            raise RuntimeError("worker_unavailable: execution workspace is not configured")
        expected = specification_checksum(spec, (action,), spec.target_worker)
        if approved_checksum != expected:
            raise PermissionError("specification_changed: approval checksum does not match")
        argv = resolve_action(action, Path(spec.repository), sys.executable)
        existing = self.repository.row("SELECT id FROM execution_jobs WHERE spec_id=? AND stage=?", (spec.id, action.kind.value))
        if existing:
            previous = self.repository.get_execution_result(existing["id"])
            if previous:
                return previous
            raise RuntimeError("duplicate_execution: an unfinished immutable job already exists")
        job = ExecutionJob(uuid4().hex, spec.project_id, spec.run_id, spec.id, spec.target_worker,
                           action.kind.value, RunStatus.RUNNING, timeout_seconds=timeout_seconds, retry_policy="bounded")
        self.repository.save_execution_job(job)
        process = await asyncio.create_subprocess_exec(*argv, cwd=str(Path(spec.repository).resolve()),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True,
            env={"PATH": "/usr/bin:/bin", "PYTHONHASHSEED": "0"})
        try:
            stdout_raw, stderr_raw = await asyncio.wait_for(process.communicate(), timeout=timeout_seconds)
        except TimeoutError:
            process.kill()
            await process.wait()
            stdout_raw, stderr_raw = b"", b"execution timed out"
        stdout = stdout_raw.decode(errors="replace")[:1_000_000]
        stderr = stderr_raw.decode(errors="replace")[:1_000_000]
        completed = replace(job, status=RunStatus.COMPLETED if process.returncode == 0 else RunStatus.FAILED)
        self.repository.save_execution_job(completed)
        provenance = ArtifactProvenance.create(project_id=spec.project_id, run_id=spec.run_id,
            source_evidence=[job.id, approved_checksum], provider="deterministic", model="typed-worker",
            prompt_version="executor-v2", content={"argv": argv, "exit_code": process.returncode})
        result = ExecutionResult(uuid4().hex, spec.project_id, spec.run_id, job.id, process.returncode or 0,
            stdout, stderr, {}, {}, (), None if process.returncode == 0 else "IMPLEMENTATION_FAILURE", provenance)
        self.repository.save_execution_result(result)
        return result
