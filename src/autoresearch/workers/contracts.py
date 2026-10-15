"""Provider-neutral worker contracts for future Pro6000/3090 experiment runs."""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class WorkerSpec:
    name: str
    host: str
    gpu_count: int
    workspace: str
    python_environment: str


@dataclass(frozen=True)
class WorkerResult:
    exit_code: int
    stdout: str
    stderr: str


class ScientificWorker(Protocol):
    async def execute(self, command: tuple[str, ...], *, timeout_seconds: int) -> WorkerResult: ...
