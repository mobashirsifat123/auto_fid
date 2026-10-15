"""Purpose: Enforces human-in-the-loop safety."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path

from autoresearch.domain.models import ExperimentSpec


class ActionType(StrEnum):
    RUN_PYTHON_MODULE = "run_python_module"
    RUN_PYTHON_SCRIPT = "run_python_script"
    RUN_PYTEST = "run_pytest"
    VERIFY_IMPORT = "verify_import"
    VERIFY_CHECKPOINT = "verify_checkpoint"
    PARSE_METRICS = "parse_metrics"


@dataclass(frozen=True)
class ExecutionAction:
    kind: ActionType
    target: str
    arguments: tuple[str, ...] = ()


def specification_checksum(spec: ExperimentSpec, actions: tuple[ExecutionAction, ...], worker: str) -> str:
    payload = {"spec": asdict(spec), "actions": [asdict(action) for action in actions], "worker": worker}
    return hashlib.sha256(json.dumps(payload, default=str, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def resolve_action(action: ExecutionAction, repository_root: Path, python_executable: str) -> tuple[str, ...]:
    root = repository_root.expanduser().resolve(strict=True)
    if any("\x00" in value or "\n" in value or "\r" in value for value in (action.target, *action.arguments)):
        raise ValueError("unsafe_execution_request: control characters are prohibited")
    if action.kind == ActionType.RUN_PYTHON_MODULE:
        if not action.target or any(part in {"", ".", ".."} for part in action.target.split(".")):
            raise ValueError("unsafe_execution_request: invalid Python module")
        return (python_executable, "-m", action.target, *action.arguments)
    if action.kind == ActionType.RUN_PYTEST:
        target = _inside(root, action.target or ".")
        return (python_executable, "-m", "pytest", str(target), *action.arguments)
    if action.kind == ActionType.RUN_PYTHON_SCRIPT:
        target = _inside(root, action.target)
        if target.suffix != ".py":
            raise ValueError("unsafe_execution_request: script must be Python")
        return (python_executable, str(target), *action.arguments)
    if action.kind == ActionType.VERIFY_IMPORT:
        if not action.target or any(part in {"", ".", ".."} for part in action.target.split(".")):
            raise ValueError("unsafe_execution_request: invalid import module")
        return (python_executable, "-c", "import importlib; importlib.import_module(%r)" % action.target)
    if action.kind in {ActionType.VERIFY_CHECKPOINT, ActionType.PARSE_METRICS}:
        target = _inside(root, action.target)
        return (python_executable, "-c", "from pathlib import Path; p=Path(%r); raise SystemExit(0 if p.is_file() else 2)" % str(target), *action.arguments)
    raise ValueError("unsafe_execution_request: unsupported action")


def _inside(root: Path, value: str) -> Path:
    raw = Path(value)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError("unsafe_execution_request: absolute paths and traversal are prohibited")
    try:
        resolved = (root / raw).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValueError("unsafe_execution_request: target path does not exist") from exc
    if resolved != root and root not in resolved.parents:
        raise ValueError("unsafe_execution_request: path escapes repository")
    return resolved
