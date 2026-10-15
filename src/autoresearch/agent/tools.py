"""Contained generic coding tools for the conversational agent."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any

from autoresearch.providers.redaction import redact_text

from .models import AgentSession, ToolDefinition, ToolObservation


def _object_schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


class AgentToolRegistry:
    def __init__(self) -> None:
        self._definitions = {
            "list_directory": ToolDefinition("list_directory", "List a contained repository directory.", "repository_read", False, _object_schema({"path": {"type": "string", "default": "."}, "limit": {"type": "integer", "minimum": 1, "maximum": 500}})),
            "read_file": ToolDefinition("read_file", "Read a bounded UTF-8 slice from a contained file.", "repository_read", False, _object_schema({"path": {"type": "string"}, "offset": {"type": "integer", "minimum": 0}, "limit": {"type": "integer", "minimum": 1, "maximum": 200000}}, ("path",))),
            "search_code": ToolDefinition("search_code", "Search text in bounded repository files.", "repository_read", False, _object_schema({"query": {"type": "string"}, "path": {"type": "string", "default": "."}, "limit": {"type": "integer", "minimum": 1, "maximum": 200}}, ("query",))),
            "git_inspect": ToolDefinition("git_inspect", "Inspect Git status, diff, or recent history without changing repository state.", "repository_read", False, _object_schema({"operation": {"type": "string", "enum": ["status", "diff", "log"]}, "limit": {"type": "integer", "minimum": 1, "maximum": 50}}, ("operation",))),
            "write_file": ToolDefinition("write_file", "Write one contained text file after explicit scoped approval.", "file_modify", True, _object_schema({"path": {"type": "string"}, "content": {"type": "string"}, "expected_checksum": {"type": "string"}}, ("path", "content"))),
            "terminal_run": ToolDefinition("terminal_run", "Run a short allowlisted development command without a shell.", "local_command", True, _object_schema({"command": {"type": "string"}, "arguments": {"type": "array", "items": {"type": "string"}}, "cwd": {"type": "string", "default": "."}, "timeout_seconds": {"type": "integer", "minimum": 1, "maximum": 120}}, ("command",))),
        }

    def definitions(self) -> tuple[ToolDefinition, ...]:
        return tuple(self._definitions.values())

    def get(self, tool_id: str) -> ToolDefinition:
        value = self._definitions.get(tool_id)
        if value is None:
            raise ValueError(f"Unknown agent tool '{tool_id}'.")
        return value

    def provider_tools(self) -> list[dict[str, Any]]:
        return [{"type": "function", "function": {"name": item.id, "description": item.description, "parameters": item.schema}} for item in self.definitions()]

    async def execute(self, session: AgentSession, tool_id: str, arguments: dict[str, Any]) -> ToolObservation:
        self.get(tool_id)
        root = Path(session.active_repository).resolve(strict=True)
        if tool_id == "list_directory":
            path = _inside(root, str(arguments.get("path") or "."), must_exist=True)
            if not path.is_dir():
                raise ValueError("tool_invalid_input: list_directory path must be a directory")
            limit = _bounded_int(arguments.get("limit"), 200, 1, 500)
            entries = [{"name": item.name, "type": "directory" if item.is_dir() else "file", "size": item.stat().st_size if item.is_file() else None} for item in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))[:limit]]
            return ToolObservation(tool_id, True, json.dumps(entries, ensure_ascii=False), {"path": str(path.relative_to(root)), "count": len(entries)})
        if tool_id == "read_file":
            path = _inside(root, str(arguments.get("path") or ""), must_exist=True)
            if not path.is_file():
                raise ValueError("tool_invalid_input: read_file path must be a file")
            offset = _bounded_int(arguments.get("offset"), 0, 0, 10_000_000_000)
            limit = _bounded_int(arguments.get("limit"), 64_000, 1, 200_000)
            with path.open("rb") as handle:
                handle.seek(offset)
                raw = handle.read(limit)
            return ToolObservation(tool_id, True, raw.decode("utf-8", errors="replace"), {"path": str(path.relative_to(root)), "offset": offset, "truncated": offset + len(raw) < path.stat().st_size, "size": path.stat().st_size})
        if tool_id == "search_code":
            query = str(arguments.get("query") or "")
            if not query or len(query) > 500:
                raise ValueError("tool_invalid_input: search query must contain 1-500 characters")
            base = _inside(root, str(arguments.get("path") or "."), must_exist=True)
            limit = _bounded_int(arguments.get("limit"), 100, 1, 200)
            matches: list[dict[str, Any]] = []
            candidates = [base] if base.is_file() else base.rglob("*")
            for path in candidates:
                if len(matches) >= limit:
                    break
                if not path.is_file() or path.is_symlink() or path.stat().st_size > 1_000_000 or any(part in {".git", "node_modules", ".venv", "dist", "build"} for part in path.parts):
                    continue
                try:
                    lines = path.read_text(encoding="utf-8").splitlines()
                except (OSError, UnicodeDecodeError):
                    continue
                matches.extend({"path": str(path.relative_to(root)), "line": index, "text": line[:500]} for index, line in enumerate(lines, 1) if query.lower() in line.lower())
            return ToolObservation(tool_id, True, json.dumps(matches[:limit], ensure_ascii=False), {"query": query, "count": min(len(matches), limit), "truncated": len(matches) >= limit})
        if tool_id == "git_inspect":
            operation = str(arguments.get("operation") or "status")
            limit = _bounded_int(arguments.get("limit"), 20, 1, 50)
            commands = {"status": ("status", "--short", "--branch"), "diff": ("diff", "--no-ext-diff", "--"), "log": ("log", f"-{limit}", "--oneline", "--decorate")}
            if operation not in commands:
                raise ValueError("tool_invalid_input: unsupported Git inspection operation")
            return await _subprocess_observation(tool_id, ("git", *commands[operation]), root, 30)
        if tool_id == "write_file":
            path = _inside(root, str(arguments.get("path") or ""), must_exist=False)
            content = str(arguments.get("content") or "")
            if len(content.encode()) > 1_000_000:
                raise ValueError("tool_invalid_input: write_file content exceeds 1 MB")
            expected = arguments.get("expected_checksum")
            if expected and path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() != str(expected):
                raise ValueError("file_conflict: expected checksum does not match current file")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.parent != root and root not in path.parent.parents:
                raise ValueError("tool_invalid_input: write path escapes repository")
            with NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
                handle.write(content)
                temporary = Path(handle.name)
            temporary.replace(path)
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            return ToolObservation(tool_id, True, f"Wrote {path.relative_to(root)} ({len(content.encode())} bytes).", {"path": str(path.relative_to(root)), "checksum": checksum})
        if tool_id == "terminal_run":
            command = str(arguments.get("command") or "")
            vector = _terminal_vector(command, tuple(str(value) for value in arguments.get("arguments") or ()))
            cwd = _inside(root, str(arguments.get("cwd") or "."), must_exist=True)
            if not cwd.is_dir():
                raise ValueError("tool_invalid_input: terminal cwd must be a directory")
            timeout = _bounded_int(arguments.get("timeout_seconds"), 60, 1, 120)
            return await _subprocess_observation(tool_id, vector, cwd, timeout)
        raise ValueError(f"Unknown agent tool '{tool_id}'.")


def _inside(root: Path, raw: str, *, must_exist: bool) -> Path:
    value = Path(raw)
    if value.is_absolute() or ".." in value.parts or any("\x00" in part for part in value.parts):
        raise ValueError("tool_invalid_input: paths must be relative and cannot traverse")
    path = (root / value).resolve(strict=must_exist)
    if path != root and root not in path.parents:
        raise ValueError("tool_invalid_input: path escapes active repository")
    return path


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        result = default if value is None else int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("tool_invalid_input: expected an integer") from exc
    if result < minimum or result > maximum:
        raise ValueError(f"tool_invalid_input: integer must be between {minimum} and {maximum}")
    return result


def _terminal_vector(command: str, arguments: tuple[str, ...]) -> tuple[str, ...]:
    if any("\x00" in item or "\n" in item or "\r" in item for item in (command, *arguments)):
        raise ValueError("unsafe_terminal_request: control characters are prohibited")
    name = Path(command).name
    if name in {"python", "python3", Path(sys.executable).name}:
        if arguments == ("--version",):
            return (sys.executable, *arguments)
        if len(arguments) >= 2 and arguments[0] == "-m" and arguments[1] in {"pytest", "compileall"}:
            return (sys.executable, *arguments)
    if name == "pytest":
        return (sys.executable, "-m", "pytest", *arguments)
    if name == "bun" and arguments and arguments[0] in {"test", "typecheck", "build", "run"}:
        return (command, *arguments)
    if name == "npm" and arguments and arguments[0] in {"test", "run"}:
        return (command, *arguments)
    raise ValueError("unsafe_terminal_request: command is outside the short-lived development allowlist")


async def _subprocess_observation(tool_id: str, vector: tuple[str, ...], cwd: Path, timeout: int) -> ToolObservation:
    env = {"PATH": os.getenv("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0", "PYTHONUNBUFFERED": "1", "LANG": os.getenv("LANG", "C.UTF-8")}
    process = await asyncio.create_subprocess_exec(*vector, cwd=str(cwd), stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, start_new_session=True, env=env)
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        process.terminate()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except asyncio.TimeoutError:
            process.kill()
        raise TimeoutError(f"terminal_timeout: command exceeded {timeout} seconds")
    output = redact_text(stdout.decode(errors="replace")[:500_000])
    errors = redact_text(stderr.decode(errors="replace")[:500_000])
    content = output + (f"\n[stderr]\n{errors}" if errors else "")
    return ToolObservation(tool_id, process.returncode == 0, content, {"argv": [Path(vector[0]).name, *vector[1:]], "exit_code": process.returncode, "stdout_truncated": len(stdout) > 500_000, "stderr_truncated": len(stderr) > 500_000})
