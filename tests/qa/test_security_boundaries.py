from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from autoresearch.agent.models import AgentSession
from autoresearch.agent.tools import AgentToolRegistry
from autoresearch.providers.redaction import redact_text


def _session(root: Path) -> AgentSession:
    return AgentSession("qa-session", "qa-project", str(root), str(root))


def test_repository_tools_reject_traversal_and_absolute_paths(tmp_path: Path):
    registry = AgentToolRegistry()
    session = _session(tmp_path)
    for tool_id, arguments in (
        ("read_file", {"path": "../outside.txt"}),
        ("read_file", {"path": str(tmp_path / "outside.txt")}),
        ("list_directory", {"path": "../"}),
        ("search_code", {"query": "secret", "path": "../"}),
    ):
        with pytest.raises(ValueError, match="paths must be relative"):
            asyncio.run(registry.execute(session, tool_id, arguments))


def test_repository_tools_reject_symlink_escape(tmp_path: Path):
    outside = tmp_path.parent / "autoresearch-qa-outside.txt"
    outside.write_text("must stay outside", encoding="utf-8")
    (tmp_path / "escape.txt").symlink_to(outside)
    try:
        with pytest.raises(ValueError, match="path escapes active repository"):
            asyncio.run(AgentToolRegistry().execute(_session(tmp_path), "read_file", {"path": "escape.txt"}))
    finally:
        outside.unlink(missing_ok=True)


def test_terminal_tool_rejects_shell_and_control_character_injection(tmp_path: Path):
    registry = AgentToolRegistry()
    session = _session(tmp_path)
    for arguments in (
        {"command": "sh", "arguments": ["-c", "touch escaped.txt"]},
        {"command": "python", "arguments": ["-c", "touch escaped.txt"]},
        {"command": "pytest\n", "arguments": []},
    ):
        with pytest.raises(ValueError, match="unsafe_terminal_request"):
            asyncio.run(registry.execute(session, "terminal_run", arguments))
    assert not (tmp_path / "escaped.txt").exists()


def test_secret_redaction_covers_headers_credentials_and_environment(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("QA_PROVIDER_API_KEY", "qa-secret-value")
    value = "Authorization: Bearer qa-secret-value https://user:pass@example.test/x?api_key=other-secret"
    redacted = redact_text(value)
    assert "qa-secret-value" not in redacted
    assert "user:pass@" not in redacted
    assert "api_key=other-secret" not in redacted
    assert "[REDACTED]" in redacted
