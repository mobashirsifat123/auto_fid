"""CLI-level functional tests for the five-skill distribution.

Every command of every skill is exercised offline through Typer's CliRunner:
help surfaces, usage validation, clean JSON error paths (no tracebacks), and
the honest not-found / credential-refusal behavior students will hit first.

No network, no paid API: provider-backed commands are expected to fail with
their documented machine-readable error, never a crash.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from autoresearch.cli.app import app

runner = CliRunner()


def _invoke(*args: str):
    return runner.invoke(app, list(args))


def _payload(result):
    return json.loads(result.output)


ALL_COMMANDS = [
    ("skill",),
    ("skill", "list"),
    ("skill", "install"),
    ("skill", "update"),
    ("skill", "uninstall"),
    ("skill", "doctor"),
    ("skill", "review"),
    ("skill", "novelty"),
    ("skill", "workflow"),
    ("skill", "manuscript"),
    ("skill", "experiment",),
    ("skill", "experiment", "plan"),
    ("skill", "experiment", "validate"),
    ("skill", "experiment", "implement"),
    ("skill", "experiment", "approve"),
    ("skill", "experiment", "execute"),
    ("skill", "experiment", "inspect"),
    ("skill", "experiment", "report"),
    ("skill", "experiment", "cancel"),
]


@pytest.mark.parametrize("command", ALL_COMMANDS, ids=[" ".join(c) for c in ALL_COMMANDS])
def test_help_works_for_every_command(command):
    result = _invoke(*command, "--help")
    assert result.exit_code == 0, result.output
    assert "Usage" in result.output


def test_root_help_lists_the_five_skills():
    result = _invoke("--help")
    assert result.exit_code == 0
    for name in ("review", "novelty", "experiment", "manuscript", "workflow"):
        assert name in result.output


def test_skill_list_reports_exactly_five_skills():
    result = _invoke("skill", "list")
    assert result.exit_code == 0
    payload = _payload(result)
    names = {skill["name"] for skill in payload["skills"]}
    assert names == {"review", "novelty", "experiment", "manuscript", "workflow"}


def test_unknown_experiment_subcommand_is_clean_usage_error():
    result = _invoke("skill", "experiment", "does-not-exist")
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_missing_required_option_is_usage_error_not_crash():
    # manuscript requires --project
    result = _invoke("skill", "manuscript")
    assert result.exit_code == 2
    assert "Traceback" not in result.output


@pytest.mark.parametrize("args", [
    ("skill", "experiment", "plan", "--candidate-id", "missing"),
    ("skill", "experiment", "report", "--candidate-id", "missing"),
    ("skill", "novelty", "--project", "missing"),
    ("skill", "manuscript", "--project", "missing"),
])
def test_unknown_ids_produce_clean_json_errors(tmp_path, monkeypatch, args):
    """Student-facing data errors must be machine-readable JSON with the
    documented exit codes — never a Python traceback."""
    monkeypatch.setenv("AUTORESEARCH_DATA_ROOT", str(tmp_path / "data"))
    result = _invoke(*args)
    assert result.exit_code != 0
    assert "Traceback" not in result.output
    try:
        payload = _payload(result)
        assert payload.get("ok") is False
        assert payload.get("error", {}).get("code")
    except json.JSONDecodeError:
        pytest.fail(f"non-JSON error output:\n{result.output[:400]}")


def test_workflow_requires_seed_or_project(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_DATA_ROOT", str(tmp_path / "data"))
    result = _invoke("skill", "workflow", "--until", "review")
    assert result.exit_code == 2  # usage: one of --seed-paper/--from-catalog/--project required
    assert "Traceback" not in result.output


def test_workflow_rejects_unknown_stop_stage(tmp_path, monkeypatch):
    monkeypatch.setenv("AUTORESEARCH_DATA_ROOT", str(tmp_path / "data"))
    result = _invoke("skill", "workflow", "--seed-paper", "x.pdf", "--until", "pubish")
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_skill_doctor_reports_without_config(tmp_path, monkeypatch):
    """A fresh machine with no provider config gets a clean diagnosis:
    machine-readable JSON naming exactly what is missing, degraded exit 1,
    never a traceback."""
    monkeypatch.setenv("AUTORESEARCH_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("AUTORESEARCH_PROVIDER_CONFIG", str(tmp_path / "none.toml"))
    # Isolate from the developer's real OpenCode session store: with HOME
    # pointing at the temp dir there is no session credential to auto-detect.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data-home"))
    monkeypatch.delenv("OPENCODE_DB", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    result = _invoke("skill", "doctor")
    assert result.exit_code == 1  # degraded state is a signal, not a crash
    assert "Traceback" not in result.output
    payload = _payload(result)
    assert payload["overall"] == "degraded"
    checks = {check["check"]: check for check in payload["checks"]}
    assert "providers" in checks
    assert checks["providers"]["ok"] is False
    assert checks["providers"]["detail"]


def test_install_targets_are_validated():
    result = _invoke("skill", "install", "--target", "not-a-target")
    assert result.exit_code != 0
    assert "Traceback" not in result.output


def test_anthropic_unlimited_calls_use_model_maximum_not_8192(monkeypatch):
    """Regression (owner-reported): the Anthropic protocol requires an explicit
    output cap; the old default of 8192 was eaten by glm-5.3's hidden reasoning
    and silently truncated every structured output — no skill could finish.
    Unlimited call sites must now send the model-family maximum default, and a
    rejected cap steps down to the largest accepted value."""
    import asyncio
    import json as _json

    import httpx

    monkeypatch.setenv("KEY_ENV", "hermetic-test-key")  # never rely on ambient credentials

    from autoresearch.providers.capabilities import Capability
    from autoresearch.providers.config import ProviderConfig
    from autoresearch.providers.llm.anthropic import AnthropicProvider

    config = ProviderConfig("reasoning", "anthropic", "https://api.z.ai/api/anthropic", "KEY_ENV", frozenset({Capability.TEXT}))
    sent: list[dict] = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = _json.loads(request.content.decode())
        sent.append(body)
        if body["max_tokens"] > 65536:  # endpoint rejects caps above its model maximum
            return httpx.Response(400, json={"error": {"message": "max_tokens: maximum 65536 is allowed for this model"}})
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "ok"}],
            "stop_reason": "end_turn", "model": "glm-5.3", "usage": {},
        })

    provider = AnthropicProvider(config, transport=httpx.MockTransport(respond))

    async def run():
        try:
            result = await provider.complete(model="glm-5.3", messages=[{"role": "user", "content": "hi"}])
            return result
        finally:
            await provider.close()

    result = asyncio.run(run())
    assert result.text == "ok"
    assert sent, "no request captured"
    assert sent[0]["max_tokens"] == 65536  # unlimited → model-family maximum, never 8192


def test_zero_timeouts_yield_finite_deadlines():
    """Zero read/total timeouts must fall back to generous FINITE deadlines —
    a wedged connection may cost an hour, never a forever-hang (regression:
    two live runs stalled ~1h on dead sockets with infinite deadlines)."""
    import httpx

    from autoresearch.providers.config import ProviderConfig
    from autoresearch.providers.llm.http_base import HTTPProviderBase

    config = ProviderConfig("reasoning", "openai_compatible", "https://api.example.com/v1", "KEY_ENV", frozenset())
    provider = HTTPProviderBase.__new__(HTTPProviderBase)
    provider.config = config
    provider.client = httpx.AsyncClient(
        timeout=httpx.Timeout(
            config.total_timeout or config.default_total_timeout,
            connect=config.connect_timeout,
            read=config.read_timeout or config.default_read_timeout,
        )
    )
    try:
        assert provider.client.timeout.read == 3600.0
        assert provider.client.timeout.connect == 5.0
    finally:
        import asyncio

        asyncio.get_event_loop_policy().new_event_loop().run_until_complete(provider.client.aclose())


def test_truncated_structured_output_self_heals_by_doubling_cap():
    """A structured call cut off by the output cap must self-heal — the cap is
    doubled and the call retried — never die as a non-retryable schema error
    (regression: a live novelty run lost its last surviving candidate when a
    truncated forced-tool response raised SCHEMA, killing the whole run)."""
    import asyncio

    from autoresearch.providers.capabilities import ModelRole
    from autoresearch.providers.config import ModelAssignment
    from autoresearch.providers.llm.base import CompletionResult
    from autoresearch.providers.router import ModelRouter

    calls: list[int | None] = []

    class TruncatingProvider:
        name = "reasoning"

        async def complete(self, *, model, messages, temperature=0.2, max_tokens=None, response_format=None, tools=None):
            calls.append(max_tokens)
            if len(calls) == 1:
                # Adapter-imposed fallback cap, reported back to the router.
                return CompletionResult(text="", model=model, raw={"finish_reason": "length", "effective_max_tokens": 8192})
            return CompletionResult(text='{"ok": true}', model=model, raw={"finish_reason": "end_turn", "effective_max_tokens": max_tokens})

    class Registry:
        def __init__(self, provider):
            self._provider = provider

        def validate_assignment(self, assignment):
            return None

        def get(self, name):
            return self._provider

    assignment = ModelAssignment(ModelRole.RESEARCH_REASONING, "reasoning", "glm-5.3")
    router = ModelRouter(Registry(TruncatingProvider()), {ModelRole.RESEARCH_REASONING: assignment})
    result = asyncio.get_event_loop_policy().new_event_loop().run_until_complete(
        router.complete("research_reasoning", messages=[{"role": "user", "content": "x"}])
    )
    # Unconfigured cap -> the adapter's reported 8192 is doubled for the retry.
    assert calls == [None, 16384]
    assert result.text == '{"ok": true}'


def test_genuine_tool_omission_still_fails_closed_with_reason(monkeypatch):
    """A response that ends cleanly without the forced tool call is real model
    misbehavior: it must stay a schema error — now carrying stop_reason — and
    only cap-truncation takes the self-heal path."""
    import asyncio
    from contextvars import ContextVar

    from autoresearch.providers.config import ProviderConfig
    from autoresearch.providers.errors import ErrorKind, ProviderError
    from autoresearch.providers.llm.anthropic import AnthropicProvider

    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.config = ProviderConfig("reasoning", "anthropic", "https://api.example.com", "TEST_KEY_ENV", frozenset())
    provider.name = "reasoning"
    provider._retry_count = ContextVar("test_retry_count", default=0)
    monkeypatch.setenv("TEST_KEY_ENV", "test-key")

    async def respond(payload):
        async def fake_request_json(method, path, headers=None, json_body=None):
            return payload, {}
        provider.request_json = fake_request_json
        return await provider.complete(model="glm-5.3", messages=[{"role": "user", "content": "x"}], response_format={"json_schema": {"name": "out", "schema": {"type": "object"}}})

    loop = asyncio.get_event_loop_policy().new_event_loop()
    try:
        # Clean end_turn without the tool call: fail closed, with the reason.
        try:
            loop.run_until_complete(respond({"content": [{"type": "text", "text": "no tool"}], "stop_reason": "end_turn"}))
            raise AssertionError("expected ProviderError")
        except ProviderError as exc:
            assert exc.kind is ErrorKind.SCHEMA
            assert exc.details == {"stop_reason": "end_turn"}

        # Truncated forced-tool call: a self-healable result, not an error.
        result = loop.run_until_complete(respond({"content": [{"type": "text", "text": "partial"}], "stop_reason": "max_tokens"}))
        assert result.text == ""
        assert result.raw["finish_reason"] == "length"
        assert result.raw["effective_max_tokens"] == 65536
    finally:
        loop.close()
