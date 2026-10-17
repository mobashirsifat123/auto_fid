"""Regression: a cap exhausted by hidden reasoning must self-heal, not error.

Live bug (2026-10-16, FIRE-Bench implementation stage): glm-5.3 via the z.ai
anthropic endpoint spends hidden thinking tokens against max_tokens. When the
whole 8192-token fallback cap was consumed before any visible block, the
free-form path raised a hard SCHEMA error ("no text or tool call") — so the
router's truncation self-heal never ran and the implementation call failed
twice with no retry. The forced-tool path already reported truncation for
this case; the free-form path now mirrors it.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.config import ModelAssignment, ProviderConfig
from autoresearch.providers.errors import ErrorKind, ProviderError
from autoresearch.providers.llm.anthropic import AnthropicProvider
from autoresearch.providers.router import ModelRouter


def _provider(payloads: list[dict], caps: list[int | None]) -> AnthropicProvider:
    provider = AnthropicProvider.__new__(AnthropicProvider)
    provider.config = ProviderConfig("reasoning", "anthropic", "https://api.example.com", "TEST_KEY_ENV", frozenset())
    provider.name = "reasoning"
    provider._retry_count = ContextVar("test_retry_count", default=0)
    queue = list(payloads)

    async def fake_request_json(method, path, headers=None, json_body=None):
        caps.append((json_body or {}).get("max_tokens"))
        return queue.pop(0), {}

    provider.request_json = fake_request_json
    return provider


def test_thinking_only_max_tokens_returns_self_healable_truncation(monkeypatch):
    monkeypatch.setenv("TEST_KEY_ENV", "test-key")
    provider = _provider(
        [{"content": [{"type": "thinking", "thinking": "..."}], "stop_reason": "max_tokens"}],
        [],
    )
    result = asyncio.run(provider.complete(model="glm-5.3", messages=[{"role": "user", "content": "x"}]))
    assert result.text == ""
    assert result.raw["finish_reason"] == "length"
    assert result.raw["effective_max_tokens"] == 65536


def test_empty_end_turn_still_fails_closed(monkeypatch):
    monkeypatch.setenv("TEST_KEY_ENV", "test-key")
    provider = _provider([{"content": [], "stop_reason": "end_turn"}], [])
    try:
        asyncio.run(provider.complete(model="glm-5.3", messages=[{"role": "user", "content": "x"}]))
        raise AssertionError("expected ProviderError")
    except ProviderError as exc:
        assert exc.kind is ErrorKind.SCHEMA
        assert exc.details == {"stop_reason": "end_turn"}


def test_router_doubles_cap_end_to_end_for_thinking_exhaustion(monkeypatch):
    monkeypatch.setenv("TEST_KEY_ENV", "test-key")
    caps: list[int | None] = []
    provider = _provider(
        [
            {"content": [{"type": "thinking", "thinking": "..."}], "stop_reason": "max_tokens"},
            {"content": [{"type": "text", "text": "print('hi')"}], "stop_reason": "end_turn"},
        ],
        caps,
    )

    class Registry:
        def validate_assignment(self, assignment):
            return None

        def get(self, name):
            return provider

    assignment = ModelAssignment(ModelRole.RESEARCH_REASONING, "reasoning", "glm-5.3")
    router = ModelRouter(Registry(), {ModelRole.RESEARCH_REASONING: assignment})
    result = asyncio.run(router.complete("research_reasoning", messages=[{"role": "user", "content": "x"}]))
    assert caps == [65536, 131072]
    assert result.text == "print('hi')"