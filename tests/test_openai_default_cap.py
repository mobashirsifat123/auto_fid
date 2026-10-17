"""Regression: the OpenAI-compatible path must send a finite default cap for
unlimited call sites and report the effective cap for the truncation self-heal.

Live need (2026-10-16, Alibaba token-plan switch): openai_compatible sent no
max_tokens when a call site requested unlimited, so the endpoint's implicit
default could truncate large generations (implementation scripts) with no way
for the router's self-heal to detect the applied cap.
"""

from __future__ import annotations

import asyncio
from contextvars import ContextVar

from autoresearch.providers.config import ProviderConfig
from autoresearch.providers.llm.openai_compatible import OpenAICompatibleProvider


def _provider(bodies: list[dict]) -> OpenAICompatibleProvider:
    provider = OpenAICompatibleProvider.__new__(OpenAICompatibleProvider)
    provider.config = ProviderConfig("reasoning", "openai_compatible", "https://api.example.com/v1", "TEST_KEY_ENV", frozenset())
    provider.name = "reasoning"
    provider._retry_count = ContextVar("test_retry_count", default=0)

    async def fake_request_json(method, path, headers=None, json_body=None):
        bodies.append(json_body or {})
        return {"choices": [{"message": {"content": "hello"}, "finish_reason": "stop"}]}, {}

    provider.request_json = fake_request_json
    return provider


def test_unlimited_gets_default_cap_and_reports_effective(monkeypatch):
    monkeypatch.setenv("TEST_KEY_ENV", "test-key")
    bodies: list[dict] = []
    provider = _provider(bodies)
    result = asyncio.run(provider.complete(model="deepseek-v4.1-flash", messages=[{"role": "user", "content": "x"}]))
    assert bodies[0]["max_tokens"] == 65536
    assert result.raw["effective_max_tokens"] == 65536


def test_explicit_cap_is_respected(monkeypatch):
    monkeypatch.setenv("TEST_KEY_ENV", "test-key")
    bodies: list[dict] = []
    provider = _provider(bodies)
    result = asyncio.run(provider.complete(model="deepseek-v4.1-flash", messages=[{"role": "user", "content": "x"}], max_tokens=1234))
    assert bodies[0]["max_tokens"] == 1234
    assert result.raw["effective_max_tokens"] == 1234