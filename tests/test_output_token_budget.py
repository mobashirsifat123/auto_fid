"""Output-token budget behavior: unlimited by default (parameter omitted),
per-model caps honored, truncation self-heal, and the Anthropic explicit-cap
fallback. Regression for the GLM-5.3 failure where hidden reasoning tokens
consumed a 2,048 default and truncated structured JSON mid-payload.
"""

from __future__ import annotations

from typing import Any

import pytest

from autoresearch.providers.capabilities import Capability, ModelRole
from autoresearch.providers.config import ModelAssignment, ProviderConfig
from autoresearch.providers.llm.anthropic import AnthropicProvider
from autoresearch.providers.llm.base import CompletionResult
from autoresearch.providers.llm.gemini import GeminiProvider
from autoresearch.providers.llm.openai_compatible import OpenAICompatibleProvider
from autoresearch.providers.registry import ProviderRegistry
from autoresearch.providers.router import ModelRouter


def _config(name: str = "p", **extra: Any) -> ProviderConfig:
    base = dict(kind="openai_compatible", name=name, base_url="https://api.example.com/v1", api_key_env="K", capabilities=frozenset(Capability))
    base.update(extra)
    return ProviderConfig(**base)


class _Capture:
    def __init__(self, results: list[CompletionResult], shape: str = "openai"):
        self.results = list(results)
        self.shape = shape
        self.calls: list[dict[str, Any]] = []

    async def __call__(self, method: str, path: str, *, headers: dict[str, Any], json_body: dict[str, Any] | None = None):
        self.calls.append(json_body or {})
        result = self.results.pop(0)
        finish = result.raw.get("finish_reason")
        if self.shape == "gemini":
            payload: dict[str, Any] = {"candidates": [{"content": {"parts": [{"text": result.text}]}}], "usageMetadata": {}}
            if finish:
                payload["candidates"][0]["finishReason"] = finish
        elif self.shape == "anthropic":
            payload = {"content": [{"type": "text", "text": result.text}], "usage": {}}
            if finish == "length":
                payload["stop_reason"] = "max_tokens"
        else:
            payload = {"id": "r", "model": "m", "choices": [{"message": {"content": result.text}, "finish_reason": finish}], "usage": {}}
        return payload, {}


def _result(text: str = "{}", finish: str | None = None) -> CompletionResult:
    return CompletionResult(text=text, model="m", raw={"finish_reason": finish})


@pytest.mark.anyio
async def test_openai_compatible_omits_cap_when_unlimited(monkeypatch):
    monkeypatch.setenv("K", "test-key")
    provider = OpenAICompatibleProvider(_config())
    capture = _Capture([_result(), _result()])
    monkeypatch.setattr(provider, "request_json", capture)
    await provider.complete(model="m", messages=[{"role": "user", "content": "x"}], max_tokens=None)
    assert "max_tokens" not in capture.calls[0], "unlimited must omit the parameter"
    await provider.complete(model="m", messages=[{"role": "user", "content": "x"}], max_tokens=123)
    assert capture.calls[1]["max_tokens"] == 123


@pytest.mark.anyio
async def test_gemini_omits_cap_when_unlimited(monkeypatch):
    monkeypatch.setenv("K", "test-key")
    provider = GeminiProvider(_config(kind="gemini"))
    capture = _Capture([_result(), _result()], shape="gemini")
    monkeypatch.setattr(provider, "request_json", capture)
    await provider.complete(model="m", messages=[{"role": "user", "content": "x"}], max_tokens=None)
    assert "maxOutputTokens" not in capture.calls[0]["generationConfig"]
    await provider.complete(model="m", messages=[{"role": "user", "content": "x"}], max_tokens=77)
    assert capture.calls[1]["generationConfig"]["maxOutputTokens"] == 77


@pytest.mark.anyio
async def test_anthropic_maps_unlimited_to_explicit_cap(monkeypatch):
    monkeypatch.setenv("K", "test-key")
    provider = AnthropicProvider(_config(kind="anthropic"))
    capture = _Capture([_result(), _result()], shape="anthropic")
    monkeypatch.setattr(provider, "request_json", capture)
    await provider.complete(model="m", messages=[{"role": "user", "content": "x"}], max_tokens=None)
    assert capture.calls[0]["max_tokens"] == 8192, "Anthropic requires an explicit cap"
    await provider.complete(model="m", messages=[{"role": "user", "content": "x"}], max_tokens=55)
    assert capture.calls[1]["max_tokens"] == 55


class _FakeProvider:
    capabilities = frozenset(Capability)

    def __init__(self, results: list[CompletionResult]):
        self.results = list(results)
        self.calls: list[int | None] = []

    async def complete(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int | None = None, response_format: dict[str, Any] | None = None, tools: list[dict[str, Any]] | None = None) -> CompletionResult:
        self.calls.append(max_tokens)
        return self.results.pop(0)

    async def stream(self, **_: Any):  # pragma: no cover
        raise NotImplementedError

    async def close(self) -> None:
        return None


def _router(provider: _FakeProvider, cap: int | None) -> ModelRouter:
    registry = ProviderRegistry()
    registry.register_llm("fake", provider)
    return ModelRouter(registry, {ModelRole.RESEARCH_REASONING: ModelAssignment(ModelRole.RESEARCH_REASONING, "fake", "m", (), None, cap)})


@pytest.mark.anyio
async def test_router_defaults_to_unlimited():
    provider = _FakeProvider([_result('{"a": 1}')])
    router = _router(provider, cap=None)
    value = await router.structured_output(ModelRole.RESEARCH_REASONING, messages=[{"role": "user", "content": "x"}], schema={"name": "s", "schema": {"type": "object"}})
    assert value == {"a": 1}
    assert provider.calls == [None], "no config cap → parameter omitted (model max)"


@pytest.mark.anyio
async def test_router_self_heals_truncation_by_doubling_cap():
    provider = _FakeProvider([
        _result('{"a":', finish="length"),
        _result('{"a": 1}', finish="stop"),
    ])
    router = _router(provider, cap=2048)
    value = await router.structured_output(ModelRole.RESEARCH_REASONING, messages=[{"role": "user", "content": "x"}], schema={"name": "s", "schema": {"type": "object"}})
    assert value == {"a": 1}
    assert provider.calls == [2048, 4096], "truncated cap must double on retry"


@pytest.mark.anyio
async def test_router_returns_last_result_after_retry_budget():
    provider = _FakeProvider([
        _result('{"a":', finish="length"),
        _result('{"b":', finish="length"),
        _result('{"c":', finish="length"),
    ])
    router = _router(provider, cap=100)
    result = await router.complete(ModelRole.RESEARCH_REASONING, messages=[{"role": "user", "content": "x"}])
    assert provider.calls == [100, 200, 400]
    assert result.raw["finish_reason"] == "length", "exhausted retries return the honest last result"
