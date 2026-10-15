"""Deterministic LLM test double; never selectable by production configuration."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Callable
from typing import Any

from ..capabilities import Capability
from ..llm.base import CompletionResult, HealthResult, ModelInfo

Responder = Callable[[list[dict[str, Any]]], dict[str, Any] | str]


class FakeLLMProvider:
    name = "fake_llm"
    capabilities = frozenset(Capability)

    def __init__(self, responder: Responder | None = None, responses: list[dict[str, Any] | str] | None = None) -> None:
        self.responder = responder
        self.responses = list(responses or [])
        self.calls: list[dict[str, Any]] = []

    async def complete(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int = 2048, response_format: dict[str, Any] | None = None, tools: list[dict[str, Any]] | None = None) -> CompletionResult:
        self.calls.append({"model": model, "messages": messages, "response_format": response_format, "tools": tools})
        if self.responder:
            value = self.responder(messages)
        elif self.responses:
            value = self.responses.pop(0)
        else:
            value = {"ok": True}
        text = value if isinstance(value, str) else json.dumps(value)
        prompt_tokens = max(1, sum(len(str(item.get("content", ""))) for item in messages) // 4)
        return CompletionResult(text, model, prompt_tokens, max(1, len(text) // 4), f"fake-{len(self.calls)}", 0.0, "USD", {"tool_calls": []})

    def stream(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int = 2048) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            result = await self.complete(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            yield result.text
        return iterator()

    async def health_check(self) -> HealthResult:
        return HealthResult(True, "ready", "Deterministic fake provider.", 0.0, ("fake-model",))

    async def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(self.name, model, self.capabilities)

    async def close(self) -> None:
        return None
