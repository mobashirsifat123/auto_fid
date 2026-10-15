"""Typed provider-neutral LLM contracts."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from ..capabilities import Capability


@dataclass(frozen=True)
class CompletionResult:
    text: str
    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    request_id: str | None = None
    estimated_cost: float | None = None
    currency: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolCallResult:
    calls: tuple[dict[str, Any], ...]
    completion: CompletionResult


@dataclass(frozen=True)
class HealthResult:
    healthy: bool
    state: str
    detail: str
    latency_ms: float | None = None
    models: tuple[str, ...] = ()


@dataclass(frozen=True)
class ModelInfo:
    provider: str
    model: str
    capabilities: frozenset[Capability]


class LLMProvider(Protocol):
    name: str
    capabilities: frozenset[Capability]

    async def complete(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int = 2048, response_format: dict[str, Any] | None = None, tools: list[dict[str, Any]] | None = None) -> CompletionResult: ...
    def stream(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int = 2048) -> AsyncIterator[str]: ...
    async def health_check(self) -> HealthResult: ...
    async def model_info(self, model: str) -> ModelInfo: ...
    async def close(self) -> None: ...
