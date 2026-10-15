"""Typed external embedding contract."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from ..capabilities import Capability
from ..llm.base import HealthResult, ModelInfo


@dataclass(frozen=True)
class EmbeddingResult:
    vectors: tuple[tuple[float, ...], ...]
    model: str
    input_tokens: int | None = None
    request_id: str | None = None
    estimated_cost: float | None = None
    currency: str | None = None
    retry_count: int = 0


class EmbeddingProvider(Protocol):
    name: str
    model_name: str
    capabilities: frozenset[Capability]

    async def embed_batch(self, *, model: str, texts: Sequence[str]) -> EmbeddingResult: ...
    async def health_check(self) -> HealthResult: ...
    async def model_info(self, model: str) -> ModelInfo: ...
    async def close(self) -> None: ...
