"""Deterministic embedding test double; never used by production factories."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Sequence

from ..capabilities import Capability
from ..embeddings.base import EmbeddingResult
from ..llm.base import HealthResult, ModelInfo


class FakeEmbeddingProvider:
    name = "fake_embeddings"
    capabilities = frozenset({Capability.EMBEDDINGS})

    def __init__(self, dimensions: int = 24) -> None:
        self.dimensions = dimensions
        self.model_name = "fake-embedding-v1"

    def _vector(self, text: str) -> tuple[float, ...]:
        tokens = text.lower().split()
        values = [0.0] * self.dimensions
        for token in tokens:
            digest = hashlib.sha256(token.encode()).digest()
            values[int.from_bytes(digest[:2], "big") % self.dimensions] += 1.0 if digest[2] % 2 else -1.0
        norm = math.sqrt(sum(value * value for value in values)) or 1.0
        return tuple(value / norm for value in values)

    async def embed_batch(self, *, model: str, texts: Sequence[str]) -> EmbeddingResult:
        return EmbeddingResult(tuple(self._vector(text) for text in texts), model, sum(max(1, len(text) // 4) for text in texts), f"fake-embed-{len(texts)}", 0.0, "USD")

    async def health_check(self) -> HealthResult:
        return HealthResult(True, "ready", "Deterministic fake embedding provider.", 0.0, (self.model_name,))

    async def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(self.name, model, self.capabilities)

    async def close(self) -> None:
        return None
