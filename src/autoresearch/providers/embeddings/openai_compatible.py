"""External OpenAI-compatible embedding adapter."""

from __future__ import annotations

import asyncio
import math
import os
from collections.abc import Sequence
from typing import Any

import httpx

from ..config import ProviderConfig
from ..errors import ErrorKind, ProviderError, kind_for_status
from ..llm.base import HealthResult, ModelInfo
from ..llm.http_base import AsyncRateLimiter
from ..retry import with_retry
from ..usage import estimate_cost
from .base import EmbeddingResult


class ExternalEmbeddingProvider:
    def __init__(self, config: ProviderConfig, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        if not config.embedding_dimension:
            raise ValueError("External embedding providers require embedding_dimension.")
        self.config = config
        self.name = config.name
        self.model_name = ""
        self.capabilities = config.capabilities
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._limiter = AsyncRateLimiter(config.requests_per_second)
        self._client = httpx.AsyncClient(base_url=config.base_url.rstrip("/"), timeout=httpx.Timeout(config.total_timeout, connect=config.connect_timeout, read=config.read_timeout), transport=transport)

    def _headers(self) -> dict[str, str]:
        key = os.getenv(self.config.api_key_env, "")
        if not key:
            raise ProviderError(f"Required environment variable '{self.config.api_key_env}' is not set.", ErrorKind.AUTHENTICATION, self.name)
        return {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    async def embed_batch(self, *, model: str, texts: Sequence[str]) -> EmbeddingResult:
        if not texts:
            return EmbeddingResult((), model, 0)
        vectors: list[tuple[float, ...]] = []
        token_total = 0
        request_id: str | None = None
        retry_total = 0
        for start in range(0, len(texts), self.config.embedding_batch_size):
            batch = list(texts[start:start + self.config.embedding_batch_size])
            async def operation() -> tuple[dict[str, Any], httpx.Headers]:
                await self._limiter.acquire()
                async with self._semaphore:
                    try:
                        response = await self._client.post("/embeddings", headers=self._headers(), json={"model": model, "input": batch})
                    except httpx.TimeoutException as exc:
                        raise ProviderError("Embedding request timed out.", ErrorKind.TIMEOUT, self.name) from exc
                    except httpx.HTTPError as exc:
                        raise ProviderError("Embedding endpoint is unavailable.", ErrorKind.UNAVAILABLE, self.name) from exc
                if response.is_error:
                    raise ProviderError(f"Embedding provider returned HTTP {response.status_code}.", kind_for_status(response.status_code), self.name, response.status_code)
                return response.json(), response.headers
            def record_retry(_attempt: int, _error: ProviderError) -> None:
                nonlocal retry_total
                retry_total += 1
            payload, headers = await with_retry(operation, attempts=self.config.max_retries + 1, on_retry=record_retry)
            rows = sorted(payload.get("data") or [], key=lambda item: int(item.get("index", 0)))
            if len(rows) != len(batch):
                raise ProviderError("Embedding response count does not match input count.", ErrorKind.SCHEMA, self.name)
            for row in rows:
                vector = tuple(float(value) for value in row.get("embedding") or [])
                if len(vector) != self.config.embedding_dimension or not all(math.isfinite(value) for value in vector):
                    raise ProviderError("Embedding vector has an invalid dimension or non-finite value.", ErrorKind.SCHEMA, self.name)
                vectors.append(vector)
            token_total += int((payload.get("usage") or {}).get("prompt_tokens") or 0)
            request_id = headers.get("x-request-id") or request_id
        cost, currency = estimate_cost(token_total, 0, self.config.pricing)
        self.model_name = model
        return EmbeddingResult(tuple(vectors), model, token_total, request_id, cost, currency, retry_total)

    async def health_check(self) -> HealthResult:
        try:
            await self.embed_batch(model=self.model_name or "health-check", texts=["health"])
        except ProviderError as exc:
            return HealthResult(False, exc.kind.value, str(exc))
        return HealthResult(True, "ready", "External embedding endpoint is reachable.")

    async def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(self.name, model, self.capabilities)

    async def close(self) -> None:
        await self._client.aclose()
