"""Pooled HTTP foundation with timeouts, concurrency, rate limiting, and retry."""

from __future__ import annotations

import asyncio
from contextvars import ContextVar
import os
import time
from typing import Any

import httpx

from ..capabilities import Capability
from ..config import ProviderConfig
from ..errors import ErrorKind, ProviderError, kind_for_status
from ..retry import with_retry
from ..usage import estimate_cost
from .base import CompletionResult, HealthResult, ModelInfo


class AsyncRateLimiter:
    def __init__(self, requests_per_second: float | None) -> None:
        self.interval = 1.0 / requests_per_second if requests_per_second else 0.0
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def acquire(self) -> None:
        if not self.interval:
            return
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self.interval


class HTTPProviderBase:
    def __init__(self, config: ProviderConfig, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self.name = config.name
        self.capabilities = config.capabilities
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._limiter = AsyncRateLimiter(config.requests_per_second)
        timeout = httpx.Timeout(config.total_timeout, connect=config.connect_timeout, read=config.read_timeout)
        self.client = httpx.AsyncClient(base_url=config.base_url.rstrip("/"), timeout=timeout, transport=transport)
        self._retry_count: ContextVar[int] = ContextVar(f"autoresearch_retry_count_{config.name}", default=0)

    def api_key(self) -> str:
        value = os.getenv(self.config.api_key_env, "")
        if not value:
            raise ProviderError(f"Required environment variable '{self.config.api_key_env}' is not set.", ErrorKind.AUTHENTICATION, self.name)
        return value

    async def request_json(self, method: str, path: str, *, headers: dict[str, str], json_body: dict[str, Any] | None = None) -> tuple[dict[str, Any], httpx.Headers]:
        async def operation() -> tuple[dict[str, Any], httpx.Headers]:
            await self._limiter.acquire()
            async with self._semaphore:
                try:
                    response = await self.client.request(method, path, headers=headers, json=json_body)
                except httpx.TimeoutException as exc:
                    raise ProviderError("Provider request timed out.", ErrorKind.TIMEOUT, self.name) from exc
                except httpx.HTTPError as exc:
                    raise ProviderError("Provider endpoint is unavailable.", ErrorKind.UNAVAILABLE, self.name) from exc
            if response.is_error:
                retry_after = response.headers.get("retry-after")
                try:
                    delay = float(retry_after) if retry_after else None
                except ValueError:
                    delay = None
                request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
                raise ProviderError(f"Provider returned HTTP {response.status_code}.", kind_for_status(response.status_code), self.name, response.status_code, request_id, delay)
            try:
                payload = response.json()
            except ValueError as exc:
                raise ProviderError("Provider returned a non-JSON response.", ErrorKind.SCHEMA, self.name) from exc
            if not isinstance(payload, dict):
                raise ProviderError("Provider returned an unexpected JSON value.", ErrorKind.SCHEMA, self.name)
            return payload, response.headers

        retry_count = 0
        def record_retry(_attempt: int, _error: ProviderError) -> None:
            nonlocal retry_count
            retry_count += 1
        result = await with_retry(operation, attempts=self.config.max_retries + 1, on_retry=record_retry)
        self._retry_count.set(retry_count)
        return result

    def completion_result(self, *, text: str, model: str, usage: dict[str, Any], request_id: str | None, raw: dict[str, Any]) -> CompletionResult:
        prompt = _int_or_none(usage.get("prompt_tokens") or usage.get("input_tokens"))
        completion = _int_or_none(usage.get("completion_tokens") or usage.get("output_tokens"))
        cost, currency = estimate_cost(prompt, completion, self.config.pricing)
        return CompletionResult(text, model, prompt, completion, request_id, cost, currency, {**raw, "retry_count": self._retry_count.get()})

    async def health_check(self) -> HealthResult:
        started = time.perf_counter()
        try:
            await self.model_info("")
        except ProviderError as exc:
            return HealthResult(False, exc.kind.value, str(exc), (time.perf_counter() - started) * 1000)
        return HealthResult(True, "ready", "Provider endpoint is reachable.", (time.perf_counter() - started) * 1000)

    async def model_info(self, model: str) -> ModelInfo:
        return ModelInfo(self.name, model, self.capabilities)

    async def close(self) -> None:
        await self.client.aclose()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None
