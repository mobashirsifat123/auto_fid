"""Pooled HTTP foundation with timeouts, concurrency, rate limiting, and retry."""

from __future__ import annotations

import asyncio
import time
from contextvars import ContextVar
from typing import Any

import httpx

from ..capabilities import Capability
from ..config import ProviderConfig
from ..credentials import resolve_api_key
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


class AdaptiveGate:
    """AIMD concurrency gate with a shared rate-limit cooldown.

    Every mode applies the global cooldown: when any request is throttled
    with HTTP 429, all in-flight slots pause for the Retry-After window
    (bounded, defaulting to 1s when the header is absent), so synchronized
    retries cannot amplify into a 429 storm. Adaptive mode (ceiling >
    concurrency) additionally probes the endpoint's real limit: the limit
    grows by one per full window of successes and halves on throttle, never
    dropping below one or exceeding the ceiling. Fixed mode (no ceiling)
    preserves plain semaphore semantics on top of the cooldown.
    """

    def __init__(self, *, concurrency: int, ceiling: int | None = None) -> None:
        self._floor = max(1, concurrency)
        self._ceiling = max(self._floor, ceiling if ceiling is not None else self._floor)
        self._limit = float(self._floor)
        self._inflight = 0
        self._successes = 0
        self._cooldown_until = 0.0
        self._cond = asyncio.Condition()

    @property
    def limit(self) -> int:
        return int(self._limit)

    async def acquire(self) -> None:
        async with self._cond:
            while True:
                now = time.monotonic()
                if self._cooldown_until > now:
                    # Wake on cooldown expiry even if nothing releases meanwhile.
                    try:
                        await asyncio.wait_for(self._cond.wait(), timeout=self._cooldown_until - now)
                    except TimeoutError:
                        pass
                    continue
                if self._inflight < self._limit:
                    self._inflight += 1
                    return
                await self._cond.wait()

    async def release(self, *, throttled: bool = False, retry_after: float | None = None) -> None:
        async with self._cond:
            self._inflight = max(0, self._inflight - 1)
            if throttled:
                if self._ceiling > self._floor:
                    self._limit = max(1.0, self._limit / 2.0)
                self._successes = 0
                delay = retry_after if retry_after is not None else 1.0
                self._cooldown_until = max(self._cooldown_until, time.monotonic() + min(max(delay, 0.0), 30.0))
            else:
                self._successes += 1
                if self._ceiling > self._floor and self._successes >= self._limit:
                    self._limit = min(float(self._ceiling), self._limit + 1.0)
                    self._successes = 0
            self._cond.notify_all()


class HTTPProviderBase:
    def __init__(self, config: ProviderConfig, *, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.config = config
        self.name = config.name
        self.capabilities = config.capabilities
        self._semaphore = asyncio.Semaphore(config.concurrency)
        self._gate = AdaptiveGate(concurrency=config.concurrency, ceiling=config.max_concurrency)
        self._limiter = AsyncRateLimiter(config.requests_per_second)
        read_deadline = config.read_timeout or config.default_read_timeout
        total_deadline = config.total_timeout or config.default_total_timeout
        timeout = httpx.Timeout(total_deadline, connect=config.connect_timeout, read=read_deadline)
        self.client = httpx.AsyncClient(base_url=config.base_url.rstrip("/"), timeout=timeout, transport=transport)
        self._retry_count: ContextVar[int] = ContextVar(f"autoresearch_retry_count_{config.name}", default=0)

    def api_key(self) -> str:
        return resolve_api_key(self.config)

    async def request_json(self, method: str, path: str, *, headers: dict[str, str], json_body: dict[str, Any] | None = None) -> tuple[dict[str, Any], httpx.Headers]:
        async def operation() -> tuple[dict[str, Any], httpx.Headers]:
            await self._limiter.acquire()
            await self._gate.acquire()
            throttled = False
            throttle_delay: float | None = None
            try:
                try:
                    response = await self.client.request(method, path, headers=headers, json=json_body)
                except httpx.TimeoutException as exc:
                    raise ProviderError("Provider request timed out.", ErrorKind.TIMEOUT, self.name) from exc
                except httpx.HTTPError as exc:
                    raise ProviderError("Provider endpoint is unavailable.", ErrorKind.UNAVAILABLE, self.name) from exc
                if response.is_error:
                    throttled = response.status_code == 429
                    retry_after_header = response.headers.get("retry-after")
                    try:
                        delay = float(retry_after_header) if retry_after_header else None
                    except ValueError:
                        delay = None
                    throttle_delay = delay
                    request_id = response.headers.get("x-request-id") or response.headers.get("request-id")
                    detail = ""
                    if response.status_code < 500:
                        # 4xx bodies explain the rejection (missing mode,
                        # bad model name, expired key…) — surface a bounded,
                        # sanitized snippet so failures are self-diagnosing.
                        body = (response.text or "").strip()[:300]
                        if body:
                            detail = f" detail: {body}"
                    raise ProviderError(
                        f"Provider returned HTTP {response.status_code}.{detail}",
                        kind_for_status(response.status_code), self.name, response.status_code, request_id, delay,
                    )
                try:
                    payload = response.json()
                except ValueError as exc:
                    raise ProviderError("Provider returned a non-JSON response.", ErrorKind.SCHEMA, self.name) from exc
                if not isinstance(payload, dict):
                    raise ProviderError("Provider returned an unexpected JSON value.", ErrorKind.SCHEMA, self.name)
                return payload, response.headers
            finally:
                await self._gate.release(throttled=throttled, retry_after=throttle_delay)

        retry_count = 0
        def record_retry(_attempt: int, _error: ProviderError) -> None:
            nonlocal retry_count
            retry_count += 1
        try:
            # httpx timeouts apply to individual I/O phases, not total elapsed
            # time. Include queueing, rate limits, retries, and backoff here.
            async with asyncio.timeout(self.config.total_timeout or self.config.default_total_timeout):
                result = await with_retry(operation, attempts=self.config.max_retries + 1, on_retry=record_retry)
        except TimeoutError as exc:
            raise ProviderError("Provider total request deadline exceeded (including retries and queueing).", ErrorKind.TIMEOUT, self.name) from exc
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
            self.api_key()
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
