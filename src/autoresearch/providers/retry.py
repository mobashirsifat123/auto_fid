"""Bounded asynchronous retry with exponential backoff and jitter."""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable
from typing import TypeVar

from .errors import ProviderError

T = TypeVar("T")


async def with_retry(
    operation: Callable[[], Awaitable[T]],
    *,
    attempts: int,
    base_delay: float = 0.25,
    max_delay: float = 8.0,
    jitter: Callable[[], float] = random.random,
    on_retry: Callable[[int, ProviderError], None] | None = None,
) -> T:
    last: ProviderError | None = None
    for attempt in range(max(1, attempts)):
        try:
            return await operation()
        except ProviderError as exc:
            last = exc
            if not exc.retryable or attempt == attempts - 1:
                raise
            if on_retry:
                on_retry(attempt, exc)
            delay = exc.retry_after if exc.retry_after is not None else min(max_delay, base_delay * (2**attempt))
            await asyncio.sleep(max(0.0, delay) + min(0.25, jitter() * base_delay))
    assert last is not None
    raise last
