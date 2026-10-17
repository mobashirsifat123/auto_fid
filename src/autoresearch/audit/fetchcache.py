"""Checksummed, replayable HTTP evidence cache for trust-audit fetches.

Every scholarly-API response used by the audit harness is persisted to a
JSONL evidence file (DESIGN §9): source, cache key, request URL, status
code, a frozen subset of response headers, the raw body (base64), its
SHA-256, and the fetch timestamp. Replay mode serves recorded responses
byte-identically and refuses cache misses — an offline rerun can never
silently hit the network (fail-closed, DESIGN §4.3).

The cache is integrated with the existing metadata sources at the httpx
transport layer (`FetchCacheTransport`), so their rate-limited
`HttpMetadataSource._get` discipline is retained untouched; the cache wraps
`search()`/`lookup()` calls without source-code changes.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from autoresearch.domain.models import content_checksum, utc_now

FetchFn = Callable[[], Awaitable[httpx.Response]]

CACHE_SCHEMA = "fetch-cache/v1"

#: Response headers persisted per record; anything outside this subset is
#: dropped so secrets and chatty headers never enter the evidence file.
RECORDED_HEADERS: tuple[str, ...] = (
    "content-type",
    "date",
    "retry-after",
    "x-ratelimit-limit",
    "x-ratelimit-remaining",
)


def normalized_cache_key(value: str) -> str:
    """Collapse whitespace and lowercase so query/DOI cache keys stay stable."""
    return " ".join(value.split()).lower()


@dataclass(frozen=True)
class CacheRecord:
    """One persisted request→response evidence record (JSONL line)."""

    source: str
    key: str
    url: str
    status_code: int
    headers: dict[str, str]
    body_b64: str
    body_sha256: str
    fetched_at: str


@dataclass(frozen=True)
class CacheAccess:
    """One in-run cache access (feeds the report's fetch-cache manifest)."""

    source: str
    key: str
    raw_checksum: str
    fetched_at: str
    cache_hit: bool


class FetchCacheError(RuntimeError):
    """Cache evidence is corrupt or the request cannot be served (fail-closed)."""


class FetchCacheMiss(FetchCacheError):
    """Replay mode was asked for a key that was never recorded."""

    def __init__(self, source: str, key: str) -> None:
        super().__init__(f"fetch-cache replay miss for source={source!r} key={key!r}")
        self.source = source
        self.key = key


class FetchCache:
    """JSONL-backed evidence store keyed by (source, key).

    Keys are used verbatim (the transport uses full request URLs); callers of
    the explicit API should normalize with :func:`normalized_cache_key`.
    First-seen evidence wins: a repeat key re-serves the original record and
    appends nothing, so an interrupted live run resumes deterministically and
    a replay run reproduces the recording run byte-identically.
    """

    def __init__(self, path: str | Path | None, *, replay: bool = False) -> None:
        self._path = Path(path) if path is not None else None
        self._replay = replay
        self._records: dict[tuple[str, str], CacheRecord] = {}
        self._accesses: list[CacheAccess] = []
        self._load()

    @property
    def replay(self) -> bool:
        return self._replay

    @property
    def path(self) -> Path | None:
        return self._path

    @property
    def accesses(self) -> tuple[CacheAccess, ...]:
        return tuple(self._accesses)

    def get(self, source: str, key: str) -> CacheRecord | None:
        """Return the first-seen record for a key without logging an access."""
        return self._records.get((source, key))

    def record(self, source: str, key: str, response: httpx.Response, *, url: str) -> CacheRecord:
        """Write one response to the evidence file (write API, DESIGN §9)."""
        if self._replay:
            raise FetchCacheError("cannot record new evidence in replay mode")
        body = response.content
        headers = {name: response.headers.get(name, "") for name in RECORDED_HEADERS if name in response.headers}
        record = CacheRecord(
            source=source,
            key=key,
            url=url,
            status_code=response.status_code,
            headers=headers,
            body_b64=base64.b64encode(body).decode("ascii"),
            body_sha256=hashlib.sha256(body).hexdigest(),
            fetched_at=utc_now(),
        )
        self._records.setdefault((source, key), record)
        self._append(record)
        return record

    async def fetch(self, source: str, key: str, fetch: FetchFn, *, url: str) -> httpx.Response:
        """Fetch (or replay) one response, keyed by (source, normalized key).

        Replay mode raises :class:`FetchCacheMiss` on a miss instead of ever
        touching the network. Live mode records every completed response
        (including error statuses) and reuses first-seen evidence on repeat
        keys. The fetcher must return a response whose request is set (any
        client-managed response qualifies); the body is fully read so it can
        be checksummed.
        """
        record = self._records.get((source, key))
        if record is not None:
            self.log_access(record, cache_hit=True)
            return _response_from_record(record, httpx.Request("GET", url))
        if self._replay:
            raise FetchCacheMiss(source, key)
        response = await fetch()
        await response.aread()
        record = self.record(source, key, response, url=url)
        self.log_access(record, cache_hit=False)
        return response

    def log_access(self, record: CacheRecord, *, cache_hit: bool) -> None:
        """Append one access to the in-run access log (manifest entries)."""
        self._accesses.append(
            CacheAccess(
                source=record.source,
                key=record.key,
                raw_checksum=record.body_sha256,
                fetched_at=record.fetched_at,
                cache_hit=cache_hit,
            )
        )

    def manifest(self) -> dict[str, Any]:
        """Fetch-cache manifest for the report (§8.1: entries + checksum)."""
        entries = [
            {
                "source": access.source,
                "key": access.key,
                "raw_checksum": access.raw_checksum,
                "fetched_at": access.fetched_at,
                "cache_hit": access.cache_hit,
            }
            for access in self._accesses
        ]
        return {"entries": entries, "checksum": content_checksum(entries)}

    def _append(self, record: CacheRecord) -> None:
        if self._path is None:
            return
        self._path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(
            {
                "schema": CACHE_SCHEMA,
                "source": record.source,
                "key": record.key,
                "url": record.url,
                "status_code": record.status_code,
                "headers": record.headers,
                "body_b64": record.body_b64,
                "body_sha256": record.body_sha256,
                "fetched_at": record.fetched_at,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        lines = self._path.read_text(encoding="utf-8").splitlines()
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                record = CacheRecord(
                    source=str(row["source"]),
                    key=str(row["key"]),
                    url=str(row.get("url", "")),
                    status_code=int(row["status_code"]),
                    headers={str(name): str(value) for name, value in (row.get("headers") or {}).items()},
                    body_b64=str(row["body_b64"]),
                    body_sha256=str(row["body_sha256"]),
                    fetched_at=str(row["fetched_at"]),
                )
            except (KeyError, TypeError, ValueError) as error:
                raise FetchCacheError(f"corrupt fetch-cache line {self._path}:{line_number}: {type(error).__name__}") from error
            body = _decode_body(record)
            if hashlib.sha256(body).hexdigest() != record.body_sha256:
                raise FetchCacheError(f"fetch-cache body checksum drift at {self._path}:{line_number}")
            self._records.setdefault((record.source, record.key), record)


class FetchCacheTransport(httpx.AsyncBaseTransport):
    """httpx transport that routes every request through a :class:`FetchCache`.

    Wrap the metadata sources with this transport so their existing
    ``search()`` and new ``lookup()`` calls are cached and replayed without
    touching source code. Cache keys are the full request URLs. With
    ``inner=None`` the transport is strictly offline: replay hits only, and
    any miss raises :class:`FetchCacheMiss` (fail-closed).
    """

    def __init__(self, cache: FetchCache, source: str, inner: httpx.AsyncBaseTransport | None = None) -> None:
        self._cache = cache
        self._source = source
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        key = str(request.url)
        record = self._cache.get(self._source, key)
        if record is not None:
            self._cache.log_access(record, cache_hit=True)
            return _response_from_record(record, request)
        if self._cache.replay:
            raise FetchCacheMiss(self._source, key)
        if self._inner is None:
            raise FetchCacheError(f"live fetch required but no inner transport configured for source {self._source!r} key {key!r}")
        response = await self._inner.handle_async_request(request)
        await response.aread()
        record = self._cache.record(self._source, key, response, url=key)
        self._cache.log_access(record, cache_hit=False)
        return response


def _response_from_record(record: CacheRecord, request: httpx.Request) -> httpx.Response:
    body = _decode_body(record)
    if hashlib.sha256(body).hexdigest() != record.body_sha256:
        raise FetchCacheError(f"fetch-cache body checksum drift for source={record.source!r} key={record.key!r}")
    return httpx.Response(record.status_code, headers=dict(record.headers), content=body, request=request)


def _decode_body(record: CacheRecord) -> bytes:
    try:
        return base64.b64decode(record.body_b64, validate=True)
    except ValueError as error:
        raise FetchCacheError(f"invalid base64 body in fetch-cache record key={record.key!r}") from error
