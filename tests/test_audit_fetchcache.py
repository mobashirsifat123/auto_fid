"""Hermetic unit tests for the trust-audit fetch cache (no network)."""

from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from autoresearch.audit.fetchcache import (
    FetchCache,
    FetchCacheError,
    FetchCacheMiss,
    FetchCacheTransport,
    normalized_cache_key,
)
from autoresearch.domain.models import content_checksum


def _fake_response(content: bytes, *, status: int = 200, url: str = "https://api.example.org/works") -> httpx.Response:
    return httpx.Response(
        status,
        headers={"Content-Type": "application/json", "Date": "Fri, 16 Oct 2026 10:00:00 GMT", "X-Dropped": "secret"},
        content=content,
        request=httpx.Request("GET", url),
    )


async def _refusing_fetch() -> httpx.Response:
    raise AssertionError("network must never be touched in replay mode")


async def test_fetch_records_evidence_line(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fetches.jsonl")
    calls: list[int] = []

    async def fetcher() -> httpx.Response:
        calls.append(1)
        return _fake_response(b'{"data": []}')

    response = await cache.fetch("crossref", normalized_cache_key("10.1000/Example DOI "), fetcher, url="https://api.example.org/works/10.1000/example-doi")
    assert response.status_code == 200
    assert response.content == b'{"data": []}'
    assert calls == [1]
    assert cache.accesses[-1].cache_hit is False
    assert cache.accesses[-1].key == "10.1000/example doi"

    lines = (tmp_path / "fetches.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["schema"] == "fetch-cache/v1"
    assert row["source"] == "crossref"
    assert row["key"] == "10.1000/example doi"
    assert row["status_code"] == 200
    assert row["headers"] == {"content-type": "application/json", "date": "Fri, 16 Oct 2026 10:00:00 GMT"}
    assert "x-dropped" not in row["headers"]
    assert row["body_sha256"] == hashlib.sha256(b'{"data": []}').hexdigest()
    assert base64.b64decode(row["body_b64"]) == b'{"data": []}'
    assert row["fetched_at"]


async def test_replay_serves_byte_identical_and_refuses_miss(tmp_path: Path) -> None:
    live = FetchCache(tmp_path / "fetches.jsonl")
    await live.fetch("crossref", normalized_cache_key("10.1000/x"), lambda: _afake(), url="https://api.example.org/works/10.1000/x")

    replay = FetchCache(tmp_path / "fetches.jsonl", replay=True)
    response = await replay.fetch("crossref", normalized_cache_key("10.1000/X "), _refusing_fetch, url="https://api.example.org/works/10.1000/x")
    assert response.status_code == 200
    assert response.content == b'{"data": []}'
    assert response.headers["content-type"] == "application/json"
    assert "x-dropped" not in response.headers
    response.raise_for_status()
    assert replay.accesses[-1].cache_hit is True

    with pytest.raises(FetchCacheMiss):
        await replay.fetch("crossref", "never-recorded", _refusing_fetch, url="https://api.example.org/works/other")


async def _afake() -> httpx.Response:
    return _fake_response(b'{"data": []}')


async def test_live_mode_reuses_first_seen_evidence(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fetches.jsonl")

    async def second_version() -> httpx.Response:
        return _fake_response(b'{"data": ["v2"]}')

    first = await cache.fetch("semantic_scholar", "title:stable", lambda: _afake(), url="https://api.example.org/search")
    second = await cache.fetch("semantic_scholar", "title:stable", second_version, url="https://api.example.org/search")
    assert first.content == second.content == b'{"data": []}'
    assert [access.cache_hit for access in cache.accesses] == [False, True]
    assert len((tmp_path / "fetches.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_corrupt_line_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "fetches.jsonl"
    cache = FetchCache(path)
    cache.record("crossref", "10.1000/x", _fake_response(b'{"ok": 1}'), url="https://api.example.org/works/10.1000/x")
    good_line = path.read_text(encoding="utf-8").splitlines()[0]

    path.write_text("not json at all\n" + good_line + "\n", encoding="utf-8")
    with pytest.raises(FetchCacheError):
        FetchCache(path)


def test_tampered_body_checksum_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "fetches.jsonl"
    cache = FetchCache(path)
    cache.record("crossref", "10.1000/x", _fake_response(b'{"ok": 1}'), url="https://api.example.org/works/10.1000/x")
    row = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
    row["body_b64"] = base64.b64encode(b'{"tampered": 1}').decode("ascii")
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(FetchCacheError):
        FetchCache(path)


def test_record_refuses_replay_mode(tmp_path: Path) -> None:
    cache = FetchCache(None, replay=True)
    with pytest.raises(FetchCacheError):
        cache.record("crossref", "k", _fake_response(b"{}"), url="https://api.example.org/works")


async def test_manifest_shape(tmp_path: Path) -> None:
    cache = FetchCache(tmp_path / "fetches.jsonl")
    await cache.fetch("crossref", "10.1000/a", lambda: _afake(), url="https://api.example.org/works/10.1000/a")
    await cache.fetch("openalex", "title:query", lambda: _afake(), url="https://api.openalex.org/works")
    manifest = cache.manifest()
    assert set(manifest) == {"entries", "checksum"}
    assert [entry["source"] for entry in manifest["entries"]] == ["crossref", "openalex"]
    assert manifest["entries"][0]["cache_hit"] is False
    assert manifest["entries"][0]["raw_checksum"] == hashlib.sha256(b'{"data": []}').hexdigest()
    assert manifest["entries"][0]["fetched_at"]
    assert manifest["checksum"] == content_checksum(manifest["entries"])


async def test_transport_live_then_replay(tmp_path: Path) -> None:
    path = tmp_path / "transport.jsonl"
    inner = httpx.MockTransport(lambda request: httpx.Response(200, json={"source": request.url.host}))
    cache = FetchCache(path)
    client = httpx.AsyncClient(
        base_url="https://api.example.org",
        transport=FetchCacheTransport(cache, "semantic_scholar", inner),
    )
    response = await client.get("/paper/search", params={"query": "attention"})
    await client.aclose()
    assert response.json() == {"source": "api.example.org"}
    assert cache.accesses[0].source == "semantic_scholar"
    assert cache.accesses[0].cache_hit is False

    replay_cache = FetchCache(path, replay=True)
    replay_client = httpx.AsyncClient(
        base_url="https://api.example.org",
        transport=FetchCacheTransport(replay_cache, "semantic_scholar"),
    )
    replayed = await replay_client.get("/paper/search", params={"query": "attention"})
    assert replayed.status_code == response.status_code
    assert replayed.content == response.content
    assert replayed.headers["content-type"] == response.headers["content-type"]
    assert replay_cache.accesses[0].cache_hit is True

    with pytest.raises(FetchCacheMiss):
        await replay_client.get("/paper/search", params={"query": "different"})
    await replay_client.aclose()
