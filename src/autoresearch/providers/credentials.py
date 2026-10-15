"""Resolve provider keys without exporting or persisting another copy.

Environment wins. OpenCode API credentials are usable only for a recognized
provider's official endpoint; OAuth/subscription tokens are never repurposed.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit

from .config import ProviderConfig
from .errors import ErrorKind, ProviderError


def _opencode_provider(config: ProviderConfig) -> str | None:
    url = urlsplit(config.base_url)
    if url.scheme != "https" or url.username or url.password or url.port not in (None, 443) or url.query or url.fragment:
        return None
    endpoints = {
        ("api.deepseek.com", ""): ("openai_compatible", "deepseek"),
        ("api.deepseek.com", "/v1"): ("openai_compatible", "deepseek"),
        ("api.openai.com", "/v1"): ("openai_compatible", "openai"),
        ("api.anthropic.com", ""): ("anthropic", "anthropic"),
        ("api.anthropic.com", "/v1"): ("anthropic", "anthropic"),
        ("generativelanguage.googleapis.com", ""): ("gemini", "google"),
        ("openrouter.ai", "/api/v1"): ("openai_compatible", "openrouter"),
    }
    match = endpoints.get((url.hostname, url.path.rstrip("/")))
    if match and (config.kind == match[0] or config.kind == "external_embeddings" and match[1] == "openai"):
        return match[1]
    return None


def _api_key(value: object) -> str | None:
    if not isinstance(value, dict) or value.get("type") not in ("api", "key"):
        return None
    key = value.get("key")
    return key if isinstance(key, str) and key.strip() else None


def _opencode_key(provider: str) -> str | None:
    root = Path(os.environ.get("XDG_DATA_HOME", str(Path.home() / ".local/share"))) / "opencode"
    database = Path(os.environ.get("OPENCODE_DB", str(root / "opencode.db"))).expanduser()
    # A v2 credential table is authoritative. Older databases may contain only
    # sessions and still use auth.json. Never resurrect logged-out v2 keys.
    if database.exists():
        connection = sqlite3.connect(database.resolve().as_uri() + "?mode=ro", uri=True, timeout=1)
        try:
            has_credentials = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'credential'"
            ).fetchone()
            if has_credentials:
                rows = connection.execute(
                    "SELECT value FROM credential WHERE integration_id = ? AND active = 1 AND connector_id IS NULL",
                    (provider,),
                ).fetchall()
                if len(rows) != 1:
                    return None
                return _api_key(json.loads(rows[0][0]))
        finally:
            connection.close()
    legacy = root / "auth.json"
    if legacy.is_file():
        values = json.loads(legacy.read_text(encoding="utf-8"))
        if isinstance(values, dict):
            return _api_key(values.get(provider))
    return None


def resolve_api_key(config: ProviderConfig) -> str:
    value = os.environ.get(config.api_key_env, "")
    if value.strip():
        return value
    provider = _opencode_provider(config)
    if provider:
        try:
            value = _opencode_key(provider)
        except (OSError, ValueError, sqlite3.Error):
            # Neither file contents nor database exception text may reach logs.
            raise ProviderError(
                f"Cannot read the matching OpenCode API credential; reconnect '{provider}' in OpenCode or set '{config.api_key_env}'.",
                ErrorKind.AUTHENTICATION, config.name,
            ) from None
        if value:
            return value
    raise ProviderError(
        f"Required environment variable '{config.api_key_env}' is not set and no matching OpenCode API key is available. "
        "Use OpenCode /connect for this provider or configure the environment. OAuth tokens are not API keys.",
        ErrorKind.AUTHENTICATION, config.name,
    )
