"""Resolve provider keys without exporting or persisting another copy.

Resolution order (first hit wins):
1. The provider's ``api_key_env`` environment variable (explicit config).
2. The OpenCode credential store — usable only for a recognized provider's
   official endpoint; OAuth/subscription tokens are never repurposed.
3. Ambient Anthropic-protocol connector credentials — ONLY for providers of
   ``kind = "anthropic"`` (the key is protocol- and endpoint-family-scoped):
   the ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_AUTH_TOKEN`` variables injected by
   agent sessions, then the local Claude/Codex connector settings file the
   owner designated. Nothing is copied or persisted; keys stay where they are.
"""

from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path
from urllib.parse import urlsplit

from .config import ProviderConfig
from .errors import ErrorKind, ProviderError

# Integration catalog: OpenCode credential-store integration id → protocol
# details. Autodetect picks the session's provider from here; the endpoint is
# always the integration's official one (never repurposed elsewhere).
INTEGRATION_CATALOG: dict[str, dict[str, object]] = {
    "deepseek": {
        "kind": "openai_compatible",
        "base_url": "https://api.deepseek.com/v1",
        # deepseek-chat supports json_object; json_schema is rejected with 400.
        "structured_output_mode": "json_object",
        "models": {
            "literature_extraction": "deepseek-chat",
            "research_reasoning": "deepseek-chat",
            "scientific_critic": "deepseek-chat",
        },
    },
    "alibaba-token-plan-cn": {
        "kind": "openai_compatible",
        "base_url": "https://token-plan.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
        "structured_output_mode": "json_object",
        "models": {
            "literature_extraction": "deepseek-v4.1-flash",
            "research_reasoning": "deepseek-v4.1-flash",
            "scientific_critic": "deepseek-v4.1-flash",
        },
    },
    "zhipuai-coding-plan": {
        "kind": "anthropic",
        "base_url": "https://api.z.ai/api/anthropic",
        "models": {
            "literature_extraction": "glm-5.3-flash",
            "research_reasoning": "glm-5.3",
            "scientific_critic": "glm-5.3",
        },
    },
}

# Priority when several session credentials match: the session's model
# provider wins over other plans the user has connected.
_AUTODETECT_PRIORITY = ("deepseek", "alibaba-token-plan-cn", "zhipuai-coding-plan", "openai")


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
        ("api.z.ai", "/api/anthropic"): ("anthropic", "zhipuai-coding-plan"),
        ("generativelanguage.googleapis.com", ""): ("gemini", "google"),
        ("openrouter.ai", "/api/v1"): ("openai_compatible", "openrouter"),
        # Alibaba Token Plan (China): the OpenCode session's credential is
        # usable only for this official endpoint, selected by exact host+path.
        ("token-plan.cn-beijing.maas.aliyuncs.com", "/compatible-mode/v1"): ("openai_compatible", "alibaba-token-plan-cn"),
        ("dashscope.aliyuncs.com", "/compatible-mode/v1"): ("openai_compatible", "alibaba"),
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


def _ambient_anthropic_key() -> str | None:
    """Anthropic-protocol connector credentials an agent session already has.

    Accepts the standard ANTHROPIC_* variables and the local Claude/Codex
    connector settings file. Only plain API keys are used — OAuth or
    subscription tokens are never repurposed for another protocol.
    """
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN"):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    settings = Path.home() / ".claude" / "settings.json"
    try:
        if settings.is_file():
            data = json.loads(settings.read_text(encoding="utf-8"))
            env_block = data.get("env") if isinstance(data, dict) else None
            if isinstance(env_block, dict):
                value = str(env_block.get("ANTHROPIC_API_KEY") or "").strip()
                if value:
                    return value
    except (OSError, ValueError):
        return None
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
    if config.kind == "anthropic":
        value = _ambient_anthropic_key() or ""
        if value.strip():
            return value.strip()
    raise ProviderError(
        f"Required environment variable '{config.api_key_env}' is not set and no ambient session credential "
        f"matched. Set '{config.api_key_env}', connect the provider in OpenCode, or (for Anthropic-protocol "
        "providers) make the session's ANTHROPIC_API_KEY / connector settings available. OAuth tokens are not API keys.",
        ErrorKind.AUTHENTICATION, config.name,
    )


def detect_session_provider(database: Path | None = None) -> dict[str, object] | None:
    """Auto-detect the OpenCode session's model provider from its credential
    store (the same keys the session itself uses — nothing copied).

    Returns {integration_id, kind, base_url, models, structured_output_mode}
    for the highest-priority ACTIVE api key credential that matches the
    catalog, or None when no session credential is usable. OAuth and
    subscription tokens are never repurposed."""
    from pathlib import Path as _Path

    root = Path(os.environ.get("XDG_DATA_HOME", str(_Path.home() / ".local/share"))) / "opencode"
    db = database or Path(os.environ.get("OPENCODE_DB", str(root / "opencode.db"))).expanduser()
    rows: list[tuple[str, str]] = []
    if db.exists():
        try:
            connection = sqlite3.connect(db.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)
            try:
                has_credentials = connection.execute(
                    "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'credential'"
                ).fetchone()
                if has_credentials:
                    for integration_id, value in connection.execute(
                        "SELECT integration_id, value FROM credential WHERE active = 1 AND connector_id IS NULL"
                    ):
                        parsed = json.loads(value) if isinstance(value, str) else (value or {})
                        key = _api_key(parsed)
                        if key:
                            rows.append((str(integration_id), key))
            finally:
                connection.close()
        except (OSError, ValueError, sqlite3.Error):
            return None
    if not rows:
        return None
    for integration_id in _AUTODETECT_PRIORITY:
        entry = INTEGRATION_CATALOG.get(integration_id)
        if entry and any(row_id == integration_id for row_id, _key in rows):
            return {
                "integration_id": integration_id,
                "kind": entry["kind"],
                "base_url": entry["base_url"],
                "models": dict(entry.get("models") or {}),
                "structured_output_mode": entry.get("structured_output_mode"),
            }
    return None
