"""Offline credential handoff tests; never read the developer's credentials."""

import json
import sqlite3

import httpx
import pytest
from typer.testing import CliRunner

from autoresearch.providers.capabilities import Capability
from autoresearch.providers.config import ProviderConfig
from autoresearch.providers.credentials import resolve_api_key
from autoresearch.providers.errors import ProviderError
from autoresearch.providers.llm import OpenAICompatibleProvider


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "data"))
    monkeypatch.delenv("OPENCODE_DB", raising=False)
    monkeypatch.delenv("AUTORESEARCH_TEST_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    root = tmp_path / "data/opencode"
    root.mkdir(parents=True)
    return root


def config(url="https://api.deepseek.com"):
    return ProviderConfig("reasoning", "openai_compatible", url, "AUTORESEARCH_TEST_KEY", frozenset({Capability.TEXT}))


def anthropic_config(url="https://api.z.ai/api/anthropic"):
    return ProviderConfig("reasoning", "anthropic", url, "AUTORESEARCH_TEST_KEY", frozenset({Capability.TEXT}))


def database(root, rows):
    path = root / "opencode.db"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE credential (integration_id TEXT, active INTEGER, connector_id TEXT, value TEXT)")
        db.executemany("INSERT INTO credential VALUES (?, ?, ?, ?)", rows)
    return path


def row(provider="deepseek", active=1, kind="key", key="fixture-only-key"):
    return provider, active, None, json.dumps({"type": kind, "key": key})


def test_database_key_in_memory_without_environment_mutation(store, monkeypatch):
    import os

    path = database(store, [row(active=0, key="inactive"), row()])
    before = path.read_bytes()
    assert resolve_api_key(config()) == "fixture-only-key"
    assert "AUTORESEARCH_TEST_KEY" not in os.environ
    assert path.read_bytes() == before
    monkeypatch.setenv("AUTORESEARCH_TEST_KEY", "explicit-environment")
    assert resolve_api_key(config()) == "explicit-environment"


@pytest.mark.parametrize("rows", [[row(provider="openai")], [row(active=0)], [row(kind="oauth")], [row(), row()]])
def test_no_wrong_inactive_oauth_or_ambiguous_key(store, rows):
    database(store, rows)
    with pytest.raises(ProviderError):
        resolve_api_key(config())


@pytest.mark.parametrize("url", ["https://api.deepseek.com.evil.test", "http://api.deepseek.com", "https://api.deepseek.com:444", "https://api.deepseek.com/proxy", "https://user@api.deepseek.com", "https://api.openai.com/v1"])
def test_key_cannot_cross_endpoint_boundary(store, url):
    database(store, [row()])
    with pytest.raises(ProviderError):
        resolve_api_key(config(url))


def test_legacy_key_and_authoritative_database(store):
    (store / "auth.json").write_text(json.dumps({"deepseek": {"type": "api", "key": "legacy"}}))
    assert resolve_api_key(config("https://api.deepseek.com/v1/")) == "legacy"
    database(store, [row(active=0)])
    with pytest.raises(ProviderError):
        resolve_api_key(config())


def test_older_session_database_still_uses_legacy_auth(store):
    with sqlite3.connect(store / "opencode.db") as db:
        db.execute("CREATE TABLE session (id TEXT)")
    (store / "auth.json").write_text(json.dumps({"deepseek": {"type": "api", "key": "legacy"}}))
    assert resolve_api_key(config()) == "legacy"


def test_corrupt_store_error_does_not_include_contents(store):
    (store / "auth.json").write_text("fixture-secret-not-valid-json")
    with pytest.raises(ProviderError) as exc:
        resolve_api_key(config())
    assert "fixture-secret" not in str(exc.value)


def test_explicit_database_path(store, monkeypatch, tmp_path):
    custom = tmp_path / "alternate"
    custom.mkdir()
    path = database(custom, [row()])
    monkeypatch.setenv("OPENCODE_DB", str(path))
    assert resolve_api_key(config()) == "fixture-only-key"


@pytest.mark.asyncio
async def test_real_adapter_uses_key_only_in_request_header(store):
    database(store, [row()])

    def respond(request):
        assert request.headers["Authorization"] == "Bearer fixture-only-key"
        assert "fixture-only-key" not in request.content.decode()
        return httpx.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    provider = OpenAICompatibleProvider(config(), transport=httpx.MockTransport(respond))
    try:
        assert (await provider.complete(model="fixture", messages=[])).text == "ok"
    finally:
        await provider.close()


def test_ambient_connector_key_used_for_anthropic_kind(store, tmp_path):
    """Skills run inside an agent session pick the connector's key without a
    separate export: the Claude/Codex connector settings file is an accepted
    ambient source for Anthropic-protocol providers."""
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(json.dumps(
        {"env": {"ANTHROPIC_API_KEY": "connector-fixture-key"}}
    ))
    assert resolve_api_key(anthropic_config()) == "connector-fixture-key"


def test_anthropic_env_var_beats_connector_file(store, tmp_path, monkeypatch):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(json.dumps(
        {"env": {"ANTHROPIC_API_KEY": "connector-fixture-key"}}
    ))
    monkeypatch.setenv("ANTHROPIC_AUTH_TOKEN", "session-fixture-key")
    assert resolve_api_key(anthropic_config()) == "session-fixture-key"


def test_explicit_env_still_wins_over_ambient_chain(store, tmp_path, monkeypatch):
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(json.dumps(
        {"env": {"ANTHROPIC_API_KEY": "connector-fixture-key"}}
    ))
    monkeypatch.setenv("AUTORESEARCH_TEST_KEY", "explicit-wins")
    assert resolve_api_key(anthropic_config()) == "explicit-wins"


def test_ambient_anthropic_key_never_crosses_protocol(store, tmp_path, monkeypatch):
    """An OpenAI-compatible provider must never borrow the Anthropic session
    key, even when both are configured in the same shell."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "session-fixture-key")
    (tmp_path / ".claude").mkdir()
    (tmp_path / ".claude" / "settings.json").write_text(json.dumps(
        {"env": {"ANTHROPIC_API_KEY": "session-fixture-key"}}
    ))
    with pytest.raises(ProviderError):
        resolve_api_key(config())


def test_skill_doctor_detects_missing_credentials_without_network(store, monkeypatch):
    from types import SimpleNamespace

    from autoresearch.cli import skills
    from autoresearch.providers.registry import ProviderRegistry

    provider = OpenAICompatibleProvider(config())
    registry = ProviderRegistry()
    registry.register_llm("reasoning", provider)

    async def close():
        await registry.close()

    fake = SimpleNamespace(router=SimpleNamespace(
        assignments={r: SimpleNamespace(provider="reasoning") for r in ["literature_extraction", "research_reasoning", "scientific_critic"]},
        registry=registry,
    ), close=close)
    monkeypatch.setattr(skills, "ApplicationContainer", lambda _: fake)
    result = CliRunner().invoke(skills.skill_app, ["doctor"])
    checks = json.loads(result.output)["checks"]
    check = next(c for c in checks if c["check"] == "providers")
    assert check["ok"] is False and "credentials unavailable" in check["detail"]


def _credential_row(integration_id, active=1, kind="key", key="fixture-session-key"):
    return integration_id, active, None, json.dumps({"type": kind, "key": key})


def test_autodetect_builds_router_from_session_store(store, monkeypatch, tmp_path):
    """With NO provider config file, the skills follow the OpenCode session's
    model provider automatically (owner requirement: no separate API-key
    setup inside an agent session)."""
    database(store, [_credential_row("deepseek")])
    from autoresearch.providers.factory import build_router

    router = build_router(tmp_path / "missing.toml")
    provider = router.registry.get("reasoning")
    assert provider is not None
    assert provider.config.kind == "openai_compatible"
    assert provider.config.base_url == "https://api.deepseek.com/v1"
    assert router.assignments["research_reasoning"].model == "deepseek-chat"
    assert provider.api_key() == "fixture-session-key"


def test_autodetect_skips_oauth_and_unknown_integrations(store):
    database(store, [_credential_row("openai", kind="oauth", key=""), _credential_row("some-random-vendor")])
    from autoresearch.providers.factory import build_router
    from autoresearch.providers.errors import ProviderError

    router = build_router(__import__("pathlib").Path("/nonexistent/config.toml"))
    assert not router.assignments
    import pytest

    with pytest.raises(ProviderError):
        router.registry.get("reasoning")


def test_autodetect_priority_prefers_session_provider(store, tmp_path):
    database(store, [_credential_row("zhipuai-coding-plan"), _credential_row("deepseek")])
    from autoresearch.providers.factory import build_router

    router = build_router(tmp_path / "missing.toml")
    assert router.registry.get("reasoning").config.base_url == "https://api.deepseek.com/v1"


def test_auto_provider_flag_overrides_existing_config(store, monkeypatch, tmp_path):
    database(store, [_credential_row("deepseek")])
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[providers.reasoning]\nkind = "openai_compatible"\nbase_url = "https://api.example.com/v1"\napi_key_env = "OTHER"\n'
        '[models.research_reasoning]\nprovider = "reasoning"\nmodel = "other-model"\n'
    )
    monkeypatch.setenv("OTHER", "configured-key")
    from autoresearch.providers.factory import build_router

    configured = build_router(config_path)
    assert configured.registry.get("reasoning").config.base_url == "https://api.example.com/v1"
    auto = build_router(config_path, auto_provider=True)
    assert auto.registry.get("reasoning").config.base_url == "https://api.deepseek.com/v1"
    assert auto.registry.get("reasoning").api_key() == "fixture-session-key"
