from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from autoresearch.providers.budget import ProviderBudgetGuard
from autoresearch.providers.capabilities import Capability, ModelRole
from autoresearch.providers.config import ModelAssignment, ProviderConfig, reject_secret_fields
from autoresearch.providers.errors import ErrorKind, ProviderError
from autoresearch.providers.llm import AnthropicProvider, GeminiProvider, OpenAICompatibleProvider
from autoresearch.providers.registry import ProviderRegistry
from autoresearch.providers.redaction import redact_text
from autoresearch.providers.router import ModelRouter
from autoresearch.providers.testing import FakeEmbeddingProvider, FakeLLMProvider
from autoresearch.storage import ResearchRepository


def _router(tmp_path, *, usage=True):
    repository = ResearchRepository(tmp_path / "research.sqlite3", tmp_path / "artifacts")
    project = repository.create_project("Provider test")
    llm = FakeLLMProvider(responses=[{"value": "ok"}])
    embeddings = FakeEmbeddingProvider()
    registry = ProviderRegistry()
    registry.register_llm("fake", llm)
    registry.register_embeddings("vectors", embeddings)
    assignments = {
        ModelRole.LITERATURE_EXTRACTION: ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "fake", "fake-model"),
        ModelRole.EMBEDDINGS: ModelAssignment(ModelRole.EMBEDDINGS, "vectors", "fake-embedding-v1"),
    }
    return ModelRouter(registry, assignments, usage_sink=repository.record_provider_usage if usage else None, budget_guard=ProviderBudgetGuard(repository)), repository, project


@pytest.mark.asyncio
async def test_router_records_structured_and_embedding_usage(tmp_path):
    router, repository, project = _router(tmp_path)
    result = await router.structured_output(ModelRole.LITERATURE_EXTRACTION, messages=[{"role": "user", "content": "test"}], schema={"name": "test"}, project_id=project.id, run_id=None)
    vectors = await router.embed(["alpha", "beta"], project_id=project.id)
    assert result == {"value": "ok"}
    assert len(vectors.vectors) == 2
    rows = repository.rows("SELECT role,provider,estimated_cost,structured_output_status,prompt_version FROM provider_usage ORDER BY created_at")
    assert {row["role"] for row in rows} == {"literature_extraction", "embeddings"}
    assert all(row["provider"] in {"fake", "vectors"} for row in rows)
    structured = next(row for row in rows if row["role"] == "literature_extraction")
    assert structured["structured_output_status"] == "requested"
    assert structured["prompt_version"] == "structured-output-v1"


def test_registry_rejects_missing_capabilities():
    provider = FakeLLMProvider()
    provider.capabilities = frozenset({Capability.TEXT})
    registry = ProviderRegistry()
    registry.register_llm("weak", provider)
    with pytest.raises(ProviderError, match="structured_output"):
        ModelRouter(registry, {ModelRole.LITERATURE_EXTRACTION: ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "weak", "model")})


def test_research_reasoning_assignment_requires_structured_output():
    provider = FakeLLMProvider()
    provider.capabilities = frozenset({Capability.TEXT, Capability.REASONING})
    registry = ProviderRegistry()
    registry.register_llm("reasoner", provider)
    with pytest.raises(ProviderError, match="structured_output"):
        ModelRouter(
            registry,
            {ModelRole.RESEARCH_REASONING: ModelAssignment(ModelRole.RESEARCH_REASONING, "reasoner", "model")},
        )


def test_agent_history_redacts_secrets(tmp_path):
    repository = ResearchRepository(tmp_path / "research.sqlite3", tmp_path / "artifacts")
    project = repository.create_project("Redaction test")
    from autoresearch.agent.store import AgentStore

    store = AgentStore(repository)
    session = store.create_session(project.id, str(tmp_path), str(tmp_path))
    turn = store.create_turn(session.id, "token=super-secret-value", override_mode=None, max_tool_iterations=1, max_model_calls=1, time_budget_seconds=10, token_budget=100, cost_budget=None)
    store.append_message(session.id, "assistant", "authorization: Bearer super-secret-value", turn_id=turn.id)
    store.append_event(session.id, "agent.test", {"url": "https://user:pass@example.test/x", "token": "super-secret-value"})
    store.create_tool_call(session.id, turn.id, "read_file", {"path": "notes.txt", "api_key": "super-secret-value"})

    raw = repository.rows("SELECT content,metadata_json FROM agent_messages")
    raw_events = repository.rows("SELECT payload_json FROM agent_events")
    raw_calls = repository.rows("SELECT arguments_json FROM agent_tool_calls")
    assert all("super-secret-value" not in str(row) for row in (*raw, *raw_events, *raw_calls))
    assert "[REDACTED]" in raw[0]["content"]


@pytest.mark.asyncio
async def test_fallback_is_transient_only():
    class Failure(FakeLLMProvider):
        def __init__(self, kind):
            super().__init__()
            self.kind = kind
        async def complete(self, **kwargs):
            raise ProviderError("failure", self.kind, self.name)

    fallback = FakeLLMProvider(responses=[{"ok": True}])
    registry = ProviderRegistry()
    registry.register_llm("primary", Failure(ErrorKind.RATE_LIMIT))
    registry.register_llm("fallback", fallback)
    assignment = ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "primary", "a", (("fallback", "b"),))
    router = ModelRouter(registry, {ModelRole.LITERATURE_EXTRACTION: assignment})
    assert await router.structured_output(ModelRole.LITERATURE_EXTRACTION, messages=[], schema={}) == {"ok": True}

    registry2 = ProviderRegistry()
    registry2.register_llm("primary", Failure(ErrorKind.INVALID_REQUEST))
    registry2.register_llm("fallback", FakeLLMProvider(responses=[{"wrong": True}]))
    router2 = ModelRouter(registry2, {ModelRole.LITERATURE_EXTRACTION: assignment})
    with pytest.raises(ProviderError) as error:
        await router2.structured_output(ModelRole.LITERATURE_EXTRACTION, messages=[], schema={})
    assert error.value.kind is ErrorKind.INVALID_REQUEST


@pytest.mark.asyncio
async def test_budget_hard_stop_blocks_provider_call(tmp_path):
    router, repository, project = _router(tmp_path)
    repository.set_project_budget(project.id, hard_stop=0)
    with pytest.raises(ProviderError) as error:
        await router.embed(["blocked"], project_id=project.id)
    assert error.value.kind is ErrorKind.BUDGET


@pytest.mark.asyncio
async def test_openai_compatible_adapter_contract(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "not-a-real-secret")
    def handler(request: httpx.Request):
        assert request.headers["authorization"] == "Bearer not-a-real-secret"
        body = json.loads(request.content)
        assert body["model"] == "external-model"
        assert body["response_format"]["json_schema"]["strict"] is True
        return httpx.Response(200, json={"id": "req-1", "model": "external-model", "choices": [{"message": {"content": '{"ok":true}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
    config = ProviderConfig("external", "openai_compatible", "https://provider.example/v1", "TEST_PROVIDER_KEY", frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}), pricing={"input_per_million": 1, "output_per_million": 2, "currency": "USD"})
    provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler))
    result = await provider.complete(model="external-model", messages=[{"role": "user", "content": "hello"}], response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}})
    assert result.text == '{"ok":true}'
    assert result.estimated_cost == pytest.approx(0.000014)
    await provider.close()


@pytest.mark.asyncio
async def test_openai_compatible_json_object_mode_translates_schema_and_injects_json_word(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "not-a-real-secret")

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["response_format"] == {"type": "json_object"}, "json_schema must be translated to json_object"
        user_content = body["messages"][-1]["content"]
        assert "json" in user_content.lower(), "providers requiring the json word must get an injected instruction"
        assert "Return your response as a single JSON object" in user_content
        return httpx.Response(200, json={"id": "req-3", "model": "external-model", "choices": [{"message": {"content": '{"ok":true}'}}], "usage": {"prompt_tokens": 10, "completion_tokens": 2}})

    config = ProviderConfig("external", "openai_compatible", "https://provider.example/v1", "TEST_PROVIDER_KEY", frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}), structured_output_mode="json_object")
    provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler))
    result = await provider.complete(model="external-model", messages=[{"role": "user", "content": "Return one object with a boolean ok field."}], response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}})
    assert json.loads(result.text) == {"ok": True}
    await provider.close()


@pytest.mark.asyncio
async def test_openai_compatible_json_object_mode_leaves_prompt_untouched_when_json_present(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "not-a-real-secret")

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        user_content = body["messages"][-1]["content"]
        assert user_content == "Return JSON now", "prompt with the json word must not be amended"
        assert body["response_format"] == {"type": "json_object"}
        return httpx.Response(200, json={"id": "req-4", "model": "external-model", "choices": [{"message": {"content": '{"ok":true}'}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})

    config = ProviderConfig("external", "openai_compatible", "https://provider.example/v1", "TEST_PROVIDER_KEY", frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}), structured_output_mode="json_object")
    provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler))
    result = await provider.complete(model="external-model", messages=[{"role": "user", "content": "Return JSON now"}], response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}})
    assert result.text == '{"ok":true}'
    await provider.close()


@pytest.mark.asyncio
async def test_openai_compatible_adapter_retries_rate_limit(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "not-a-real-secret")
    calls = 0

    def handler(request: httpx.Request):
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"retry-after": "0"}, json={"error": {"message": "slow down"}})
        return httpx.Response(200, json={"id": "req-2", "model": "external-model", "choices": [{"message": {"content": '{"ok":true}'}}], "usage": {"prompt_tokens": 1, "completion_tokens": 1}})

    config = ProviderConfig("external", "openai_compatible", "https://provider.example/v1", "TEST_PROVIDER_KEY", frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}), max_retries=1)
    provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler))
    result = await provider.complete(model="external-model", messages=[{"role": "user", "content": "hello"}], response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object"}}})
    assert calls == 2 and json.loads(result.text) == {"ok": True}
    await provider.close()


@pytest.mark.asyncio
async def test_anthropic_adapter_forces_strict_structured_tool(monkeypatch):
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "not-a-real-secret")

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        assert body["tools"][0]["name"] == "answer"
        assert body["tools"][0]["strict"] is True
        assert body["tools"][0]["input_schema"]["required"] == ["ok"]
        assert body["tool_choice"] == {"type": "tool", "name": "answer", "disable_parallel_tool_use": True}
        return httpx.Response(200, json={"id": "msg-1", "model": "claude-test", "content": [{"type": "tool_use", "name": "answer", "input": {"ok": True}}], "usage": {"input_tokens": 8, "output_tokens": 3}})

    config = ProviderConfig("anthropic", "anthropic", "https://api.anthropic.com", "TEST_ANTHROPIC_KEY", frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}))
    provider = AnthropicProvider(config, transport=httpx.MockTransport(handler))
    result = await provider.complete(model="claude-test", messages=[{"role": "user", "content": "hello"}], response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}})
    assert json.loads(result.text) == {"ok": True}
    await provider.close()


@pytest.mark.asyncio
async def test_gemini_adapter_sends_json_schema(monkeypatch):
    monkeypatch.setenv("TEST_GEMINI_KEY", "not-a-real-secret")

    def handler(request: httpx.Request):
        body = json.loads(request.content)
        config = body["generationConfig"]
        assert config["responseMimeType"] == "application/json"
        assert config["responseJsonSchema"]["required"] == ["ok"]
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": '{"ok":true}'}]}}], "usageMetadata": {"promptTokenCount": 7, "candidatesTokenCount": 2}})

    config = ProviderConfig("gemini", "gemini", "https://generativelanguage.googleapis.com", "TEST_GEMINI_KEY", frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}))
    provider = GeminiProvider(config, transport=httpx.MockTransport(handler))
    result = await provider.complete(model="gemini-test", messages=[{"role": "user", "content": "hello"}], response_format={"type": "json_schema", "json_schema": {"name": "answer", "schema": {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}}})
    assert json.loads(result.text) == {"ok": True}
    await provider.close()


def test_structured_output_extracts_fenced_or_prose_wrapped_json(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "not-a-real-secret")

    async def probe(fenced_text: str):
        def handler(request: httpx.Request):
            return httpx.Response(200, json={"id": "req-5", "model": "external-model", "choices": [{"message": {"content": fenced_text}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
        config = ProviderConfig("external", "openai_compatible", "https://provider.example/v1", "TEST_PROVIDER_KEY", frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}), structured_output_mode="json_object")
        provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler))
        registry = ProviderRegistry()
        registry.register_llm("external", provider)
        router = ModelRouter(registry, {ModelRole.LITERATURE_EXTRACTION: ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "external", "external-model")})
        try:
            return await router.structured_output(ModelRole.LITERATURE_EXTRACTION, messages=[{"role": "user", "content": "Return JSON now"}], schema={})
        finally:
            await provider.close()

    assert asyncio.run(probe('```json\n{"ok": true}\n```')) == {"ok": True}
    assert asyncio.run(probe('Here is the result: {"ok": true}. That is all.')) == {"ok": True}


def test_structured_output_rejects_when_no_object_is_parseable(monkeypatch):
    monkeypatch.setenv("TEST_PROVIDER_KEY", "not-a-real-secret")

    def handler(request: httpx.Request):
        return httpx.Response(200, json={"id": "req-6", "model": "external-model", "choices": [{"message": {"content": "I am sorry, I cannot produce JSON."}}], "usage": {"prompt_tokens": 5, "completion_tokens": 2}})
    config = ProviderConfig("external", "openai_compatible", "https://provider.example/v1", "TEST_PROVIDER_KEY", frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}), structured_output_mode="json_object")
    provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler))
    registry = ProviderRegistry()
    registry.register_llm("external", provider)
    router = ModelRouter(registry, {ModelRole.LITERATURE_EXTRACTION: ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "external", "external-model")})
    with pytest.raises(ProviderError, match="invalid JSON"):
        asyncio.run(router.structured_output(ModelRole.LITERATURE_EXTRACTION, messages=[{"role": "user", "content": "Return JSON now"}], schema={}))


def test_configuration_rejects_inline_secrets():
    with pytest.raises(ValueError, match="Secret values"):
        reject_secret_fields({"api_key": "never-store-this"}, location="providers.test")


def test_redaction_removes_url_credentials_and_sensitive_query_values():
    value = "https://alice:password@example.test/v1?token=secret-value&safe=yes"
    redacted = redact_text(value)
    assert "password" not in redacted
    assert "secret-value" not in redacted
    assert redacted == "https://[REDACTED]example.test/v1?token=[REDACTED]&safe=yes"
