"""Build provider registries and role assignments from secret-free TOML."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

from .budget import ProviderBudgetGuard
from .capabilities import Capability, ModelRole
from .config import ModelAssignment, ProviderConfig, reject_secret_fields
from .embeddings import ExternalEmbeddingProvider
from .llm import AnthropicProvider, GeminiProvider, OpenAICompatibleProvider
from .registry import ProviderRegistry
from .router import ModelRouter, UsageSink


DEFAULT_CAPABILITIES: dict[str, frozenset[Capability]] = {
    "openai_compatible": frozenset({Capability.TEXT, Capability.STREAMING, Capability.STRUCTURED_OUTPUT, Capability.TOOL_USE, Capability.VISION, Capability.LONG_CONTEXT, Capability.REASONING, Capability.CODE}),
    "anthropic": frozenset({Capability.TEXT, Capability.STREAMING, Capability.STRUCTURED_OUTPUT, Capability.TOOL_USE, Capability.VISION, Capability.LONG_CONTEXT, Capability.REASONING, Capability.CODE}),
    "gemini": frozenset({Capability.TEXT, Capability.STREAMING, Capability.STRUCTURED_OUTPUT, Capability.TOOL_USE, Capability.VISION, Capability.LONG_CONTEXT, Capability.REASONING, Capability.CODE}),
    "external_embeddings": frozenset({Capability.EMBEDDINGS}),
}


def load_provider_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Provider configuration not found: {path}")
    with path.open("rb") as handle:
        document = tomllib.load(handle)
    if not isinstance(document, dict):
        raise ValueError("Provider configuration must be a TOML table.")
    return document


def provider_config(name: str, raw: dict[str, Any]) -> ProviderConfig:
    reject_secret_fields(raw, location=f"providers.{name}")
    kind = str(raw.get("kind") or name)
    if kind not in DEFAULT_CAPABILITIES:
        raise ValueError(f"Unsupported provider kind '{kind}'.")
    capabilities = frozenset(Capability(value) for value in raw.get("capabilities", DEFAULT_CAPABILITIES[kind]))
    base_url = str(raw.get("base_url") or "").rstrip("/")
    if not base_url:
        defaults = {"anthropic": "https://api.anthropic.com", "gemini": "https://generativelanguage.googleapis.com"}
        base_url = defaults.get(kind, "")
    if not base_url:
        raise ValueError(f"Provider '{name}' requires base_url.")
    return ProviderConfig(
        name=name,
        kind=kind,
        base_url=base_url,
        api_key_env=str(raw.get("api_key_env") or ""),
        capabilities=capabilities,
        connect_timeout=float(raw.get("connect_timeout", 5.0)),
        read_timeout=float(raw.get("read_timeout", 120.0)),
        total_timeout=float(raw.get("total_timeout", 180.0)),
        max_retries=int(raw.get("max_retries", 2)),
        concurrency=int(raw.get("concurrency", 4)),
        requests_per_second=float(raw["requests_per_second"]) if raw.get("requests_per_second") is not None else None,
        embedding_dimension=int(raw["embedding_dimension"]) if raw.get("embedding_dimension") is not None else None,
        embedding_batch_size=int(raw.get("embedding_batch_size", 64)),
        structured_output_mode=str(raw.get("structured_output_mode", "json_schema")),
        pricing=dict(raw.get("pricing") or {}),
    )


def build_registry(document: dict[str, Any]) -> ProviderRegistry:
    registry = ProviderRegistry()
    for name, values in (document.get("providers") or {}).items():
        if not isinstance(values, dict):
            raise ValueError(f"providers.{name} must be a table.")
        config = provider_config(str(name), values)
        if config.kind == "external_embeddings":
            registry.register_embeddings(str(name), ExternalEmbeddingProvider(config))
        elif config.kind == "anthropic":
            registry.register_llm(str(name), AnthropicProvider(config))
        elif config.kind == "gemini":
            registry.register_llm(str(name), GeminiProvider(config))
        else:
            registry.register_llm(str(name), OpenAICompatibleProvider(config))
    return registry


def build_assignments(document: dict[str, Any]) -> dict[ModelRole, ModelAssignment]:
    assignments: dict[ModelRole, ModelAssignment] = {}
    for role_name, values in (document.get("models") or {}).items():
        if not isinstance(values, dict):
            raise ValueError(f"models.{role_name} must be a table.")
        reject_secret_fields(values, location=f"models.{role_name}")
        role = ModelRole(role_name)
        fallbacks = tuple((str(item["provider"]), str(item["model"])) for item in values.get("fallbacks", []))
        assignments[role] = ModelAssignment(role, str(values.get("provider") or ""), str(values.get("model") or ""), fallbacks, values.get("max_input_tokens"), values.get("max_output_tokens"))
    return assignments


def build_router(path: Path, *, usage_sink: UsageSink | None = None, budget_guard: ProviderBudgetGuard | None = None) -> ModelRouter:
    # A missing provider document is a valid local empty state. Keep the API,
    # projects, sessions, tools, and durable inspection usable; model-backed
    # operations will report an explicit unavailable capability when invoked.
    if not path.exists():
        return ModelRouter(ProviderRegistry(), {}, usage_sink=usage_sink, budget_guard=budget_guard)
    document = load_provider_document(path)
    return ModelRouter(build_registry(document), build_assignments(document), usage_sink=usage_sink, budget_guard=budget_guard)
