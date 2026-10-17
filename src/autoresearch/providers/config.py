"""Secret-free provider and role configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .capabilities import Capability, ModelRole


@dataclass(frozen=True)
class ProviderConfig:
    name: str
    kind: str
    base_url: str
    api_key_env: str
    capabilities: frozenset[Capability]
    connect_timeout: float = 5.0
    # Zero means "no user-specified deadline": a generous FINITE default is
    # applied instead (read 3600s, total 7200s) — slow reasoning still
    # finishes, but a wedged connection can no longer stall a run forever.
    read_timeout: float = 0.0
    total_timeout: float = 0.0
    # Fallback deadlines used when the config leaves read/total at zero.
    default_read_timeout: float = 3600.0
    default_total_timeout: float = 7200.0
    # Output-cap default used when a call site requests UNLIMITED output but
    # the wire protocol REQUIRES a finite cap (Anthropic /v1/messages).
    # Reasoning models spend hidden thinking against the same budget, so this
    # must be model-family maximum class — the old 8192 default silently
    # truncated structured outputs and stalled every skill.
    default_max_output_tokens: int = 65536
    max_retries: int = 2
    concurrency: int = 4
    max_concurrency: int | None = None
    requests_per_second: float | None = None
    thinking_budget: int | None = None
    embedding_dimension: int | None = None
    embedding_batch_size: int = 64
    structured_output_mode: str = "json_schema"
    pricing: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise ValueError("Provider name and kind are required.")
        if not self.api_key_env:
            raise ValueError("Provider api_key_env is required; inline secrets are forbidden.")
        if self.concurrency < 1:
            raise ValueError("Provider concurrency must be at least one.")
        if self.max_concurrency is not None and self.max_concurrency < self.concurrency:
            raise ValueError("Provider max_concurrency must be greater than or equal to concurrency when set.")
        if self.thinking_budget is not None and self.thinking_budget < 1:
            raise ValueError("Provider thinking_budget must be a positive token count.")
        if self.requests_per_second is not None and self.requests_per_second <= 0:
            raise ValueError("requests_per_second must be positive.")
        if self.structured_output_mode not in ("json_schema", "json_object"):
            raise ValueError(f"Provider '{self.name}' structured_output_mode must be 'json_schema' or 'json_object'.")


@dataclass(frozen=True)
class ModelAssignment:
    role: ModelRole
    provider: str
    model: str
    fallbacks: tuple[tuple[str, str], ...] = ()
    max_input_tokens: int | None = None
    max_output_tokens: int | None = None

    def __post_init__(self) -> None:
        if not self.provider or not self.model:
            raise ValueError(f"Role '{self.role.value}' requires provider and model names.")


SECRET_FIELD_NAMES = frozenset({"api_key", "key", "token", "secret", "password", "credential", "authorization"})


def reject_secret_fields(value: dict[str, Any], *, location: str) -> None:
    found = sorted(str(key) for key in value if str(key).lower() in SECRET_FIELD_NAMES)
    if found:
        raise ValueError(
            f"Secret values are not allowed in {location}: {', '.join(found)}. "
            "Store only an api_key_env name and export the secret in the process environment."
        )
