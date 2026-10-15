"""Provider registration and role-capability validation."""

from __future__ import annotations

from typing import Any

from .capabilities import ROLE_REQUIREMENTS, ModelRole
from .config import ModelAssignment
from .errors import ErrorKind, ProviderError


class ProviderRegistry:
    def __init__(self) -> None:
        self._llm: dict[str, Any] = {}
        self._embeddings: dict[str, Any] = {}

    def register_llm(self, name: str, provider: Any) -> None:
        self._register(name, provider, self._llm)

    def register_embeddings(self, name: str, provider: Any) -> None:
        self._register(name, provider, self._embeddings)

    def _register(self, name: str, provider: Any, target: dict[str, Any]) -> None:
        if not name or name in self._llm or name in self._embeddings:
            raise ValueError(f"Provider '{name}' is invalid or already registered.")
        target[name] = provider

    def names(self) -> list[str]:
        return sorted({*self._llm, *self._embeddings})

    def get(self, name: str) -> Any:
        provider = self._llm.get(name) or self._embeddings.get(name)
        if provider is None:
            raise ProviderError(f"Provider '{name}' is not configured.", ErrorKind.UNSUPPORTED, name)
        return provider

    def get_embeddings(self, name: str) -> Any:
        provider = self._embeddings.get(name)
        if provider is None:
            raise ProviderError(f"Embedding provider '{name}' is not configured.", ErrorKind.UNSUPPORTED, name)
        return provider

    def validate_assignment(self, assignment: ModelAssignment) -> None:
        provider = self.get(assignment.provider)
        required = ROLE_REQUIREMENTS[assignment.role]
        missing = required - frozenset(provider.capabilities)
        if missing:
            names = ", ".join(sorted(item.value for item in missing))
            raise ProviderError(f"Role '{assignment.role.value}' requires unavailable capabilities: {names}.", ErrorKind.UNSUPPORTED, assignment.provider)
        for fallback_name, _ in assignment.fallbacks:
            fallback = self.get(fallback_name)
            missing = required - frozenset(fallback.capabilities)
            if missing:
                names = ", ".join(sorted(item.value for item in missing))
                raise ProviderError(f"Fallback for role '{assignment.role.value}' lacks: {names}.", ErrorKind.UNSUPPORTED, fallback_name)

    async def close(self) -> None:
        seen: set[int] = set()
        for provider in [*self._llm.values(), *self._embeddings.values()]:
            if id(provider) in seen:
                continue
            seen.add(id(provider))
            result = provider.close()
            if hasattr(result, "__await__"):
                await result
