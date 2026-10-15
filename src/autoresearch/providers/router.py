"""Capability-aware routing for every LLM and embedding operation."""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable, Sequence
from typing import Any
from uuid import uuid4

from autoresearch.domain.models import ProviderUsage

from .budget import ProviderBudgetGuard
from .capabilities import ModelRole
from .config import ModelAssignment
from .embeddings.base import EmbeddingResult
from .errors import ErrorKind, ProviderError
from .llm.base import CompletionResult, HealthResult, ToolCallResult
from .registry import ProviderRegistry

UsageSink = Callable[[ProviderUsage], Any]


class ModelRouter:
    def __init__(self, registry: ProviderRegistry, assignments: dict[ModelRole, ModelAssignment], *, usage_sink: UsageSink | None = None, budget_guard: ProviderBudgetGuard | None = None) -> None:
        self.registry = registry
        self.assignments = assignments
        self.usage_sink = usage_sink
        self.budget_guard = budget_guard
        for assignment in assignments.values():
            registry.validate_assignment(assignment)
            if assignment.role is ModelRole.EMBEDDINGS:
                registry.get_embeddings(assignment.provider).model_name = assignment.model

    def assignment(self, role: ModelRole | str) -> ModelAssignment:
        resolved = ModelRole(role)
        if resolved not in self.assignments:
            raise ProviderError(f"No provider is assigned to role '{resolved.value}'.", ErrorKind.UNSUPPORTED)
        return self.assignments[resolved]

    async def complete(self, role: ModelRole | str, *, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int | None = None, project_id: str | None = None, run_id: str | None = None, session_id: str | None = None, prompt_version: str = "", operation: str = "completion", response_format: dict[str, Any] | None = None, tools: list[dict[str, Any]] | None = None, structured_output_status: str | None = None) -> CompletionResult:
        assignment = self.assignment(role)
        if project_id and self.budget_guard:
            self.budget_guard.require_allowed(project_id)
        targets = ((assignment.provider, assignment.model), *assignment.fallbacks)
        previous: ProviderError | None = None
        for index, (provider_name, model) in enumerate(targets):
            provider = self.registry.get(provider_name)
            started = time.perf_counter()
            try:
                result = await provider.complete(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens or assignment.max_output_tokens or 2048, response_format=response_format, tools=tools)
            except ProviderError as exc:
                previous = exc
                if not exc.fallback_allowed or index == len(targets) - 1:
                    raise
                continue
            self._record_usage(project_id, run_id, session_id, prompt_version, operation, ModelRole(role), provider_name, model, result.prompt_tokens, result.completion_tokens, result.estimated_cost, result.currency, result.request_id, (time.perf_counter() - started) * 1000, index + int(result.raw.get("retry_count") or 0), structured_output_status if response_format else None)
            return result
        assert previous is not None
        raise previous

    async def structured_output(self, role: ModelRole | str, *, messages: list[dict[str, Any]], schema: dict[str, Any], max_tokens: int | None = None, project_id: str | None = None, run_id: str | None = None, operation: str = "structured_output") -> dict[str, Any]:
        result = await self.complete(role, messages=messages, max_tokens=max_tokens, project_id=project_id, run_id=run_id, operation=operation, prompt_version="structured-output-v1", response_format={"type": "json_schema", "json_schema": schema}, structured_output_status="requested")
        value = _extract_json_object(result.text)
        if value is None:
            raise ProviderError("Provider returned invalid JSON for a structured operation.", ErrorKind.SCHEMA, request_id=result.request_id)
        if not isinstance(value, dict):
            raise ProviderError("Structured provider output must be a JSON object.", ErrorKind.SCHEMA, request_id=result.request_id)
        return value

    async def tool_call(self, role: ModelRole | str, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]], project_id: str | None = None, run_id: str | None = None, session_id: str | None = None, operation: str = "tool_call") -> ToolCallResult:
        result = await self.complete(role, messages=messages, project_id=project_id, run_id=run_id, session_id=session_id, prompt_version="agent-tools-v1", tools=tools, operation=operation)
        raw_calls = result.raw.get("tool_calls") or []
        return ToolCallResult(tuple(raw_calls), result)

    async def embed(self, texts: Sequence[str], *, project_id: str | None = None, run_id: str | None = None, session_id: str | None = None, operation: str = "embeddings") -> EmbeddingResult:
        assignment = self.assignment(ModelRole.EMBEDDINGS)
        if project_id and self.budget_guard:
            self.budget_guard.require_allowed(project_id)
        targets = ((assignment.provider, assignment.model), *assignment.fallbacks)
        previous: ProviderError | None = None
        for index, (provider_name, model) in enumerate(targets):
            provider = self.registry.get_embeddings(provider_name)
            started = time.perf_counter()
            try:
                result = await provider.embed_batch(model=model, texts=texts)
            except ProviderError as exc:
                previous = exc
                if not exc.fallback_allowed or index == len(targets) - 1:
                    raise
                continue
            self._record_usage(project_id, run_id, session_id, "embeddings-v1", operation, ModelRole.EMBEDDINGS, provider_name, model, result.input_tokens, None, result.estimated_cost, result.currency, result.request_id, (time.perf_counter() - started) * 1000, index + int(getattr(result, "retry_count", 0)), None)
            return result
        assert previous is not None
        raise previous

    def stream(self, role: ModelRole | str, *, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int | None = None) -> AsyncIterator[str]:
        assignment = self.assignment(role)
        provider = self.registry.get(assignment.provider)
        return provider.stream(model=assignment.model, messages=messages, temperature=temperature, max_tokens=max_tokens or assignment.max_output_tokens or 2048)

    async def health_check(self, role: ModelRole | str) -> HealthResult:
        assignment = self.assignment(role)
        return await self.registry.get(assignment.provider).health_check()

    def identity(self, role: ModelRole | str) -> tuple[str, str]:
        assignment = self.assignment(role)
        return assignment.provider, assignment.model

    def _record_usage(self, project_id: str | None, run_id: str | None, session_id: str | None, prompt_version: str, operation: str, role: ModelRole, provider: str, model: str, prompt_tokens: int | None, completion_tokens: int | None, estimated_cost: float | None, currency: str | None, request_id: str | None, latency_ms: float, retry_count: int, structured_output_status: str | None) -> None:
        if not self.usage_sink:
            return
        self.usage_sink(ProviderUsage(uuid4().hex, project_id, run_id, operation, role.value, provider, model, prompt_tokens, completion_tokens, estimated_cost, currency, request_id, round(latency_ms, 3), session_id, prompt_version, retry_count, structured_output_status))

    async def close(self) -> None:
        await self.registry.close()


def _extract_json_object(text: str) -> dict[str, Any] | None:
    """Parse the first JSON object embedded in provider output.

    Text-only providers in json_object mode occasionally wrap the object in
    markdown code fences or surrounding prose. Scan for a balanced top-level
    object and raw-decode it; returns None only when no object is parseable.
    """
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = "\n".join(line for line in candidate.splitlines() if not line.strip().startswith("```")).strip()
    decoder = json.JSONDecoder()
    for index, char in enumerate(candidate):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(candidate[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None
