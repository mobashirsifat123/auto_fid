"""OpenAI-compatible chat-completions adapter."""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from typing import Any

from ..config import ProviderConfig
from ..errors import ErrorKind, ProviderError
from .base import CompletionResult, ModelInfo
from .http_base import HTTPProviderBase


def _thinking_budget(configured: int | None) -> int | None:
    """Resolve the reasoning budget: the env override wins, else provider config.

    GLM-family thinking-on token-plan endpoints must keep thinking enabled but
    accept budget_tokens; capping it prevents minutes-long deliberation per
    extraction call. Config is the permanent path; the env var stays for
    per-shell experiments. No effect when both are unset.
    """
    raw = os.getenv("AUTORESEARCH_THINKING_BUDGET", "").strip()
    if raw.isdigit() and int(raw) > 0:
        return int(raw)
    return configured if configured and configured > 0 else None


class OpenAICompatibleProvider(HTTPProviderBase):
    def __init__(self, config: ProviderConfig, *, transport: Any | None = None) -> None:
        super().__init__(config, transport=transport)

    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key()}", "Content-Type": "application/json"}

    def _structured_output_format(self, response_format: dict[str, Any]) -> dict[str, Any]:
        """Translate the router's json_schema request into the provider's supported mode."""
        if self.config.structured_output_mode == "json_object":
            return {"type": "json_object"}
        normalized_format = dict(response_format)
        if normalized_format.get("type") == "json_schema":
            schema_document = dict(normalized_format.get("json_schema") or {})
            schema_document["strict"] = True
            normalized_format["json_schema"] = schema_document
        return normalized_format

    @staticmethod
    def _ensure_json_instruction(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Some json_object providers require the word 'json' in the prompt.

        If no message already mentions it, append a JSON-format instruction to
        the last user message so the request is accepted without corrupting the
        original prompt.
        """
        if any("json" in str(item.get("content", "")).lower() for item in messages):
            return messages
        for index in range(len(messages) - 1, -1, -1):
            if messages[index].get("role") == "user":
                amended = list(messages)
                amended[index] = {**messages[index], "content": f"{messages[index].get('content', '')}\n\nReturn your response as a single JSON object."}
                return amended
        return messages

    async def complete(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int | None = None, response_format: dict[str, Any] | None = None, tools: list[dict[str, Any]] | None = None) -> CompletionResult:
        body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature}
        # Unlimited call sites map to the provider's default output cap (the
        # OpenAI protocol has no "unlimited"); the reported effective cap keeps
        # the router's truncation self-heal able to double it. Mirrors the
        # anthropic adapter's policy.
        body["max_tokens"] = max_tokens if max_tokens is not None else int(getattr(self.config, "default_max_output_tokens", 65536))
        # Reasoning-budget cap (see _thinking_budget): config-first, env override.
        budget = _thinking_budget(self.config.thinking_budget)
        if budget is not None:
            body["thinking"] = {"type": "enabled", "budget_tokens": budget}
        if response_format:
            if self.config.structured_output_mode == "json_object":
                messages = self._ensure_json_instruction(messages)
            body["messages"] = messages
            body["response_format"] = self._structured_output_format(response_format)
        if tools:
            body["tools"] = tools
        payload, headers = await self.request_json("POST", "/chat/completions", headers=self.headers(), json_body=body)
        try:
            choice = payload["choices"][0]["message"]
            text = choice.get("content") or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Provider response omitted choices[0].message.", ErrorKind.SCHEMA, self.name) from exc
        raw = {"tool_calls": choice.get("tool_calls") or [], "finish_reason": (payload.get("choices") or [{}])[0].get("finish_reason"), "effective_max_tokens": body["max_tokens"]}
        return self.completion_result(text=str(text), model=str(payload.get("model") or model), usage=payload.get("usage") or {}, request_id=headers.get("x-request-id") or payload.get("id"), raw=raw)

    def stream(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int | None = None) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            await self._limiter.acquire()
            async with self._semaphore:
                try:
                    stream_body: dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature, "stream": True}
                    if max_tokens is not None:
                        stream_body["max_tokens"] = max_tokens
                    async with self.client.stream("POST", "/chat/completions", headers=self.headers(), json=stream_body) as response:
                        if response.is_error:
                            raise ProviderError(f"Provider returned HTTP {response.status_code}.", ErrorKind.SERVER, self.name, response.status_code)
                        async for line in response.aiter_lines():
                            if not line.startswith("data:"):
                                continue
                            data = line[5:].strip()
                            if data == "[DONE]":
                                return
                            payload = json.loads(data)
                            token = payload.get("choices", [{}])[0].get("delta", {}).get("content")
                            if token:
                                yield str(token)
                except ProviderError:
                    raise
                except Exception as exc:
                    raise ProviderError("Provider streaming failed.", ErrorKind.UNAVAILABLE, self.name) from exc
        return iterator()

    async def model_info(self, model: str) -> ModelInfo:
        if model:
            return await super().model_info(model)
        await self.request_json("GET", "/models", headers=self.headers())
        return await super().model_info(model)
