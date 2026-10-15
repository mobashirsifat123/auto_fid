"""Google Gemini generateContent adapter using header-based credentials."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from ..errors import ErrorKind, ProviderError
from .base import CompletionResult
from .http_base import HTTPProviderBase


class GeminiProvider(HTTPProviderBase):
    def headers(self) -> dict[str, str]:
        return {"x-goog-api-key": self.api_key(), "Content-Type": "application/json"}

    async def complete(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int = 2048, response_format: dict[str, Any] | None = None, tools: list[dict[str, Any]] | None = None) -> CompletionResult:
        contents = [{"role": "model" if item.get("role") == "assistant" else "user", "parts": [{"text": str(item.get("content", ""))}]} for item in messages]
        body: dict[str, Any] = {"contents": contents, "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
        if response_format:
            body["generationConfig"]["responseMimeType"] = "application/json"
            schema_document = response_format.get("json_schema") or {}
            if schema_document.get("schema"):
                body["generationConfig"]["responseJsonSchema"] = schema_document["schema"]
        if tools:
            body["tools"] = [{"function_declarations": [_gemini_tool(tool) for tool in tools]}]
        payload, headers = await self.request_json("POST", f"/v1beta/models/{model}:generateContent", headers=self.headers(), json_body=body)
        try:
            parts = payload["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError("Gemini response omitted candidate content.", ErrorKind.SCHEMA, self.name) from exc
        text = "".join(str(part.get("text", "")) for part in parts)
        tool_calls = [{"name": part["functionCall"].get("name"), "input": part["functionCall"].get("args") or {}} for part in parts if isinstance(part.get("functionCall"), dict)]
        usage = payload.get("usageMetadata") or {}
        normalized = {"prompt_tokens": usage.get("promptTokenCount"), "completion_tokens": usage.get("candidatesTokenCount")}
        return self.completion_result(text=text, model=model, usage=normalized, request_id=headers.get("x-request-id"), raw={"tool_calls": tool_calls})

    def stream(self, *, model: str, messages: list[dict[str, Any]], temperature: float = 0.2, max_tokens: int = 2048) -> AsyncIterator[str]:
        async def iterator() -> AsyncIterator[str]:
            result = await self.complete(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
            yield result.text
        return iterator()


def _gemini_tool(tool: dict[str, Any]) -> dict[str, Any]:
    function = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return {"name": function.get("name"), "description": function.get("description") or "", "parameters": function.get("parameters") or {"type": "object"}}
