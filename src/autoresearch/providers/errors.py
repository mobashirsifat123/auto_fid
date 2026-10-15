"""Normalized provider errors used by retry and fallback policies."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class ErrorKind(StrEnum):
    AUTHENTICATION = "authentication"
    INVALID_REQUEST = "invalid_request"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"
    SERVER = "server_error"
    CONTENT_POLICY = "content_policy"
    UNSUPPORTED = "unsupported_capability"
    SCHEMA = "schema_validation"
    BUDGET = "budget_exceeded"
    UNKNOWN = "unknown"


@dataclass
class ProviderError(RuntimeError):
    message: str
    kind: ErrorKind = ErrorKind.UNKNOWN
    provider: str = ""
    status_code: int | None = None
    request_id: str | None = None
    retry_after: float | None = None
    details: dict[str, Any] | None = None

    def __str__(self) -> str:
        return f"{self.provider}: {self.message}" if self.provider else self.message

    @property
    def retryable(self) -> bool:
        return self.kind in {ErrorKind.RATE_LIMIT, ErrorKind.TIMEOUT, ErrorKind.UNAVAILABLE, ErrorKind.SERVER}

    @property
    def fallback_allowed(self) -> bool:
        return self.retryable


def kind_for_status(status: int) -> ErrorKind:
    if status in {401, 403}:
        return ErrorKind.AUTHENTICATION
    if status == 429:
        return ErrorKind.RATE_LIMIT
    if status in {408, 504}:
        return ErrorKind.TIMEOUT
    if status in {500, 502, 503}:
        return ErrorKind.SERVER
    if status in {400, 404, 409, 413, 422}:
        return ErrorKind.INVALID_REQUEST
    return ErrorKind.UNKNOWN
