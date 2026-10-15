"""Secret redaction at logging and error boundaries."""

from __future__ import annotations

import os
import re
from typing import Any

_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+"),
    re.compile(r"(?i)((?:api[-_ ]?key|token|secret|password)\s*[:=]\s*)[^\s,;&#]+"),
    re.compile(r"(?i)(https?://)[^/@\s]+:[^/@\s]+@"),
    re.compile(r"(?i)([?&](?:access_token|api_key|key|token|secret|password)=)[^&#\s]+"),
)


def redact_text(value: str) -> str:
    output = value
    for pattern in _PATTERNS:
        output = pattern.sub(r"\1[REDACTED]", output)
    for name, secret in os.environ.items():
        if secret and len(secret) >= 8 and any(marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD")):
            output = output.replace(secret, "[REDACTED]")
    return output


def redact(value: Any) -> Any:
    if isinstance(value, str):
        return redact_text(value)
    if isinstance(value, dict):
        return {key: "[REDACTED]" if str(key).lower() in {"api_key", "token", "secret", "password", "authorization"} else redact(item) for key, item in value.items()}
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value
