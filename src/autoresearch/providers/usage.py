"""Provider usage and user-supplied price calculations."""

from __future__ import annotations

from typing import Any


def estimate_cost(prompt_tokens: int | None, completion_tokens: int | None, pricing: dict[str, Any]) -> tuple[float | None, str | None]:
    """Estimate cost only when explicit user-maintained prices are present."""
    if prompt_tokens is None and completion_tokens is None:
        return None, None
    try:
        input_rate = float(pricing["input_per_million"])
        output_rate = float(pricing["output_per_million"])
        currency = str(pricing.get("currency") or "USD")
    except (KeyError, TypeError, ValueError):
        return None, None
    total = ((prompt_tokens or 0) * input_rate + (completion_tokens or 0) * output_rate) / 1_000_000
    return round(total, 12), currency
