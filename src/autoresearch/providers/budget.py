"""Project API-budget enforcement: USD hard stop and total-token ceiling.

Both are durable project-scoped guards enforced per provider call. The token
ceiling bounds aggregate spend in tokens regardless of pricing availability;
the USD hard stop bounds it in cost. A single in-flight call is bounded by the
per-model ``max_output_tokens`` cap (with truncation self-heal); these
ceilings stop *subsequent* calls once the aggregate is reached.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .errors import ErrorKind, ProviderError


class BudgetStore(Protocol):
    def get_project_budget(self, project_id: str) -> dict[str, float | str | int | None] | None: ...
    def provider_spend(self, project_id: str) -> float: ...
    def provider_tokens(self, project_id: str) -> int: ...


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str
    current_spend: float
    hard_stop: float | None
    consumed_tokens: int = 0
    token_ceiling: int | None = None


class ProviderBudgetGuard:
    def __init__(self, store: BudgetStore, *, default_token_ceiling: int | None = None, default_hard_stop: float | None = None) -> None:
        self.store = store
        # Defaults apply to every project without an explicit budget: a
        # per-project value (set via --max-tokens/--max-cost-usd) always
        # overrides them; unset means unbounded for that dimension.
        self.default_token_ceiling = default_token_ceiling
        self.default_hard_stop = default_hard_stop

    def evaluate(self, project_id: str, estimated_cost: float | None = None) -> BudgetDecision:
        budget = self.store.get_project_budget(project_id) or {}
        spend = self.store.provider_spend(project_id)
        tokens = self.store.provider_tokens(project_id)
        hard_stop_raw = budget.get("hard_stop")
        hard_stop = float(hard_stop_raw) if hard_stop_raw is not None else self.default_hard_stop
        ceiling_raw = budget.get("token_ceiling")
        ceiling = int(ceiling_raw) if ceiling_raw is not None else self.default_token_ceiling

        if ceiling is not None and tokens >= ceiling:
            return BudgetDecision(
                False,
                f"Token ceiling reached: {tokens} tokens consumed of {ceiling} allowed. "
                "Raise --max-tokens or start a new project to continue.",
                spend, hard_stop, tokens, ceiling,
            )
        if hard_stop is not None:
            projected = spend + (estimated_cost or 0.0)
            if projected >= hard_stop:
                return BudgetDecision(
                    False,
                    f"Projected provider spend {projected:.6f} reaches hard stop {hard_stop:.6f}.",
                    spend, hard_stop, tokens, ceiling,
                )
        return BudgetDecision(True, "Within configured project API budget.", spend, hard_stop, tokens, ceiling)

    def require_allowed(self, project_id: str, estimated_cost: float | None = None) -> None:
        decision = self.evaluate(project_id, estimated_cost)
        if not decision.allowed:
            raise ProviderError(decision.reason, ErrorKind.BUDGET)
