"""Project API-budget enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from .errors import ErrorKind, ProviderError


class BudgetStore(Protocol):
    def get_project_budget(self, project_id: str) -> dict[str, float | str | None] | None: ...
    def provider_spend(self, project_id: str) -> float: ...


@dataclass(frozen=True)
class BudgetDecision:
    allowed: bool
    reason: str
    current_spend: float
    hard_stop: float | None


class ProviderBudgetGuard:
    def __init__(self, store: BudgetStore) -> None:
        self.store = store

    def evaluate(self, project_id: str, estimated_cost: float | None = None) -> BudgetDecision:
        budget = self.store.get_project_budget(project_id)
        spend = self.store.provider_spend(project_id)
        if not budget:
            return BudgetDecision(True, "No project API budget is configured.", spend, None)
        hard_stop_raw = budget.get("hard_stop")
        hard_stop = float(hard_stop_raw) if hard_stop_raw is not None else None
        projected = spend + (estimated_cost or 0.0)
        if hard_stop is not None and projected >= hard_stop:
            return BudgetDecision(False, f"Projected provider spend {projected:.6f} reaches hard stop {hard_stop:.6f}.", spend, hard_stop)
        return BudgetDecision(True, "Within configured project API budget.", spend, hard_stop)

    def require_allowed(self, project_id: str, estimated_cost: float | None = None) -> None:
        decision = self.evaluate(project_id, estimated_cost)
        if not decision.allowed:
            raise ProviderError(decision.reason, ErrorKind.BUDGET)
