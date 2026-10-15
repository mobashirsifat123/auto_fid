"""Project total-token ceiling: aggregate spend guard regardless of pricing,
preserving-on-None budget semantics, and reasoning-appropriate timeouts.
"""

from __future__ import annotations

import pytest

from autoresearch.providers.budget import BudgetDecision, ProviderBudgetGuard
from autoresearch.providers.errors import ErrorKind, ProviderError
from autoresearch.storage.repository import ResearchRepository


class _Store:
    def __init__(self, budget: dict | None, spend: float, tokens: int):
        self._budget, self._spend, self._tokens = budget, spend, tokens

    def get_project_budget(self, project_id: str):
        return self._budget

    def provider_spend(self, project_id: str) -> float:
        return self._spend

    def provider_tokens(self, project_id: str) -> int:
        return self._tokens


def test_token_ceiling_blocks_when_reached():
    guard = ProviderBudgetGuard(_Store({"hard_stop": None, "token_ceiling": 1_000_000}, 0.0, 1_000_000))
    decision = guard.evaluate("p")
    assert not decision.allowed
    assert "Token ceiling reached" in decision.reason and "1000000" in decision.reason
    with pytest.raises(ProviderError) as excinfo:
        guard.require_allowed("p")
    assert excinfo.value.kind == ErrorKind.BUDGET


def test_token_ceiling_allows_below_and_absent():
    assert ProviderBudgetGuard(_Store({"token_ceiling": 100}, 0.0, 99)).evaluate("p").allowed
    assert ProviderBudgetGuard(_Store(None, 0.0, 10_000_000)).evaluate("p").allowed


def test_cost_hard_stop_still_enforced():
    guard = ProviderBudgetGuard(_Store({"hard_stop": 1.0, "token_ceiling": None}, 0.9, 0), )
    assert not guard.evaluate("p", estimated_cost=0.2).allowed


def test_budget_decision_defaults_backfilled():
    decision = BudgetDecision(True, "ok", 0.0, None)
    assert decision.consumed_tokens == 0 and decision.token_ceiling is None


def test_set_project_budget_preserves_unset_dimension(tmp_path):
    repository = ResearchRepository(tmp_path / "s.db", tmp_path / "a")
    repository.migrate()
    project = repository.create_project("Budgets")
    repository.set_project_budget(project.id, hard_stop=5.0)
    repository.set_project_budget(project.id, token_ceiling=1_000)  # cost untouched
    budget = repository.get_project_budget(project.id)
    assert budget["hard_stop"] == 5.0 and budget["token_ceiling"] == 1_000
    repository.set_project_budget(project.id, hard_stop=None)  # None preserves both
    budget = repository.get_project_budget(project.id)
    assert budget["hard_stop"] == 5.0 and budget["token_ceiling"] == 1_000


def test_provider_tokens_sums_usage(tmp_path):
    repository = ResearchRepository(tmp_path / "s.db", tmp_path / "a")
    repository.migrate()
    project = repository.create_project("Tokens")
    from autoresearch.domain.models import ProviderUsage
    for index, (prompt, completion) in enumerate(((100, 200), (10, None), (None, None))):
        repository.record_provider_usage(ProviderUsage(
            id=f"u{index}", project_id=project.id, run_id=None,
            operation="test", role="research_reasoning", provider="p", model="m",
            prompt_tokens=prompt, completion_tokens=completion,
            estimated_cost=0.0, currency="USD", request_id=None, latency_ms=1.0,
        ))
    assert repository.provider_tokens(project.id) == 310


def test_reasoning_model_timeouts():
    from autoresearch.providers.config import ProviderConfig
    from autoresearch.providers.capabilities import Capability
    config = ProviderConfig(name="p", kind="openai_compatible", base_url="u", api_key_env="K", capabilities=frozenset(Capability))
    assert config.read_timeout >= 300, "reasoning models think for minutes; short read timeouts cause retry→double-spend"
    assert config.total_timeout >= 600


@pytest.mark.anyio
async def test_workflow_enforces_token_ceiling_end_to_end(tmp_path):
    """A tiny ceiling must stop the pipeline: every model call after the
    aggregate is reached is denied, the denials are recorded per-paper, and
    the run fails — never silently continues spending."""
    import test_workflow_coordination as fixtures

    repository, workflow, _, seed, _ = fixtures._harness(tmp_path)
    with pytest.raises(RuntimeError) as excinfo:
        await workflow.run(seed=str(seed), until="review", max_total_tokens=1)
    assert "evidence-backed reviews" in str(excinfo.value)
    failures = repository.review_failures(
        repository.row("SELECT id AS run_id FROM pipeline_runs WHERE kind='deep_literature_review' ORDER BY rowid DESC LIMIT 1")["run_id"]
    )
    assert failures and all("Token ceiling reached" in str(item.get("reason")) for item in failures[:2])
