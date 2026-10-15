"""Cross-run review reuse: a verbatim-verified review of the same paper under
the same prompt version is reused instead of re-paid; --fresh forces
re-extraction. Also covers default budget ceilings from the environment."""

from __future__ import annotations

import pytest

import test_workflow_coordination as fixtures


@pytest.mark.anyio
async def test_second_review_run_reuses_verified_reviews(tmp_path):
    repository, workflow, _, seed, _ = fixtures._harness(tmp_path)
    first = await workflow.run(seed=str(seed), until="review")

    usage_after_first = repository.row("SELECT COUNT(*) AS n FROM provider_usage")["n"]
    usage_after_first_paper_reviews = repository.row(
        "SELECT COUNT(*) AS n FROM provider_usage WHERE operation LIKE 'paper_review%'"
    )["n"]
    second = await workflow.run(project_id=first["project_id"], until="review")
    # Second run re-ran only gap analysis / clustering (papers all reused).
    review_stage = next(s for s in second["stages"] if s["stage"] == "review")
    assert review_stage["status"] in ("completed", "reused")
    direct = await workflow.review.review(first["project_id"], top_k=3)
    assert direct.reused > 0, "re-review must report reused reviews"
    assert direct.reviewed >= 3
    # Reuse means the paper-review extractor was NOT billed again: the count
    # of billed paper-review operations is unchanged by the re-review run.
    reviews_billed = repository.row(
        "SELECT COUNT(*) AS n FROM provider_usage WHERE operation LIKE 'paper_review%'"
    )["n"]
    assert reviews_billed == usage_after_first_paper_reviews, "paper reviews must not be re-billed per re-run"


@pytest.mark.anyio
async def test_fresh_review_reextracts(tmp_path):
    repository, workflow, _, seed, _ = fixtures._harness(tmp_path)
    first = await workflow.run(seed=str(seed), until="review")
    before = repository.row(
        "SELECT COUNT(*) AS n FROM provider_usage WHERE operation LIKE 'paper_review%'"
    )["n"]
    outcome = await workflow.review.review(first["project_id"], top_k=3, fresh=True)
    after = repository.row(
        "SELECT COUNT(*) AS n FROM provider_usage WHERE operation LIKE 'paper_review%'"
    )["n"]
    assert outcome.reused == 0
    assert after > before, "--fresh must re-bill every paper review"


def test_default_budget_from_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTORESEARCH_DEFAULT_MAX_TOKENS", "123456")
    monkeypatch.setenv("AUTORESEARCH_DEFAULT_MAX_COST_USD", "7.5")
    from autoresearch.application.config import load_settings

    settings = load_settings()
    assert settings.default_max_tokens == 123456
    assert settings.default_max_cost_usd == 7.5
    monkeypatch.delenv("AUTORESEARCH_DEFAULT_MAX_TOKENS")
    monkeypatch.delenv("AUTORESEARCH_DEFAULT_MAX_COST_USD", raising=False)
    settings = load_settings()
    assert settings.default_max_tokens is None and settings.default_max_cost_usd is None


def test_guard_default_applies_when_project_budget_unset():
    import test_token_ceiling
    from autoresearch.providers.budget import ProviderBudgetGuard

    guard = ProviderBudgetGuard(
        test_token_ceiling._Store({"hard_stop": None, "token_ceiling": None}, 0.0, 50),
        default_token_ceiling=1_000_000, default_hard_stop=10.0,
    )
    decision = guard.evaluate("p")
    assert decision.token_ceiling == 1_000_000 and decision.hard_stop == 10.0
    assert decision.allowed

    # an explicit per-project value overrides the default
    overridden = ProviderBudgetGuard(
        test_token_ceiling._Store({"hard_stop": None, "token_ceiling": 2}, 0.0, 5),
        default_token_ceiling=1_000_000,
    )
    assert not overridden.evaluate("p").allowed
