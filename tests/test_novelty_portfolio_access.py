"""Regression: two-tier novelty portfolios must be readable through every accessor.

Live bug (2026-10-16, FIRE-Bench project): a portfolio with only low_risk +
balanced tiers is written to novelty_portfolio_records (+members), but the
legacy novelty_portfolios row is written ONLY when all three tiers exist.
Every consumer (CLI novelty read-back, workflow hypotheses reuse, manuscript
lineage) read the legacy table directly, so a legitimate two-tier portfolio
made `skill novelty` fail with "portfolio_not_found" and would have crashed
the workflow design stage. These tests pin the new accessors.
"""

from __future__ import annotations

from uuid import uuid4

from autoresearch.domain.models import (
    ArtifactProvenance,
    CandidateStatus,
    NoveltyCandidate,
    NoveltyPortfolio,
    PipelineKind,
)
from autoresearch.storage.repository import ResearchRepository


def _candidate(project_id: str, run_id: str, gap_id: str, name: str) -> NoveltyCandidate:
    return NoveltyCandidate(
        id=uuid4().hex,
        project_id=project_id,
        run_id=run_id,
        gap_id=gap_id,
        strategy="test-strategy",
        method=f"method-{name}",
        causal_mechanism="test mechanism",
        target_limitation="test limitation",
        difference_from_existing="test difference",
        expected_effect="test effect",
        falsification_condition="test falsification",
        minimum_validating_experiment="test experiment",
        required_ablations=(),
        implementation_complexity="low",
        expected_compute="low",
        risks_and_objections=(),
        status=CandidateStatus.GENERATED,
        provenance=ArtifactProvenance.create(
            project_id=project_id, run_id=run_id, source_evidence=(),
            provider="deterministic", model="test", prompt_version="test", content=name,
        ),
    )


def _portfolio(project_id: str, run_id: str, tiers: dict[str, str]) -> NoveltyPortfolio:
    return NoveltyPortfolio(
        id=uuid4().hex,
        project_id=project_id,
        run_id=run_id,
        low_risk_candidate_id=tiers.get("low_risk"),
        balanced_candidate_id=tiers.get("balanced"),
        high_risk_candidate_id=tiers.get("high_risk"),
        selection_reasons={"low_risk": "Highest feasibility."},
        provenance=ArtifactProvenance.create(
            project_id=project_id, run_id=run_id, source_evidence=tuple(tiers.values()),
            provider="deterministic", model="selector", prompt_version="test", content=tiers,
        ),
    )


def test_two_tier_portfolio_is_readable_by_id_latest_and_list(tmp_path):
    repository = ResearchRepository(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    project = repository.create_project("portfolio access regression")
    run = repository.create_run(project.id, PipelineKind.NOVELTY, {})

    low = _candidate(project.id, run.id, "gap-1", "low")
    balanced = _candidate(project.id, run.id, "gap-1", "balanced")
    repository.save_novelty_candidate(low)
    repository.save_novelty_candidate(balanced)

    portfolio = _portfolio(project.id, run.id, {"low_risk": low.id, "balanced": balanced.id})
    repository.save_novelty_portfolio(portfolio)

    by_id = repository.get_novelty_portfolio_by_id(portfolio.id)
    assert by_id is not None
    assert (by_id.low_risk_candidate_id, by_id.balanced_candidate_id, by_id.high_risk_candidate_id) == (low.id, balanced.id, None)
    assert by_id.selection_reasons == {"low_risk": "Highest feasibility."}

    latest = repository.latest_novelty_portfolio(project.id)
    assert latest is not None and latest.id == portfolio.id

    listed = repository.novelty_portfolios_for_project(project.id)
    assert [item.id for item in listed] == [portfolio.id]


def test_three_tier_portfolio_is_deduped_across_schemas(tmp_path):
    repository = ResearchRepository(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    project = repository.create_project("portfolio dedupe regression")
    run = repository.create_run(project.id, PipelineKind.NOVELTY, {})

    candidates = {tier: _candidate(project.id, run.id, "gap-1", tier) for tier in ("low_risk", "balanced", "high_risk")}
    for candidate in candidates.values():
        repository.save_novelty_candidate(candidate)

    portfolio = _portfolio(project.id, run.id, {tier: candidate.id for tier, candidate in candidates.items()})
    repository.save_novelty_portfolio(portfolio)

    # Three-tier portfolios are written to BOTH schemas; the accessor must not
    # return the legacy row as a second portfolio.
    listed = repository.novelty_portfolios_for_project(project.id)
    assert [item.id for item in listed] == [portfolio.id]

    found = repository.get_novelty_portfolio_by_id(portfolio.id)
    assert found is not None
    assert found.high_risk_candidate_id == candidates["high_risk"].id