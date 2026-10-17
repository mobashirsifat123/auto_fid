"""Tests for the manuscript defense loop (objection routing + defensibility)."""

from __future__ import annotations

import pytest

from autoresearch.application.defense import (
    Objection,
    ObjectionCategory,
    ReviewReport,
    defensibility,
    plan_fixes,
)


def _report(*objections: Objection) -> ReviewReport:
    return ReviewReport(
        round_index=1, reviewer_identity="critic-fake", objections=objections
    )


def test_routing_by_category() -> None:
    report = _report(
        Objection(
            "o1", ObjectionCategory.CLARITY, 1, "method section is hard to follow"
        ),
        Objection("o2", ObjectionCategory.ANALYSIS, 2, "no effect size reported"),
        Objection(
            "o3",
            ObjectionCategory.MISSING_EXPERIMENT,
            3,
            "missing ablation of module X",
        ),
        Objection("o4", ObjectionCategory.LIMITATION, 1, "single dataset"),
    )
    plan = plan_fixes(report)
    assert plan.routes == {
        "o1": "revise_prose",
        "o2": "reanalyze",
        "o3": "new_experiment",
        "o4": "disclose_limitation",
    }
    assert plan.requires_new_experiments


def test_clarity_promoted_by_experiment_keywords() -> None:
    report = _report(
        Objection(
            "o1", ObjectionCategory.CLARITY, 2, "the baseline comparison is unclear"
        ),
        Objection("o2", ObjectionCategory.CLARITY, 2, "no seed variance discussion"),
    )
    plan = plan_fixes(report)
    assert plan.routes["o1"] == "new_experiment"  # 'baseline' keyword → real gap
    assert plan.routes["o2"] == "new_experiment"  # 'seed' keyword


def test_severity_bounds_enforced() -> None:
    with pytest.raises(ValueError):
        Objection("x", ObjectionCategory.CLARITY, 4, "bad")


def test_addressed_requires_evidence_ref() -> None:
    from autoresearch.application.defense import Resolution

    with pytest.raises(ValueError):
        Resolution("o1", "ADDRESSED", "reanalysis")  # no evidence ref → refuse


def test_defensible_when_all_closed_with_evidence() -> None:
    objections = (
        Objection("o1", ObjectionCategory.MISSING_EXPERIMENT, 3, "missing ablation"),
        Objection("o2", ObjectionCategory.LIMITATION, 1, "single dataset"),
    )
    from autoresearch.application.defense import Resolution

    report = defensibility(
        (
            Resolution("o1", "ADDRESSED", "experiment", evidence_ref="run:abc123"),
            Resolution("o2", "DISCLOSED", "limitations_section"),
        ),
        rounds=2,
        terminated_by="all_closed",
        objections=objections,
    )
    assert report.verdict == "DEFENSIBLE"
    assert report.evidence_map["o1"] == "ADDRESSED:experiment:run:abc123"


def test_not_defensible_when_blocking_open_or_disclosure_only() -> None:
    from autoresearch.application.defense import Resolution

    objections = (
        Objection("o1", ObjectionCategory.MISSING_EXPERIMENT, 3, "missing ablation"),
        Objection("o2", ObjectionCategory.ANALYSIS, 3, "no significance test"),
        Objection(
            "o3", ObjectionCategory.LIMITATION, 3, "cannot test on other hardware"
        ),
    )
    report = defensibility(
        (
            Resolution(
                "o1", "DISCLOSED", "limitations_section"
            ),  # blocking → disclosure without justification
            Resolution("o2", "OPEN", "budget_exhausted"),
            Resolution("o3", "DISCLOSED", "irreducible"),  # justified irreducible
        ),
        rounds=3,
        terminated_by="max_rounds",
        objections=objections,
    )
    assert report.verdict == "NOT_DEFENSIBLE"
    assert set(report.open_blocking) == {"o1", "o2"}
    assert "o3" in report.evidence_map  # justified disclosure stands
