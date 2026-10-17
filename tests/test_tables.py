"""Tests for the deterministic result-table engine (upgrade M1)."""

from __future__ import annotations

from types import SimpleNamespace

from autoresearch.application.tables import build_result_tables


def _analysis(
    rows: list[dict], tests: dict | None = None, gates: dict | None = None
) -> SimpleNamespace:
    return SimpleNamespace(
        proposed_metrics={"accuracy": 0.9},
        baseline_metrics={"accuracy": 0.8},
        effect_size={"accuracy": 0.1},
        confidence_intervals={"accuracy": (0.05, 0.15)},
        seed_level_results=tuple(rows),
        statistical_tests=tests or {},
        decision_gates=gates or {"accuracy_gate": True},
    )


def test_mean_std_and_p_rendered() -> None:
    rows = [
        {"seed": 42, "baseline": {"accuracy": 0.80}, "candidate": {"accuracy": 0.90}},
        {"seed": 101, "baseline": {"accuracy": 0.82}, "candidate": {"accuracy": 0.92}},
        {"seed": 202, "baseline": {"accuracy": 0.78}, "candidate": {"accuracy": 0.88}},
    ]
    tests = {
        "accuracy": {"status": "VALID", "test": "wilcoxon_signed_rank", "p_value": 0.25}
    }
    md, tex = build_result_tables(_analysis(rows, tests))
    # mean±std computed from the seed rows (sample std of 0.80/0.82/0.78 = 0.0200)
    assert "0.8000 ±0.0200" in md
    assert "0.9000 ±0.0200" in md
    assert "| +0.1000 |" in md  # Δ mean from the analysis record
    assert "| 3 |" in md and "| PASS |" in md
    assert "| 0.2500 |" in md  # exact p surfaced
    # Cohen's d: constant differences (0.10 each) → undefined → em dash (never
    # a fabricated huge value from floating-point residue)
    assert "1903295219350126" not in md

    assert "—" in md.split("\n")[2]
    assert "\\pm" in tex and "\\toprule" in tex and "Wilcoxon" in tex


def test_single_seed_std_is_em_dash_not_zero() -> None:
    rows = [{"seed": 42, "baseline": {"accuracy": 0.8}, "candidate": {"accuracy": 0.9}}]
    md, _ = build_result_tables(_analysis(rows))
    row = md.split("\n")[2]
    assert "±" not in row  # undefined std rendered as —, never fabricated 0.0000
    assert "| 1 |" in row


def test_missing_test_and_gate_render_dashes() -> None:
    rows = [
        {"seed": 1, "baseline": {"accuracy": 0.8}, "candidate": {"accuracy": 0.9}},
        {"seed": 2, "baseline": {"accuracy": 0.8}, "candidate": {"accuracy": 0.91}},
    ]
    md, _ = build_result_tables(_analysis(rows, tests={}, gates={}))
    row = md.split("\n")[2]
    assert "| — |" in row  # p missing, gate absent → dashes, honest


def test_cohens_d_computed_when_variance_exists() -> None:
    rows = [
        {"seed": 1, "baseline": {"accuracy": 0.80}, "candidate": {"accuracy": 0.90}},
        {"seed": 2, "baseline": {"accuracy": 0.78}, "candidate": {"accuracy": 0.95}},
    ]
    tests = {"accuracy": {"status": "VALID", "p_value": 0.5}}
    md, _ = build_result_tables(_analysis(rows, tests))
    row = md.split("\n")[2]
    # diffs = 0.10, 0.17 → avg 0.135, std 0.035 → d = 0.135/0.035 ≈ 3.857
    assert "+3.857" in row or "+3.858" in row


def test_metric_key_resolution_matches_analyzer() -> None:
    # Resolution semantics mirror the analyzer: parenthetical spec descriptors
    # ("Accuracy (PRIMARY: higher)") resolve to bare document keys, case- and
    # qualifier-insensitively.
    rows = [
        {"seed": 1, "baseline": {"accuracy": 0.8}, "candidate": {"accuracy": 0.9}},
        {"seed": 2, "baseline": {"accuracy": 0.82}, "candidate": {"accuracy": 0.92}},
    ]
    analysis = _analysis(
        rows, tests={"Accuracy (PRIMARY: higher)": {"status": "VALID", "p_value": 0.5}}
    )
    analysis.proposed_metrics = {"Accuracy (PRIMARY: higher)": 0.9}
    analysis.statistical_tests = {
        "Accuracy (PRIMARY: higher)": {"status": "VALID", "p_value": 0.5}
    }
    md, _ = build_result_tables(analysis)
    assert "0.8100" in md and "0.9100" in md
