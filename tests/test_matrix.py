"""Tests for the experiment matrix (declared ablation/robustness cells)."""

from __future__ import annotations

from types import SimpleNamespace

from autoresearch.experiments.matrix import ExperimentMatrix


def _spec() -> SimpleNamespace:
    return SimpleNamespace(
        metrics=["accuracy", "f1"],
        random_seeds=[42, 101, 202],
        ablations=("without attention module", "without data augmentation"),
        robustness_tests=("label noise 10%",),
        compute_budget="fixed 10 minutes wall-clock",
    )


def test_matrix_declares_every_cell_from_spec() -> None:
    matrix = ExperimentMatrix.from_spec(_spec())
    kinds = [c.kind for c in matrix.cells]
    assert kinds == ["main", "ablation", "ablation", "robustness"]
    assert all(c.seeds == (42, 101, 202) for c in matrix.cells)
    assert all(c.compute_budget == "fixed 10 minutes wall-clock" for c in matrix.cells)
    assert matrix.primary_metric == "accuracy"


def test_completeness_ledger_missing_cells_visible() -> None:
    matrix = ExperimentMatrix.from_spec(_spec())
    report = matrix.completeness(["main"])  # only the main comparison ran
    assert report["status"] == "INCOMPLETE"
    assert report["missing_cells"] == 3
    missing = [c for c in report["cells"] if c["status"] == "MISSING"]
    assert {c["kind"] for c in missing} == {"ablation", "robustness"}


def test_completeness_passes_when_all_cells_run() -> None:
    matrix = ExperimentMatrix.from_spec(_spec())
    all_ids = [c.cell_id for c in matrix.cells]
    report = matrix.completeness(all_ids)
    assert report["status"] == "COMPLETE"
    assert report["missing_cells"] == 0


def test_budget_mismatch_fails_closed() -> None:
    matrix = ExperimentMatrix.from_spec(_spec())
    all_ids = [c.cell_id for c in matrix.cells]
    budgets = {cid: ("10 min" if i == 0 else "2 min") for i, cid in enumerate(all_ids)}
    report = matrix.completeness(all_ids, run_budgets=budgets)
    assert report["budget_mismatch"] is True
    assert (
        report["status"] == "INCOMPLETE"
    )  # cheap ablation vs budgeted main ≠ evidence


def test_ablation_contributions_real_values_only() -> None:
    matrix = ExperimentMatrix.from_spec(_spec())
    contributions = matrix.ablation_contributions(
        0.92, {"without attention module": 0.90}
    )  # one ablation cell never ran
    assert contributions == {"without attention module": 0.02}
    # missing cell is an absent entry — never a zero or an estimate


def test_ablation_contributions_require_main_and_metric() -> None:
    matrix = ExperimentMatrix.from_spec(_spec())
    assert matrix.ablation_contributions(None, {"without attention module": 0.9}) == {}
    empty_spec = SimpleNamespace(
        metrics=[],
        random_seeds=[1],
        ablations=("x",),
        robustness_tests=(),
        compute_budget="",
    )
    assert (
        ExperimentMatrix.from_spec(empty_spec).ablation_contributions(0.9, {"x": 0.8})
        == {}
    )
