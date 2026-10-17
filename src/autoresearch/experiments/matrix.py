"""Experiment matrix: declared ablation/robustness cells become runnable,
trackable, and honestly reported (2026-10-16).

Fixes the declared-but-never-run gap: the spec planner already declares
``ablations``/``robustness_tests`` (from the novelty skill's
``required_ablations``), but the executor runs only the main comparison, so
``StatisticalAnalysis.ablation_contributions`` was always empty and reports
said ``ABLATION_NOT_RUN``.

Design:
- :meth:`ExperimentMatrix.from_spec` — one MAIN cell plus one cell per
  declared ablation/robustness entry, each seeded with the spec's
  ``random_seeds`` (the same seed discipline as the main arm).
- :meth:`ExperimentMatrix.completeness` — declared vs executed ledger:
  every declared cell is either RUN or MISSING, visibly. A missing cell is
  never silently dropped from the paper-facing completeness report.
- :meth:`ExperimentMatrix.ablation_contributions` — fills the analysis
  record's always-empty mapping from REAL per-cell results only:
  ``contribution[label] = main_metric - cell_metric`` for the spec's primary
  metric; absent cells are absent entries (never zeros).

Karpathy-autoresearch protocol note (design attribution, karpathy/autoresearch
2026, MIT-style single-GPU loop): fair comparison requires the baseline,
candidate, and every ablation cell to run under an IDENTICAL fixed wall-clock
budget. The matrix records the spec's ``compute_budget`` per cell and the
completeness report fails closed if cells were run under heterogeneous
budgets (declared in the ledger as BUDGET_MISMATCH), because a cheap ablation
compared against a budgeted main run is not evidence.

Execution of a cell still goes through the existing approval-gated path
(implement → verify → ``experiment approve`` bound to the spec checksum →
execute → ingest): this module plans and tracks; it never self-authorizes.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["ExperimentCell", "ExperimentMatrix"]

_KIND_MAIN = "main"
_KIND_ABLATION = "ablation"
_KIND_ROBUSTNESS = "robustness"


def _slug(text: str) -> str:
    lowered = re.sub(r"[^a-z0-9]+", "-", text.strip().lower())
    return lowered.strip("-")[:48] or "unnamed"


@dataclass(frozen=True)
class ExperimentCell:
    cell_id: str
    kind: str  # main | ablation | robustness
    label: str
    seeds: tuple[int, ...]
    compute_budget: str = ""

    @property
    def declared(self) -> bool:
        return True  # matrix cells exist only if declared by the spec


class ExperimentMatrix:
    """The full declared experimental program for one spec."""

    def __init__(
        self, cells: tuple[ExperimentCell, ...], primary_metric: str | None
    ) -> None:
        self._cells = cells
        self._primary_metric = primary_metric

    @classmethod
    def from_spec(cls, spec: Any) -> ExperimentMatrix:
        seeds = tuple(int(s) for s in (spec.random_seeds or ()))
        budget = str(getattr(spec, "compute_budget", "") or "")
        primary = spec.metrics[0] if spec.metrics else None
        cells = [
            ExperimentCell(
                cell_id="main",
                kind=_KIND_MAIN,
                label="baseline vs candidate (main comparison)",
                seeds=seeds,
                compute_budget=budget,
            )
        ]
        for label in spec.ablations or ():
            cells.append(
                ExperimentCell(
                    cell_id=f"ablation:{_slug(label)}",
                    kind=_KIND_ABLATION,
                    label=str(label),
                    seeds=seeds,
                    compute_budget=budget,
                )
            )
        for label in spec.robustness_tests or ():
            cells.append(
                ExperimentCell(
                    cell_id=f"robustness:{_slug(label)}",
                    kind=_KIND_ROBUSTNESS,
                    label=str(label),
                    seeds=seeds,
                    compute_budget=budget,
                )
            )
        return cls(tuple(cells), primary)

    @property
    def cells(self) -> tuple[ExperimentCell, ...]:
        return self._cells

    @property
    def primary_metric(self) -> str | None:
        return self._primary_metric

    def completeness(
        self,
        executed_cell_ids: Iterable[str],
        *,
        run_budgets: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        """Declared-vs-executed ledger. Every cell is RUN or MISSING — a
        missing cell is visible, never dropped. Budget heterogeneity across
        executed cells is flagged as BUDGET_MISMATCH (fail-closed)."""
        executed = set(executed_cell_ids)
        run_budgets = run_budgets or {}
        rows = []
        missing = 0
        for cell in self._cells:
            ran = cell.cell_id in executed
            if not ran:
                missing += 1
            rows.append(
                {
                    "cell_id": cell.cell_id,
                    "kind": cell.kind,
                    "label": cell.label,
                    "seeds": list(cell.seeds),
                    "status": "RUN" if ran else "MISSING",
                }
            )
        declared_budgets = {
            cid: budget
            for cid, budget in run_budgets.items()
            if cid in executed and budget
        }
        budget_mismatch = (
            bool(declared_budgets) and len(set(declared_budgets.values())) > 1
        )
        return {
            "status": "COMPLETE"
            if missing == 0 and not budget_mismatch
            else "INCOMPLETE",
            "declared_cells": len(rows),
            "executed_cells": len(rows) - missing,
            "missing_cells": missing,
            "budget_mismatch": budget_mismatch,
            "cells": rows,
        }

    def ablation_contributions(
        self,
        main_value: float | None,
        cell_values: Mapping[str, float],
    ) -> dict[str, float]:
        """Per-cell contribution for the analysis record, from REAL results
        only: contribution = main − cell on the primary metric. Missing cells
        are absent entries — never zeros, never estimates."""
        if main_value is None or not self._primary_metric:
            return {}
        contributions: dict[str, float] = {}
        for cell in self._cells:
            if cell.kind == _KIND_MAIN:
                continue
            if cell.label in cell_values:
                contributions[cell.label] = round(
                    float(main_value) - float(cell_values[cell.label]), 6
                )
        return contributions
