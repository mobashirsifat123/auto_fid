"""Deterministic result-table engine (upgrade M1, 2026-10-16).

Publication-grade tables derived *only* from the StatisticalAnalysis record:

- ``mean ± std`` over seeds per cell (std rendered as ``—`` when n < 2 — an
  undefined std is never fabricated as ``0.0000``)
- mean difference Δ, Cohen's d (same avg/std-of-differences semantics as
  experiments.result_analysis.paired_effects), exact two-sided Wilcoxon p
  from the analyzer's stored test record, seed count n, and the
  preregistered gate verdict
- LaTeX via booktabs; caption states n, test identity, and that every number
  is deterministically derived from per-seed observations

No model involvement: zero fabrication surface. Metric-key resolution mirrors
``result_analysis._resolve_document_metric`` (imported, not duplicated, so the
two can never drift).
"""

from __future__ import annotations

from typing import Any

from autoresearch.experiments.result_analysis import _resolve_document_metric

__all__ = ["ResultTables", "build_result_tables"]


class ResultTables(tuple):
    """(markdown, latex) pair with the rendered tables."""


def _series(analysis: Any, metric: str) -> tuple[list[float], list[float]]:
    """Per-seed (baseline, candidate) series for one metric from the stored
    seed-level rows."""
    before: list[float] = []
    after: list[float] = []
    rows = list(analysis.seed_level_results or ())
    if rows:
        sample_left = rows[0].get("baseline") or {}
        sample_right = (rows[0].get("candidate") or rows[0]) or {}
        key = _resolve_document_metric(metric, sample_left, sample_right)
        if key is None:
            return before, after
        for row in rows:
            left = row.get("baseline", {})
            right = row.get("candidate", row)
            if (
                isinstance(left, dict)
                and isinstance(right, dict)
                and key in left
                and key in right
            ):
                before.append(float(left[key]))
                after.append(float(right[key]))
    return before, after


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _std(values: list[float]) -> float | None:
    """Sample std; None when undefined (n < 2) — rendered as an em dash."""
    if len(values) < 2:
        return None
    avg = _mean(values)
    return (sum((v - avg) ** 2 for v in values) / (len(values) - 1)) ** 0.5


def _cohens_d(before: list[float], after: list[float]) -> float | None:
    """avg / std-of-differences — identical semantics to
    paired_effects()['cohens_d'] (result_analysis.py); recomputed here from
    stored seed rows because the analysis record persists only the mean
    difference. None when undefined: n < 2, or deviation below 1e-12 (constant
    differences leave ~1e-17 floating-point residue that would otherwise
    explode into a fabricated huge d)."""
    if len(before) < 2 or len(before) != len(after):
        return None
    diffs = [b - a for a, b in zip(before, after)]
    avg = _mean(diffs)
    deviation = (sum((d - avg) ** 2 for d in diffs) / len(diffs)) ** 0.5
    if deviation < 1e-12:
        return None
    return avg / deviation


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def _cell(values: list[float]) -> str:
    std = _std(values)
    std_text = "—" if std is None else f"±{_fmt(std)}"
    return f"{_fmt(_mean(values))} {std_text}" if values else "—"


def _p_value(analysis: Any, metric: str) -> str:
    test = dict(analysis.statistical_tests or {}).get(metric)
    if (
        isinstance(test, dict)
        and test.get("status") == "VALID"
        and test.get("p_value") is not None
    ):
        return f"{float(test['p_value']):.4f}"
    return "—"


def build_result_tables(analysis: Any) -> ResultTables:
    """Render the main results table (markdown + LaTeX) from one analysis."""
    metrics = sorted(dict(analysis.proposed_metrics or {}))
    n = len(analysis.seed_level_results or ())

    header = (
        "| Metric | Baseline (mean±std) | Candidate (mean±std) | Δ mean | "
        "Cohen's d | exact p | n | Gate |"
    )
    divider = "|---|---|---|---|---|---|---|---|"
    rows: list[str] = []
    tex_rows: list[str] = []
    for metric in metrics:
        before, after = _series(analysis, metric)
        delta = float(dict(analysis.effect_size).get(metric, 0.0))
        d = _cohens_d(before, after)
        gate = dict(analysis.decision_gates or {}).get(f"{metric}_gate")
        gate_text = "PASS" if gate else "FAIL" if gate is not None else "—"
        d_text = "—" if d is None else f"{d:+.3f}"
        delta_text = f"{delta:+.4f}"
        p_text = _p_value(analysis, metric)
        rows.append(
            f"| {metric} | {_cell(before)} | {_cell(after)} | {delta_text} | "
            f"{d_text} | {p_text} | {n} | {gate_text} |"
        )
        # PEP-701-incompatible escapes (backslashes inside f-string
        # expressions) would break Python 3.11 — precompute the TeX cells.
        before_tex = _cell(before).replace("±", "$\\pm$")
        after_tex = _cell(after).replace("±", "$\\pm$")
        tex_rows.append(
            f"{metric} & {before_tex} & {after_tex} & {delta_text} & {d_text} & "
            f"{p_text} & {n} & {gate_text} \\\\"
        )

    markdown = "\n".join([header, divider, *rows]) if rows else header + "\n" + divider
    caption = (
        f"Results over {n} seed{'s' if n != 1 else ''} (mean±std; exact two-sided "
        "Wilcoxon signed-rank; Cohen's d on paired differences; every value "
        "deterministically derived from per-seed observations)."
    )
    latex = (
        "\\begin{table}[h]\\centering\\begin{tabular}{lccccccc}\\toprule\n"
        "Metric & Baseline & Candidate & $\\Delta$ mean & Cohen's $d$ & exact $p$ & $n$ & Gate \\\\\n"
        "\\midrule\n" + "\n".join(tex_rows) + "\n\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{{caption}}}\n\\end{{table}}"
    )
    return ResultTables((markdown, latex))
