"""Purpose: Numeric-provenance auditing for the trust-audit harness (DESIGN.md §5.3).

Deterministic and fail-closed: every numeric token in the paper text is
classified as TRACED_ARTIFACT / TRACED_SOURCE / STRUCTURAL / UNTRACED
against an allowed-value set harvested from artifact-role files, and every
claimed metric inside a measurement document is independently recomputed
from raw predictions/labels/scores through the ``MetricRecomputers``
registry. Metrics without a recomputer, or claims without raw arrays, are
recorded honestly as NOT_RECOMPUTABLE with coverage stats — never counted
as verified.

Frozen rules (reuse templates and the deliberate deviations from them):

* Number-token regex ``r"-?\\d+(?:\\.\\d+)?"`` and the structural-int cap
  follow ``application/manuscript.py::_audit_draft``. Deviation: STRUCTURAL
  additionally requires a non-negative magnitude — manuscript.py's signed
  comparison ``value <= 12`` also accepted every negative integer, while
  DESIGN §5.3 says "small integers" (section numbers, enumeration), so the
  audit freezes ``0 <= |value| <= STRUCTURAL_INT_MAX``. Percentages are
  never structural.
* Rounding-variant registration (round to 2..6 decimals) follows
  ``application/manuscript.py::_allowed_numbers``. Percent normalization
  ("92.3%", "92.3", "0.923" all register) is applied asymmetrically and
  conservatively: a token adjoined by "%" also tries ``value/100``, and a
  harvested value under a percent-suggesting key/column also registers
  ``value/100``. Plain values never gain percent variants (that would let
  a plain "5" trace to a harvested 0.05).
* Recomputation semantics follow ``experiments/ingestion.py``
  (tolerance 1e-6, ``IntegrityFailure`` on invalid arrays, baseline arm
  uses ``raw_outputs.baseline_predictions`` when present). Deviation:
  where ingestion *rejects* a mismatched document, the auditor *records*
  RECOMPUTED_MISMATCH — a mismatch is a property of the audited artifact,
  not a harness failure.
* Artifacts whose own provenance declares FIXTURE (vocabulary of
  ``experiments/integrity.py::ProvenanceType``) are rejected entirely as
  numeric evidence, mirroring the ingester's fail-closed gate.

This module defines its own lightweight input dataclasses (``TextSection``,
``ArtifactFile``, ``SourceValue``) mirroring the documented contract shapes
(DESIGN §4.1/§5.3) so it stays importable while ``audit/contract.py`` is
under construction; the integrator adapts them later. ``audit()`` never
raises on artifact-side problems (unparseable files, fixture taint, missing
raw outputs): everything is recorded as a skip or a NOT_RECOMPUTABLE with a
reason.
"""

from __future__ import annotations

import csv
import io
import json
import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from autoresearch.experiments.integrity import IntegrityFailure, ProvenanceType

NUMBERS_METRIC_VERSION = "audit-numbers-v1"

# Pattern frozen from application/manuscript.py::_audit_draft.
NUMBER_TOKEN_RE = re.compile(r"-?\d+(?:\.\d+)?")
STRUCTURAL_INT_MAX = 12
ROUNDING_PRECISIONS = (2, 3, 4, 5, 6)
MATCH_TOLERANCE = 1e-9
RECOMPUTE_TOLERANCE = 1e-6

# Log-style ``key = value`` / ``key: value`` lines (DESIGN §5.3.2 "log files
# (regex-keyed)").
_LOG_KEY_VALUE_RE = re.compile(
    r"(?P<key>[A-Za-z_][A-Za-z0-9_.\-]*)\s*[=:]\s*(?P<value>-?\d+(?:\.\d+)?)(?P<percent>\s*%)?"
)
_REFERENCE_PREFIX_RE = re.compile(
    r"(figure|fig|table|section|sec|eq|equation|step|page)\.?\s*$", re.IGNORECASE
)
_YEAR_MIN, _YEAR_MAX = 1500, 2100


class NumberClass(StrEnum):
    TRACED_ARTIFACT = "TRACED_ARTIFACT"
    TRACED_SOURCE = "TRACED_SOURCE"
    STRUCTURAL = "STRUCTURAL"
    UNTRACED = "UNTRACED"


class ComparisonVerdict(StrEnum):
    RECOMPUTED_MATCH = "RECOMPUTED_MATCH"
    RECOMPUTED_MISMATCH = "RECOMPUTED_MISMATCH"
    NOT_RECOMPUTABLE = "NOT_RECOMPUTABLE"


# ----------------------------------------------------------------------
# Lightweight input records (contract adapters map onto these)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TextSection:
    """One named paper-text section (source file reference is the path)."""

    name: str
    text: str


@dataclass(frozen=True)
class ArtifactFile:
    """One artifact-role file, content-decoded and sha-pinned."""

    path: str
    sha256: str
    text: str


@dataclass(frozen=True)
class SourceValue:
    """A bibliographic number (reference year, stated dataset size from a
    RESOLVED_EXACT source) forming the TRACED_SOURCE allowed set."""

    value: float
    label: str
    origin: str = "bibliography"


@dataclass(frozen=True)
class HarvestedValue:
    """One numeric value harvested from an artifact, with provenance."""

    value: float
    path: str
    sha256: str
    location: str  # JSON path, line, or row:column inside the file
    percent: bool = False  # key/column suggested a percent form


# ----------------------------------------------------------------------
# Number tokens
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class NumberToken:
    section: str
    value: float
    text: str
    start: int
    end: int
    # plain | percent | year | figure_ref | table_ref | section_ref | other_ref
    context: str


@dataclass(frozen=True)
class NumberFinding:
    token: NumberToken
    classification: NumberClass
    trace: HarvestedValue | SourceValue | None = None


def _peek_percent(text: str, end: int) -> bool:
    index = end
    while index < len(text) and text[index] == " ":
        index += 1
    return index < len(text) and text[index] == "%"


def _context_for(text: str, start: int, end: int, value: float) -> str:
    if _peek_percent(text, end):
        return "percent"
    prefix = text[max(0, start - 16) : start]
    match = _REFERENCE_PREFIX_RE.search(prefix)
    if match is not None:
        kind = match.group(1).lower()
        if kind in ("figure", "fig"):
            return "figure_ref"
        if kind == "table":
            return "table_ref"
        if kind in ("section", "sec"):
            return "section_ref"
        if kind in ("eq", "equation"):
            return "equation_ref"
        return "other_ref"
    if float(value).is_integer() and _YEAR_MIN <= value <= _YEAR_MAX:
        return "year"
    return "plain"


def _extract_tokens(section: TextSection) -> list[NumberToken]:
    tokens: list[NumberToken] = []
    for match in NUMBER_TOKEN_RE.finditer(section.text):
        value = float(match.group(0))
        tokens.append(
            NumberToken(
                section=section.name,
                value=value,
                text=match.group(0),
                start=match.start(),
                end=match.end(),
                context=_context_for(section.text, match.start(), match.end(), value),
            )
        )
    return tokens


# ----------------------------------------------------------------------
# Allowed-value registry (manuscript.py::_allowed_numbers pattern)
# ----------------------------------------------------------------------


class _ValueRegistry:
    """Allowed-value set with rounding-variant registration and tolerance
    matching. Records keep provenance for trace detail records."""

    def __init__(self) -> None:
        self._entries: list[tuple[float, Any]] = []

    def register(
        self, value: float, record: Any, *, percent_variant: bool = False
    ) -> None:
        number = _as_float(value)
        if number is None:
            return
        variants: list[float] = [number]
        variants.extend(round(number, precision) for precision in ROUNDING_PRECISIONS)
        if percent_variant:
            variants.append(number / 100.0)
        for variant in variants:
            self._entries.append((float(variant), record))

    def match(self, candidates: Sequence[float]) -> Any | None:
        for candidate in candidates:
            for value, record in self._entries:
                if abs(candidate - value) <= MATCH_TOLERANCE:
                    return record
        return None

    def __len__(self) -> int:
        return len(self._entries)


def _classify_token(
    token: NumberToken,
    artifact_registry: _ValueRegistry,
    source_registry: _ValueRegistry,
) -> NumberFinding:
    if token.context != "percent":  # percentages are never structural
        for candidate in (token.value, abs(token.value)):
            if float(candidate).is_integer() and 0.0 <= candidate <= STRUCTURAL_INT_MAX:
                return NumberFinding(token, NumberClass.STRUCTURAL)
    candidates = [token.value]
    if token.context == "percent":
        candidates.append(token.value / 100.0)
    artifact_trace = artifact_registry.match(candidates)
    if artifact_trace is not None:
        return NumberFinding(token, NumberClass.TRACED_ARTIFACT, artifact_trace)
    source_trace = source_registry.match(candidates)
    if source_trace is not None:
        return NumberFinding(token, NumberClass.TRACED_SOURCE, source_trace)
    return NumberFinding(token, NumberClass.UNTRACED)


# ----------------------------------------------------------------------
# Artifact harvesting (DESIGN §5.3.2)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class _HarvestOutcome:
    path: str
    values: tuple[HarvestedValue, ...]
    measurement_docs: tuple[tuple[dict[str, Any], str], ...]
    skip_reason: str | None = None


_NO_JSON = object()


def _try_json(text: str) -> Any:
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return _NO_JSON


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number):
        return None
    return number


def _is_fixture_tainted(node: Any) -> bool:
    """True when any nested object declares FIXTURE provenance (mirrors the
    ingester's seed-row/artifact FIXTURE gate)."""
    if isinstance(node, Mapping):
        declared = str(node.get("provenance_type", "")).strip().upper()
        if declared == ProvenanceType.FIXTURE.value:
            return True
        return any(_is_fixture_tainted(child) for child in node.values())
    if isinstance(node, (list, tuple)):
        return any(_is_fixture_tainted(child) for child in node)
    return False


def _leaf_key(path: str) -> str:
    leaf = path.rsplit(".", 1)[-1]
    return leaf.split("[", 1)[0]


def _key_suggests_percent(key: str) -> bool:
    lowered = (key or "").lower()
    return "percent" in lowered or "pct" in lowered or "%" in key


def _walk_json(
    node: Any, path: str, artifact: ArtifactFile, values: list[HarvestedValue]
) -> None:
    if node is None or isinstance(node, bool):
        return
    if isinstance(node, (int, float)):
        values.append(
            HarvestedValue(
                float(node),
                artifact.path,
                artifact.sha256,
                path,
                _key_suggests_percent(_leaf_key(path)),
            )
        )
        return
    if isinstance(node, Mapping):
        for key, child in node.items():
            _walk_json(child, f"{path}.{key}", artifact, values)
        return
    if isinstance(node, (list, tuple)):
        for index, child in enumerate(node):
            _walk_json(child, f"{path}[{index}]", artifact, values)


def _harvest_csv(artifact: ArtifactFile) -> list[HarvestedValue]:
    values: list[HarvestedValue] = []
    reader = csv.DictReader(io.StringIO(artifact.text))
    for row_number, row in enumerate(reader, start=1):
        for column, cell in row.items():
            if column is None or cell is None:
                continue
            number = _as_float(str(cell).strip())
            if number is None:
                continue
            values.append(
                HarvestedValue(
                    number,
                    artifact.path,
                    artifact.sha256,
                    f"R{row_number}:{column.strip()}",
                    _key_suggests_percent(column),
                )
            )
    return values


def _harvest_artifact(artifact: ArtifactFile) -> _HarvestOutcome:
    text = artifact.text
    stripped = text.strip()
    lowered_path = artifact.path.lower()
    if not stripped:
        return _HarvestOutcome(artifact.path, (), (), "empty")

    whole = _try_json(stripped)
    if whole is not _NO_JSON and isinstance(whole, (dict, list)):
        if _is_fixture_tainted(whole):
            return _HarvestOutcome(artifact.path, (), (), "fixture_prohibited")
        values: list[HarvestedValue] = []
        _walk_json(whole, "$", artifact, values)
        docs: list[tuple[dict[str, Any], str]] = []
        if isinstance(whole, dict) and "seeds" in whole:
            docs.append((whole, f"{artifact.path}#$"))
        return _HarvestOutcome(artifact.path, tuple(values), tuple(docs))

    if lowered_path.endswith(".csv"):
        return _HarvestOutcome(artifact.path, tuple(_harvest_csv(artifact)), ())

    # Per-line pass: single-line JSON objects (JSONL and the measurement
    # document contract — "last single-line JSON with `seeds`") plus
    # regex-keyed key=value lines from logs.
    values = []
    docs = []
    tainted = False
    parsed_json_line = False
    for line_number, line in enumerate(text.splitlines(), start=1):
        line_text = line.strip()
        if not line_text:
            continue
        obj = _try_json(line_text)
        if isinstance(obj, dict):
            parsed_json_line = True
            if _is_fixture_tainted(obj):
                tainted = True
                continue
            _walk_json(obj, f"L{line_number}$", artifact, values)
            if "seeds" in obj:
                docs.append((obj, f"{artifact.path}#L{line_number}"))
            continue
        for match in _LOG_KEY_VALUE_RE.finditer(line):
            key = match.group("key")
            number = _as_float(match.group("value"))
            if number is None:
                continue
            percent = match.group("percent") is not None or _key_suggests_percent(key)
            values.append(
                HarvestedValue(
                    number,
                    artifact.path,
                    artifact.sha256,
                    f"L{line_number}:{key}",
                    percent,
                )
            )
    if tainted:
        return _HarvestOutcome(artifact.path, (), (), "fixture_prohibited")
    if lowered_path.endswith((".json", ".jsonl")) and not parsed_json_line:
        return _HarvestOutcome(artifact.path, (), (), "unparseable")
    if not values and not docs:
        return _HarvestOutcome(artifact.path, (), (), "no_numeric_values")
    return _HarvestOutcome(artifact.path, tuple(values), tuple(docs))


# ----------------------------------------------------------------------
# Metric recomputers (DESIGN §5.3.4; extends ingestion.py's accuracy-only
# recomputation to precision/recall/F1/AUROC)
# ----------------------------------------------------------------------

MetricRecomputeFunction = Callable[
    [Sequence[Any], Sequence[Any], Sequence[float] | None], float
]


def _require_paired(
    name: str, predictions: Sequence[Any], labels: Sequence[Any]
) -> None:
    if len(predictions) != len(labels) or not labels:
        raise IntegrityFailure(
            f"METRIC_RECOMPUTATION_INVALID:{name}: predictions/labels length mismatch or empty"
        )


def _recompute_accuracy(
    predictions: Sequence[Any], labels: Sequence[Any], scores: Sequence[float] | None
) -> float:
    """Exact semantics of experiments/ingestion.py::_recompute_metric."""
    _require_paired("accuracy", predictions, labels)
    correct = sum(
        1 for predicted, actual in zip(predictions, labels) if predicted == actual
    )
    return correct / len(labels)


def _per_class_counts(
    predictions: Sequence[Any], normalized_labels: Sequence[str], class_name: str
) -> tuple[int, int, int]:
    true_positive = false_positive = false_negative = 0
    for predicted, actual in zip(predictions, normalized_labels):
        predicted_label = str(predicted)
        if predicted_label == class_name and actual == class_name:
            true_positive += 1
        elif predicted_label == class_name and actual != class_name:
            false_positive += 1
        elif predicted_label != class_name and actual == class_name:
            false_negative += 1
    return true_positive, false_positive, false_negative


def _precision_recall_f1(
    predictions: Sequence[Any], labels: Sequence[Any]
) -> tuple[float, float, float]:
    normalized_labels = [str(label) for label in labels]
    binary = set(normalized_labels) <= {"0", "1"}
    if binary:
        tp, fp, fn = _per_class_counts(predictions, normalized_labels, "1")
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
    else:
        precisions: list[float] = []
        recalls: list[float] = []
        for class_name in sorted(set(normalized_labels)):
            tp, fp, fn = _per_class_counts(predictions, normalized_labels, class_name)
            precisions.append(tp / (tp + fp) if (tp + fp) else 0.0)
            recalls.append(tp / (tp + fn) if (tp + fn) else 0.0)
        precision = sum(precisions) / len(precisions) if precisions else 0.0
        recall = sum(recalls) / len(recalls) if recalls else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    )
    return precision, recall, f1


def _recompute_precision(
    predictions: Sequence[Any], labels: Sequence[Any], scores: Sequence[float] | None
) -> float:
    _require_paired("precision", predictions, labels)
    return _precision_recall_f1(predictions, labels)[0]


def _recompute_recall(
    predictions: Sequence[Any], labels: Sequence[Any], scores: Sequence[float] | None
) -> float:
    _require_paired("recall", predictions, labels)
    return _precision_recall_f1(predictions, labels)[1]


def _recompute_f1(
    predictions: Sequence[Any], labels: Sequence[Any], scores: Sequence[float] | None
) -> float:
    _require_paired("f1", predictions, labels)
    return _precision_recall_f1(predictions, labels)[2]


def _recompute_auroc(
    predictions: Sequence[Any], labels: Sequence[Any], scores: Sequence[float] | None
) -> float:
    if scores is None:
        raise IntegrityFailure(
            "METRIC_RECOMPUTATION_INVALID:auroc: prediction scores required"
        )
    _require_paired("auroc", predictions, labels)
    if len(scores) != len(labels):
        raise IntegrityFailure(
            "METRIC_RECOMPUTATION_INVALID:auroc: scores/labels length mismatch"
        )
    numeric_scores: list[float] = []
    for score in scores:
        number = _as_float(score)
        if number is None:
            raise IntegrityFailure(
                "METRIC_RECOMPUTATION_INVALID:auroc: non-numeric score"
            )
        numeric_scores.append(number)
    normalized_labels = [str(label) for label in labels]
    if not set(normalized_labels) <= {"0", "1"}:
        raise IntegrityFailure("METRIC_RECOMPUTATION_INVALID:auroc: non-binary labels")
    positives = normalized_labels.count("1")
    negatives = normalized_labels.count("0")
    if positives == 0 or negatives == 0:
        raise IntegrityFailure(
            "METRIC_RECOMPUTATION_INVALID:auroc: single-class labels"
        )
    # Rank-based Mann-Whitney U with average ranks for ties.
    order = sorted(range(len(numeric_scores)), key=lambda index: numeric_scores[index])
    ranks = [0.0] * len(numeric_scores)
    i = 0
    while i < len(order):
        j = i
        while (
            j + 1 < len(order)
            and numeric_scores[order[j + 1]] == numeric_scores[order[i]]
        ):
            j += 1
        average_rank = (i + j + 2) / 2.0  # 1-based ranks i+1 .. j+1
        for position in range(i, j + 1):
            ranks[order[position]] = average_rank
        i = j + 1
    rank_sum_positives = sum(
        rank for rank, label in zip(ranks, normalized_labels) if label == "1"
    )
    return (rank_sum_positives - positives * (positives + 1) / 2.0) / (
        positives * negatives
    )


class MetricRecomputers:
    """Registry of pure per-metric recomputation functions over raw
    predictions/labels(/scores). Anything without a recomputer is honestly
    NOT_RECOMPUTABLE — never counted as verified (DESIGN §7.2)."""

    def __init__(self) -> None:
        self._functions: dict[str, MetricRecomputeFunction] = {}

    def register(self, name: str, function: MetricRecomputeFunction) -> None:
        self._functions[str(name).strip().lower()] = function

    def has(self, name: str) -> bool:
        return str(name).strip().lower() in self._functions

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._functions))

    def recompute(
        self,
        name: str,
        *,
        predictions: Sequence[Any],
        labels: Sequence[Any],
        scores: Sequence[float] | None = None,
    ) -> float | None:
        """Recompute one metric; None when no recomputer is registered.
        Raises ``IntegrityFailure`` for invalid raw arrays (same vocabulary
        as experiments/ingestion.py)."""
        function = self._functions.get(str(name).strip().lower())
        if function is None:
            return None
        return function(predictions, labels, scores)

    @classmethod
    def with_standard(cls) -> MetricRecomputers:
        registry = cls()
        registry.register("accuracy", _recompute_accuracy)
        registry.register("acc", _recompute_accuracy)
        registry.register("precision", _recompute_precision)
        registry.register("recall", _recompute_recall)
        registry.register("f1", _recompute_f1)
        registry.register("macro_f1", _recompute_f1)
        registry.register("auroc", _recompute_auroc)
        registry.register("auc", _recompute_auroc)
        return registry


# ----------------------------------------------------------------------
# Recomputation comparison (pattern of ingestion.py::_verify_recomputation)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RecomputeComparison:
    metric: str
    arm: str  # baseline | candidate
    seed: str
    claimed: float
    recomputed: float | None
    verdict: ComparisonVerdict
    reason: str = "none"
    provenance: str = ""


def _compare_metrics(
    measurement_docs: Sequence[tuple[dict[str, Any], str]],
    recomputers: MetricRecomputers,
    skip_reasons: Counter[str],
) -> tuple[list[RecomputeComparison], dict[str, int]]:
    comparisons: list[RecomputeComparison] = []
    reportable = recomputable = matched = mismatched = 0
    for document, location in measurement_docs:
        seeds = document.get("seeds")
        if not isinstance(seeds, list):
            continue
        for row in seeds:
            if not isinstance(row, Mapping):
                continue
            raw = row.get("raw_outputs") or {}
            if not isinstance(raw, Mapping):
                raw = {}
            predictions = raw.get("predictions")
            labels = raw.get("labels")
            scores = raw.get("scores")
            seed = str(row.get("seed"))
            arrays_valid = isinstance(predictions, list) and isinstance(labels, list)
            for arm in ("baseline", "candidate"):
                claimed_metrics = row.get(arm) or {}
                if not isinstance(claimed_metrics, Mapping):
                    continue
                arm_predictions = (
                    predictions
                    if arm == "candidate"
                    else raw.get("baseline_predictions", predictions)
                )
                for metric, value in claimed_metrics.items():
                    claimed = _as_float(value)
                    if claimed is None:
                        skip_reasons["claimed_not_numeric"] += 1
                        continue
                    reportable += 1
                    verdict = ComparisonVerdict.NOT_RECOMPUTABLE
                    recomputed: float | None = None
                    reason = "none"
                    if not recomputers.has(str(metric)):
                        reason = "no_recomputer"
                    elif not arrays_valid or not isinstance(arm_predictions, list):
                        reason = "no_raw_outputs"
                    else:
                        try:
                            recomputed = recomputers.recompute(
                                str(metric),
                                predictions=arm_predictions,
                                labels=labels,
                                scores=scores if isinstance(scores, list) else None,
                            )
                        except IntegrityFailure as exc:
                            reason = str(exc)
                        else:
                            recomputable += 1
                            if abs(float(recomputed) - claimed) <= RECOMPUTE_TOLERANCE:
                                verdict = ComparisonVerdict.RECOMPUTED_MATCH
                                matched += 1
                            else:
                                verdict = ComparisonVerdict.RECOMPUTED_MISMATCH
                                mismatched += 1
                    comparisons.append(
                        RecomputeComparison(
                            metric=str(metric),
                            arm=arm,
                            seed=seed,
                            claimed=claimed,
                            recomputed=recomputed,
                            verdict=verdict,
                            reason=reason,
                            provenance=location,
                        )
                    )
    counts = {
        "metrics_reportable": reportable,
        "metrics_recomputable": recomputable,
        "recomputed_match": matched,
        "recomputed_mismatch": mismatched,
        "not_recomputable": reportable - recomputable,
    }
    return comparisons, counts


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class NumericProvenanceResult:
    status: str  # RUN_COMPLETE | NOT_RUN
    not_run_reason: str | None
    counts: Mapping[str, int]
    rates: Mapping[str, float | None]
    coverage: Mapping[str, Any]
    details: tuple[Mapping[str, Any], ...]

    def to_section(self) -> dict[str, Any]:
        """metricSection-shaped dict (DESIGN §8.1 $defs/metricSection)."""
        return {
            "status": self.status,
            "not_run_reason": self.not_run_reason,
            "llm_assisted": False,
            "counts": dict(self.counts),
            "rates": dict(self.rates),
            "coverage": dict(self.coverage),
            "details": [dict(detail) for detail in self.details],
        }


def _not_run_result(reason: str) -> NumericProvenanceResult:
    return NumericProvenanceResult(
        status="NOT_RUN",
        not_run_reason=reason,
        counts={},
        rates={
            "numeric_provenance_rate": None,
            "recomputation_match_rate": None,
            "recomputable_coverage_rate": None,
        },
        coverage={"items_examined": 0, "items_skipped": 0, "skip_reasons": {}},
        details=(),
    )


def _token_detail(finding: NumberFinding) -> dict[str, Any]:
    token = finding.token
    trace: dict[str, Any] | None = None
    if isinstance(finding.trace, HarvestedValue):
        trace = {
            "kind": "artifact",
            "path": finding.trace.path,
            "sha256": finding.trace.sha256,
            "location": finding.trace.location,
        }
    elif isinstance(finding.trace, SourceValue):
        trace = {
            "kind": "source",
            "label": finding.trace.label,
            "origin": finding.trace.origin,
        }
    return {
        "kind": "number",
        "section": token.section,
        "token": token.text,
        "value": token.value,
        "start": token.start,
        "end": token.end,
        "context": token.context,
        "classification": finding.classification.value,
        "trace": trace,
    }


def _comparison_detail(comparison: RecomputeComparison) -> dict[str, Any]:
    return {
        "kind": "recompute",
        "metric": comparison.metric,
        "arm": comparison.arm,
        "seed": comparison.seed,
        "claimed": comparison.claimed,
        "recomputed": comparison.recomputed,
        "verdict": comparison.verdict.value,
        "reason": comparison.reason,
        "provenance": comparison.provenance,
    }


class NumericProvenanceAuditor:
    """Deterministic numeric-provenance audit (DESIGN §5.3)."""

    def __init__(self, recomputers: MetricRecomputers | None = None) -> None:
        self._recomputers = (
            recomputers
            if recomputers is not None
            else MetricRecomputers.with_standard()
        )

    def audit(
        self,
        sections: Sequence[TextSection],
        artifact_files: Sequence[ArtifactFile] = (),
        source_values: Sequence[SourceValue] = (),
    ) -> NumericProvenanceResult:
        if not any(section.text.strip() for section in sections):
            return _not_run_result("no_paper_text")

        skip_reasons: Counter[str] = Counter()
        artifact_registry = _ValueRegistry()
        measurement_docs: list[tuple[dict[str, Any], str]] = []
        files_used = 0
        for artifact in artifact_files:
            outcome = _harvest_artifact(artifact)
            if outcome.skip_reason is not None:
                skip_reasons[outcome.skip_reason] += 1
                continue
            files_used += 1
            for harvested in outcome.values:
                artifact_registry.register(
                    harvested.value, harvested, percent_variant=harvested.percent
                )
            measurement_docs.extend(outcome.measurement_docs)

        source_registry = _ValueRegistry()
        for source_value in source_values:
            source_registry.register(source_value.value, source_value)

        findings: list[NumberFinding] = []
        for section in sections:
            for token in _extract_tokens(section):
                findings.append(
                    _classify_token(token, artifact_registry, source_registry)
                )

        comparisons, recompute_counts = _compare_metrics(
            measurement_docs, self._recomputers, skip_reasons
        )

        class_counts = Counter(finding.classification for finding in findings)
        traced = (
            class_counts[NumberClass.TRACED_ARTIFACT]
            + class_counts[NumberClass.TRACED_SOURCE]
            + class_counts[NumberClass.STRUCTURAL]
        )
        tokens_total = len(findings)
        recomputable = recompute_counts["metrics_recomputable"]
        reportable = recompute_counts["metrics_reportable"]
        counts: dict[str, int] = {
            "numeric_tokens_examined": tokens_total,
            "traced_artifact": class_counts[NumberClass.TRACED_ARTIFACT],
            "traced_source": class_counts[NumberClass.TRACED_SOURCE],
            "structural": class_counts[NumberClass.STRUCTURAL],
            "untraced": class_counts[NumberClass.UNTRACED],
            "artifact_files_examined": files_used,
            "artifact_files_skipped": sum(skip_reasons.values()),
            "measurement_documents": len(measurement_docs),
            **recompute_counts,
        }
        rates: dict[str, float | None] = {
            "numeric_provenance_rate": (traced / tokens_total)
            if tokens_total
            else None,
            "recomputation_match_rate": (
                recompute_counts["recomputed_match"] / recomputable
                if recomputable
                else None
            ),
            "recomputable_coverage_rate": (recomputable / reportable)
            if reportable
            else None,
        }
        coverage = {
            "items_examined": tokens_total,
            "items_skipped": sum(skip_reasons.values()),
            "skip_reasons": dict(skip_reasons),
        }
        details: list[dict[str, Any]] = [_token_detail(finding) for finding in findings]
        details.extend(_comparison_detail(comparison) for comparison in comparisons)
        return NumericProvenanceResult(
            status="RUN_COMPLETE",
            not_run_reason=None,
            counts=counts,
            rates=rates,
            coverage=coverage,
            details=tuple(details),
        )
