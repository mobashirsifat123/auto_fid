"""Purpose: Computes rigorous frequentist and Bayesian statistics on raw seed metric logs."""
from __future__ import annotations

import itertools
import math
import re
from statistics import mean, stdev
from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, ExecutionResult, ExperimentSpec, StatisticalAnalysis
from autoresearch.experiments.integrity import IntegrityFailure, ProvenanceType, decision_from_gates, validate_artifact_contract, validate_identity


def canonical_metric_name(name: object) -> str:
    """Spec planners write human-readable metric descriptors ("score (PRIMARY: ...)")
    while implementations emit bare identifiers ("score"). Canonicalize for matching."""
    return re.sub(r"\s*\(.*?\)\s*$", "", str(name).strip()).lower()


def _resolve_document_metric(metric: str, left: dict, right: dict) -> str | None:
    """Resolve a spec metric descriptor to the measurement document's emitted key."""
    if metric in left and metric in right:
        return metric
    canon = canonical_metric_name(metric)
    for key in left:
        if key in right and canonical_metric_name(key) == canon:
            return str(key)
    return None


def paired_effects(baseline: list[float], proposed: list[float]) -> dict[str, float | int | str]:
    if len(baseline) != len(proposed):
        raise ValueError("Paired analysis requires equal seed counts")
    if not baseline:
        return {"sample_count": 0, "mean_difference": 0.0, "effect_size": 0.0, "status": "INSUFFICIENT_EVIDENCE"}
    differences = [b - a for a, b in zip(baseline, proposed)]
    avg = mean(differences)
    deviation = stdev(differences) if len(differences) > 1 else 0.0
    return {
        "sample_count": len(differences), "mean_difference": round(avg, 6),
        "cohens_d": round(avg / deviation, 6) if deviation else 0.0,
        "status": "INSUFFICIENT_EVIDENCE" if len(differences) < 2 else "ESTIMATE",
    }


def exact_wilcoxon_two_sided(baseline: list[float], proposed: list[float]) -> dict[str, object]:
    """Small-n exact signed-rank test; never turns an estimate into significance."""
    differences = [b - a for a, b in zip(baseline, proposed) if b != a]
    if len(differences) < 2:
        return {"status": "INSUFFICIENT_EVIDENCE", "test": "wilcoxon_signed_rank", "alternative": "two-sided"}
    ordered = sorted(enumerate(differences), key=lambda item: abs(item[1]))
    ranks = [0.0] * len(differences)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and abs(ordered[end][1]) == abs(ordered[index][1]):
            end += 1
        rank = (index + 1 + end) / 2
        for original, _ in ordered[index:end]:
            ranks[original] = rank
        index = end
    observed = sum(rank for rank, diff in zip(ranks, differences) if diff > 0)
    total = sum(ranks)
    statistic = min(observed, total - observed)
    outcomes = [sum(rank for rank, sign in zip(ranks, signs) if sign > 0) for signs in itertools.product((-1, 1), repeat=len(ranks))]
    extreme = sum(1 for value in outcomes if min(value, total - value) <= statistic + 1e-12)
    return {
        "status": "VALID", "test": "wilcoxon_signed_rank", "alternative": "two-sided", "exact": True,
        "statistic": round(statistic, 6), "p_value": round(extreme / len(outcomes), 8),
        "raw_observations": [{"baseline": a, "candidate": b} for a, b in zip(baseline, proposed)],
    }


class StatisticalAnalyzer:
    def __init__(self, router=None) -> None:
        self.router = router

    async def analyze(self, spec: ExperimentSpec, result: ExecutionResult) -> StatisticalAnalysis:
        if result.exit_code != 0:
            raise IntegrityFailure("EVALUATION_ARTIFACT")
        payload = result.structured_metrics
        if not isinstance(payload, dict):
            raise IntegrityFailure("METRIC_SCHEMA_INVALID")
        seed_rows = [row for row in payload.get("seeds", []) if isinstance(row, dict)]
        artifacts = list(result.artifacts or payload.get("artifacts", []))
        validate_artifact_contract(spec, artifacts, seed_rows=seed_rows)
        identity = result.identity or payload.get("identity", {})
        validate_identity(identity, spec)

        baseline_metrics: dict[str, object] = {}
        proposed_metrics: dict[str, object] = {}
        effects: dict[str, float] = {}
        intervals: dict[str, tuple[float, float]] = {}
        tests: dict[str, object] = {}
        gates: dict[str, bool | None] = {}
        sample_left = dict(seed_rows[0].get("baseline") or {}) if seed_rows else {}
        sample_right = dict((seed_rows[0].get("candidate") or seed_rows[0]) or {}) if seed_rows else {}
        resolved_keys: dict[str, str] = {}
        for metric in spec.metrics:
            key = _resolve_document_metric(metric, sample_left, sample_right)
            if key is None:
                raise IntegrityFailure("INCOMPLETE_EXPERIMENT")
            resolved_keys[metric] = key
        for metric in spec.metrics:
            before: list[float] = []
            after: list[float] = []
            document_key = resolved_keys[metric]
            for row in seed_rows:
                seed = int(row.get("seed", -1))
                validate_identity(row.get("identity", identity), spec, seed=seed)
                if str(row.get("provenance_type", ProvenanceType.MISSING.value)) == ProvenanceType.FIXTURE.value:
                    raise IntegrityFailure("FIXTURE_PROHIBITED")
                left = row.get("baseline", {})
                right = row.get("candidate", row)
                if not isinstance(left, dict) or not isinstance(right, dict) or document_key not in left or document_key not in right:
                    raise IntegrityFailure("INCOMPLETE_EXPERIMENT")
                before.append(float(left[document_key])); after.append(float(right[document_key]))
            summary = paired_effects(before, after)
            test = exact_wilcoxon_two_sided(before, after)
            if test["status"] != "VALID":
                raise IntegrityFailure("INSUFFICIENT_EVIDENCE")
            baseline_metrics[metric] = mean(before); proposed_metrics[metric] = mean(after)
            effects[metric] = float(summary["mean_difference"])
            intervals[metric] = (min(after) - max(before), max(after) - min(before))
            tests[metric] = test
            threshold = float(spec.minimum_meaningful_effect.get(metric, 0.0))
            direction = spec.metric_direction.get(metric, "maximize")
            relative = (mean(after) - mean(before)) / abs(mean(before) or 1.0)
            gates[f"{metric}_gate"] = (relative >= threshold) if direction == "maximize" else (-relative >= threshold)

        decision = decision_from_gates(gates)
        return StatisticalAnalysis(
            id=uuid4().hex, project_id=spec.project_id, run_id=spec.run_id, spec_id=spec.id,
            baseline_metrics=baseline_metrics, proposed_metrics=proposed_metrics, effect_size=effects,
            confidence_intervals=intervals, seed_level_results=tuple(seed_rows), ablation_contributions={},
            parameter_count_diff=0, flops_latency_diff=None, training_inference_cost=0.0,
            failure_case_analysis="Measured evidence only; no ablation contribution is reported without descendant runs.",
            hypothesis_support_assessment=decision,
            provenance=ArtifactProvenance.create(project_id=spec.project_id, run_id=spec.run_id,
                source_evidence=[result.id, *[str(a.get("id", "")) for a in artifacts]], provider="deterministic",
                model="statistics", prompt_version="stats-v5-exact-wilcoxon", content={"gates": gates, "tests": tests}),
            decision_gates=gates, statistical_tests=tests, evidence_status=decision,
        )
