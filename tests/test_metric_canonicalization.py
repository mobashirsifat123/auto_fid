"""Regression: spec metric descriptors with parenthetical annotations must
resolve to the measurement document's bare-identifier keys.

Live bug (2026-10-16, FIRE-Bench experiment execution): the planner wrote
metrics like "assumption_sensitivity_score (PRIMARY: fraction of ...)" while
the generated implementation emitted "assumption_sensitivity_score"; the
analyzer compared strings exactly and raised IntegrityFailure
(INCOMPLETE_EXPERIMENT), leaving the run in executed_analysis_pending with no
report. Canonical matching (annotation stripped, case/whitespace-normalized)
bridges the two, with exact matches still preferred.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, ExecutionResult
from autoresearch.experiments.result_analysis import StatisticalAnalyzer

IDENTITY_KEYS = ("candidate_id", "method_id", "repository_commit", "implementation_hash", "config_hash", "dataset_checksum", "split_checksum", "metric_version")


def _identity() -> dict:
    return {
        "candidate_id": "cand-1", "method_id": "m", "repository_commit": "r",
        "implementation_hash": "i", "config_hash": "c", "dataset_checksum": "d",
        "split_checksum": "s", "metric_version": "v",
    }


def _result(seed_rows: list[dict]) -> ExecutionResult:
    return ExecutionResult(
        id=uuid4().hex, project_id="p", run_id="r", job_id="j", exit_code=0,
        stdout="", stderr="", structured_metrics={"seeds": seed_rows},
        gpu_telemetry={}, checkpoint_manifest=(), failure_classification=None,
        provenance=ArtifactProvenance.create(project_id="p", run_id="r", source_evidence=(), provider="worker", model="local", prompt_version="test", content="x"),
        identity=_identity(),
        artifacts=({"artifact_type": "seed_metrics", "verified": True},),
    )


def _spec() -> SimpleNamespace:
    return SimpleNamespace(
        id="spec-1",
        project_id="p",
        run_id="r",
        candidate_id="cand-1",
        dataset_checksum="d",
        split_definition="s",
        required_seed_count=2,
        required_artifacts=("seed_metrics",),
        metrics=["score (PRIMARY: fraction of flips)", "rank_flip_count"],
        minimum_meaningful_effect={},
        metric_direction={},
    )


def test_annotated_spec_metrics_resolve_to_clean_document_keys():
    seed_rows = []
    for seed, (b_score, c_score) in enumerate(((1.0, 2.0), (2.0, 3.5)), start=1):
        seed_rows.append({
            "seed": seed, "identity": {**_identity(), "seed": seed},
            "baseline": {"score": b_score, "rank_flip_count": float(seed)},
            "candidate": {"score": c_score, "rank_flip_count": float(seed + 1)},
        })
    analysis = asyncio.run(StatisticalAnalyzer().analyze(_spec(), _result(seed_rows)))
    assert "score (PRIMARY: fraction of flips)" in analysis.baseline_metrics
    assert "rank_flip_count" in analysis.baseline_metrics
    assert analysis.hypothesis_support_assessment != "INSUFFICIENT_EVIDENCE"


def test_missing_metric_still_fails_closed():
    seed_rows = []
    for seed in (1, 2):
        seed_rows.append({
            "seed": seed, "identity": {**_identity(), "seed": seed},
            "baseline": {"rank_flip_count": float(seed)},
            "candidate": {"rank_flip_count": float(seed + 1)},
        })
    from autoresearch.experiments.integrity import IntegrityFailure

    try:
        asyncio.run(StatisticalAnalyzer().analyze(_spec(), _result(seed_rows)))
        raise AssertionError("expected IntegrityFailure")
    except IntegrityFailure as exc:
        assert "INCOMPLETE_EXPERIMENT" in str(exc)