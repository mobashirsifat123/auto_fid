"""Hermetic unit tests for autoresearch.audit.numbers (DESIGN §5.3).

Covers: rounding-variant tracing, percent normalization, structural ints,
years/figure-ref classification, fixture rejection, metric recomputation
(match, mismatch, NOT_RECOMPUTABLE), the MetricRecomputers registry, and
the metricSection envelope shape. No network, no LLM.
"""

from __future__ import annotations

import json

import pytest

from autoresearch.audit.numbers import (
    ArtifactFile,
    MetricRecomputers,
    NumberClass,
    NumericProvenanceAuditor,
    SourceValue,
    TextSection,
    _recompute_accuracy,
)
from autoresearch.domain.models import content_checksum
from autoresearch.experiments.integrity import IntegrityFailure


def _artifact(path: str, text: str) -> ArtifactFile:
    return ArtifactFile(path=path, sha256=content_checksum(text), text=text)


def _classify_texts(auditor_result) -> dict[str, list[str]]:
    by_class: dict[str, list[str]] = {}
    for detail in auditor_result.details:
        if detail["kind"] == "number":
            by_class.setdefault(detail["classification"], []).append(detail["token"])
    return by_class


# ----------------------------------------------------------------------
# Token classification
# ----------------------------------------------------------------------


def test_numbers_traced_via_rounding_variants() -> None:
    metrics = _artifact(
        "artifacts/metrics.json",
        json.dumps(
            {"candidate": {"accuracy": 0.9166666}, "baseline": {"accuracy": 0.8713}}
        ),
    )
    sections = [
        TextSection(
            "Results",
            "The candidate reaches 0.92 accuracy and 0.9167 in long form; the baseline scores 0.8713.",
        )
    ]
    result = NumericProvenanceAuditor().audit(sections, [metrics])
    by_class = _classify_texts(result)
    assert set(by_class) == {NumberClass.TRACED_ARTIFACT.value}
    assert sorted(by_class[NumberClass.TRACED_ARTIFACT.value]) == [
        "0.8713",
        "0.9167",
        "0.92",
    ]
    assert result.counts["numeric_tokens_examined"] == 3
    assert result.counts["traced_artifact"] == 3
    assert result.rates["numeric_provenance_rate"] == 1.0
    traces = [d["trace"] for d in result.details if d["kind"] == "number"]
    assert all(trace["path"] == "artifacts/metrics.json" for trace in traces)
    assert result.status == "RUN_COMPLETE"


def test_numbers_percent_normalization() -> None:
    metrics = _artifact("artifacts/metrics.json", json.dumps({"accuracy": 0.923}))
    sections = [TextSection("Results", "Accuracy was 92.3% across seeds.")]
    result = NumericProvenanceAuditor().audit(sections, [metrics])
    by_class = _classify_texts(result)
    assert by_class[NumberClass.TRACED_ARTIFACT.value] == ["92.3"]
    detail = next(d for d in result.details if d["kind"] == "number")
    assert detail["context"] == "percent"


def test_numbers_percent_keyed_artifact_variant() -> None:
    # Artifact stores the percent form under a percent-suggesting key; the
    # paper quotes the plain proportion.
    metrics = _artifact("artifacts/results.json", json.dumps({"accuracy_pct": 92.3}))
    sections = [TextSection("Results", "Accuracy was 0.923 across seeds.")]
    result = NumericProvenanceAuditor().audit(sections, [metrics])
    by_class = _classify_texts(result)
    assert by_class[NumberClass.TRACED_ARTIFACT.value] == ["0.923"]


def test_numbers_structural_ints_and_cap() -> None:
    sections = [
        TextSection("Results", "Section 3 and Figure 2 report 13 features in 5 tables.")
    ]
    result = NumericProvenanceAuditor().audit(sections)
    by_class = _classify_texts(result)
    assert set(by_class[NumberClass.STRUCTURAL.value]) == {"3", "2", "5"}
    assert by_class[NumberClass.UNTRACED.value] == ["13"]  # exceeds the structural cap
    assert result.rates["numeric_provenance_rate"] == 0.75


def test_numbers_percent_never_structural() -> None:
    sections = [TextSection("Results", "We keep 5% of the data.")]
    result = NumericProvenanceAuditor().audit(sections)
    by_class = _classify_texts(result)
    assert by_class == {NumberClass.UNTRACED.value: ["5"]}
    detail = next(d for d in result.details if d["kind"] == "number")
    assert detail["context"] == "percent"


def test_numbers_year_classification_traced_source() -> None:
    sections = [
        TextSection("Introduction", "Prior work in 2019 and 1998 shapes our design.")
    ]
    result = NumericProvenanceAuditor().audit(
        sections, source_values=[SourceValue(2019, "paper1", "reference_year")]
    )
    by_class = _classify_texts(result)
    assert by_class[NumberClass.TRACED_SOURCE.value] == ["2019"]
    assert by_class[NumberClass.UNTRACED.value] == ["1998"]
    detail = next(d for d in result.details if d["token"] == "2019")
    assert detail["context"] == "year"
    assert detail["trace"] == {
        "kind": "source",
        "label": "paper1",
        "origin": "reference_year",
    }


def test_numbers_figure_ref_context_recorded() -> None:
    sections = [TextSection("Results", "Figure 12 and Table 4 summarize the outcome.")]
    result = NumericProvenanceAuditor().audit(sections)
    contexts = {
        d["token"]: d["context"] for d in result.details if d["kind"] == "number"
    }
    assert contexts["4"] == "table_ref"
    assert contexts["12"] == "figure_ref"  # > cap, so UNTRACED but context recorded


def test_numbers_untraced_when_no_artifacts() -> None:
    sections = [TextSection("Abstract", "We observe an accuracy of 0.99.")]
    result = NumericProvenanceAuditor().audit(sections)
    by_class = _classify_texts(result)
    assert by_class[NumberClass.UNTRACED.value] == ["0.99"]
    assert result.counts["untraced"] == 1
    assert result.rates["numeric_provenance_rate"] == 0.0


def test_numbers_not_run_without_text() -> None:
    result = NumericProvenanceAuditor().audit([TextSection("Results", "")])
    assert result.status == "NOT_RUN"
    assert result.not_run_reason == "no_paper_text"
    assert result.rates["numeric_provenance_rate"] is None


# ----------------------------------------------------------------------
# Artifact harvesting fail-closed paths
# ----------------------------------------------------------------------


def test_numbers_fixture_artifact_rejected() -> None:
    fixture = _artifact(
        "artifacts/fake.json",
        json.dumps({"provenance_type": "FIXTURE", "accuracy": 0.99}),
    )
    sections = [TextSection("Results", "Accuracy is 0.99.")]
    result = NumericProvenanceAuditor().audit(sections, [fixture])
    by_class = _classify_texts(result)
    assert by_class[NumberClass.UNTRACED.value] == ["0.99"]
    assert result.coverage["skip_reasons"] == {"fixture_prohibited": 1}
    assert result.counts["artifact_files_skipped"] == 1


def test_numbers_fixture_seed_row_rejected() -> None:
    doc = json.dumps(
        {
            "seeds": [
                {
                    "seed": 1,
                    "provenance_type": "FIXTURE",
                    "candidate": {"accuracy": 0.5},
                    "raw_outputs": {"predictions": [1, 0], "labels": [1, 0]},
                }
            ]
        }
    )
    fixture = _artifact("artifacts/measurement.json", doc)
    sections = [TextSection("Results", "Accuracy is 0.5.")]
    result = NumericProvenanceAuditor().audit(sections, [fixture])
    assert result.coverage["skip_reasons"] == {"fixture_prohibited": 1}
    assert result.counts["measurement_documents"] == 0


def test_numbers_unparseable_json_artifact_recorded() -> None:
    broken = _artifact("artifacts/broken.json", '{"accuracy": 0.5')  # truncated
    sections = [TextSection("Results", "Accuracy is 0.5.")]
    result = NumericProvenanceAuditor().audit(sections, [broken])
    assert result.status == "RUN_COMPLETE"  # never crashes
    assert result.coverage["skip_reasons"] == {"unparseable": 1}
    assert _classify_texts(result) == {NumberClass.UNTRACED.value: ["0.5"]}


def test_numbers_csv_and_log_harvest() -> None:
    csv_artifact = _artifact(
        "artifacts/table.csv",
        "seed,accuracy\n1,0.75\n",
    )
    log_artifact = _artifact(
        "artifacts/run.log",
        "run start\nfinal accuracy = 0.75\ncoverage: 0.8\n",
    )
    sections = [TextSection("Results", "Accuracy 0.75 and coverage 0.8.")]
    result = NumericProvenanceAuditor().audit(sections, [csv_artifact, log_artifact])
    by_class = _classify_texts(result)
    assert set(by_class[NumberClass.TRACED_ARTIFACT.value]) == {"0.75", "0.8"}
    traces = {d["token"]: d["trace"] for d in result.details if d["kind"] == "number"}
    assert traces["0.75"]["path"] == "artifacts/table.csv"
    assert traces["0.75"]["location"] == "R1:accuracy"  # data rows (header excluded)
    assert traces["0.8"]["path"] == "artifacts/run.log"
    assert traces["0.8"]["location"] == "L3:coverage"
    assert traces["0.75"]["sha256"] == content_checksum("seed,accuracy\n1,0.75\n")


def test_numbers_measurement_document_from_stdout_log() -> None:
    stdout_log = _artifact(
        "artifacts/stdout.txt",
        "starting run\n"
        '{"seeds": [{"seed": 7, "candidate": {"accuracy": 0.75}, '
        '"raw_outputs": {"predictions": [1, 0, 1, 1], "labels": [1, 0, 0, 1]}}]}\n'
        "done\n",
    )
    sections = [TextSection("Results", "Accuracy of 0.75.")]
    result = NumericProvenanceAuditor().audit(sections, [stdout_log])
    assert result.counts["measurement_documents"] == 1
    by_class = _classify_texts(result)
    assert by_class[NumberClass.TRACED_ARTIFACT.value] == ["0.75"]


# ----------------------------------------------------------------------
# MetricRecomputers registry
# ----------------------------------------------------------------------


def test_recompute_accuracy_matches_ingestion_semantics() -> None:
    from autoresearch.experiments.ingestion import _recompute_metric

    predictions, labels = [1, 0, 1, 1], [1, 0, 0, 1]
    registry = MetricRecomputers.with_standard()
    assert registry.recompute(
        "accuracy", predictions=predictions, labels=labels
    ) == _recompute_metric("accuracy", predictions, labels)
    with pytest.raises(IntegrityFailure):
        registry.recompute("accuracy", predictions=[1], labels=[1, 0])
    with pytest.raises(IntegrityFailure):
        registry.recompute("accuracy", predictions=[], labels=[])


def test_recompute_precision_recall_f1_binary() -> None:
    registry = MetricRecomputers.with_standard()
    predictions, labels = [1, 1, 0], [1, 0, 1]
    assert registry.recompute(
        "precision", predictions=predictions, labels=labels
    ) == pytest.approx(0.5)
    assert registry.recompute(
        "recall", predictions=predictions, labels=labels
    ) == pytest.approx(0.5)
    assert registry.recompute(
        "f1", predictions=predictions, labels=labels
    ) == pytest.approx(0.5)


def test_recompute_precision_recall_f1_macro() -> None:
    registry = MetricRecomputers.with_standard()
    predictions, labels = ["a", "b", "a"], ["a", "b", "b"]
    assert registry.recompute(
        "precision", predictions=predictions, labels=labels
    ) == pytest.approx(0.75)
    assert registry.recompute(
        "recall", predictions=predictions, labels=labels
    ) == pytest.approx(0.75)
    assert registry.recompute(
        "f1", predictions=predictions, labels=labels
    ) == pytest.approx(0.75)


def test_recompute_auroc_rank_based_with_ties() -> None:
    registry = MetricRecomputers.with_standard()
    labels = [0, 0, 1, 1]
    assert registry.recompute(
        "auroc", predictions=[0] * 4, labels=labels, scores=[0.1, 0.4, 0.35, 0.8]
    ) == pytest.approx(0.75)
    assert registry.recompute(
        "auroc", predictions=[0] * 4, labels=labels, scores=[0.5, 0.5, 0.5, 0.5]
    ) == pytest.approx(0.5)
    with pytest.raises(IntegrityFailure):  # scores required
        registry.recompute("auroc", predictions=[0, 1], labels=[0, 1])
    with pytest.raises(IntegrityFailure):  # single-class
        registry.recompute(
            "auroc", predictions=[1, 1], labels=[1, 1], scores=[0.1, 0.2]
        )


def test_recompute_registry_registration() -> None:
    registry = MetricRecomputers()
    assert not registry.has("accuracy")
    registry.register("Accuracy", _recompute_accuracy)
    assert registry.has("accuracy")  # case-insensitive
    assert registry.names() == ("accuracy",)
    assert registry.recompute("perplexity", predictions=[1], labels=[1]) is None


# ----------------------------------------------------------------------
# Recomputation comparison inside the audit
# ----------------------------------------------------------------------


def _measurement_doc() -> ArtifactFile:
    rows = [
        {
            "seed": 1,
            "candidate": {"accuracy": 0.75, "perplexity": 3.2},
            "raw_outputs": {"predictions": [1, 0, 1, 1], "labels": [1, 0, 0, 1]},
        },
        {
            "seed": 2,
            "candidate": {"accuracy": 0.9},
            "raw_outputs": {"predictions": [1, 1, 1], "labels": [1, 1, 1]},
        },
        {"seed": 3, "candidate": {"accuracy": 0.7}},  # no raw outputs
    ]
    # MeasurementIngester.extract_document contract: one JSON document
    # carrying "seeds" (whole-file parse path; the per-line stdout path is
    # covered by test_numbers_measurement_document_from_stdout_log).
    return _artifact("artifacts/measurement.json", json.dumps({"seeds": rows}) + "\n")


def test_recompute_match_mismatch_not_recomputable() -> None:
    sections = [TextSection("Results", "Results follow.")]
    result = NumericProvenanceAuditor().audit(sections, [_measurement_doc()])
    comparisons = [d for d in result.details if d["kind"] == "recompute"]
    by_verdict: dict[str, list[dict]] = {}
    for comparison in comparisons:
        by_verdict.setdefault(comparison["verdict"], []).append(comparison)
    assert by_verdict["RECOMPUTED_MATCH"][0]["metric"] == "accuracy"
    assert by_verdict["RECOMPUTED_MATCH"][0]["arm"] == "candidate"
    assert by_verdict["RECOMPUTED_MISMATCH"][0]["claimed"] == 0.9
    assert by_verdict["RECOMPUTED_MISMATCH"][0]["recomputed"] == pytest.approx(1.0)
    assert by_verdict["NOT_RECOMPUTABLE"][0]["reason"] == "no_recomputer"  # perplexity
    not_recomputable_reasons = {c["reason"] for c in by_verdict["NOT_RECOMPUTABLE"]}
    assert "no_raw_outputs" in not_recomputable_reasons  # seed 3
    # coverage: 4 reportable claimed metrics, 2 recomputable (both accuracy rows)
    assert result.counts["metrics_reportable"] == 4
    assert result.counts["metrics_recomputable"] == 2
    assert result.counts["recomputed_match"] == 1
    assert result.counts["recomputed_mismatch"] == 1
    assert result.counts["not_recomputable"] == 2
    assert result.rates["recomputation_match_rate"] == 0.5
    assert result.rates["recomputable_coverage_rate"] == 0.5


def test_recompute_baseline_arm_uses_baseline_predictions() -> None:
    row = {
        "seed": 1,
        "baseline": {"accuracy": 1.0},
        "candidate": {"accuracy": 0.5},
        "raw_outputs": {
            "predictions": [0, 0],
            "baseline_predictions": [1, 1],
            "labels": [1, 1],
        },
    }
    artifact = _artifact("artifacts/measurement.json", json.dumps({"seeds": [row]}))
    result = NumericProvenanceAuditor().audit([TextSection("Results", "x")], [artifact])
    comparisons = {d["arm"]: d for d in result.details if d["kind"] == "recompute"}
    assert comparisons["baseline"]["verdict"] == "RECOMPUTED_MATCH"
    assert comparisons["candidate"]["verdict"] == "RECOMPUTED_MISMATCH"


def test_recompute_invalid_arrays_recorded_not_raised() -> None:
    row = {
        "seed": 1,
        "candidate": {"accuracy": 0.5},
        "raw_outputs": {"predictions": [1, 0, 1], "labels": [1, 0]},  # length mismatch
    }
    artifact = _artifact("artifacts/measurement.json", json.dumps({"seeds": [row]}))
    result = NumericProvenanceAuditor().audit([TextSection("Results", "x")], [artifact])
    comparison = next(d for d in result.details if d["kind"] == "recompute")
    assert comparison["verdict"] == "NOT_RECOMPUTABLE"
    assert "METRIC_RECOMPUTATION_INVALID" in comparison["reason"]
    assert result.counts["metrics_recomputable"] == 0
    assert result.rates["recomputation_match_rate"] is None  # never fabricates a rate


# ----------------------------------------------------------------------
# Envelope shape (integration contract for report.py)
# ----------------------------------------------------------------------


def test_result_to_section_metric_envelope() -> None:
    sections = [TextSection("Results", "Accuracy 0.5 from Figure 2.")]
    section = NumericProvenanceAuditor().audit(sections).to_section()
    assert set(section) == {
        "status",
        "not_run_reason",
        "llm_assisted",
        "counts",
        "rates",
        "coverage",
        "details",
    }
    assert section["status"] == "RUN_COMPLETE"
    assert section["llm_assisted"] is False
    assert set(section["coverage"]) == {
        "items_examined",
        "items_skipped",
        "skip_reasons",
    }
    assert all(isinstance(value, int) for value in section["counts"].values())
    for rate in section["rates"].values():
        assert rate is None or 0.0 <= rate <= 1.0
