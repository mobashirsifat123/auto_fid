"""Unit tests for the TrustAuditReport v1 writer/loader (audit/report.py).

Hermetic: no network; bundles are built under ``tmp_path``. The §8.3
no-fabrication guarantee is under test: every rate must be re-derivable
from recorded counts, the report must validate against the frozen schema,
and the checksum must refuse drift.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from autoresearch.audit.contract import AutoresearchAdapter, FileRole
from autoresearch.audit.report import (
    REPORT_SCHEMA,
    AuditRefusal,
    CoverageInfo,
    FailureRecord,
    LlmUsageRecord,
    MetricSection,
    TrustAuditReport,
    bundle_to_report_input,
    load_report,
    write_report,
)

CITATION_COUNTS = {
    "total_entries": 4,
    "resolved_exact": 3,
    "resolved_misattributed": 0,
    "unresolvable": 1,
}
NUMBER_COUNTS = {
    "numeric_tokens_examined": 10,
    "traced_artifact": 4,
    "traced_source": 3,
    "structural": 2,
    "untraced": 1,
    "recomputable": 0,
    "recomputed_match": 0,
}


@pytest.fixture
def bundle_root(tmp_path: Path) -> Path:
    root = tmp_path / "paper"
    root.mkdir()
    (root / "paper.md").write_text("# Abstract\nWe report 92.3%.\n", encoding="utf-8")
    (root / "references.bib").write_text(
        "@article{k1, author={A, B}, title={T}, year={2020}}\n", encoding="utf-8"
    )
    return root


def _report(bundle_root: Path) -> TrustAuditReport:
    bundle = AutoresearchAdapter().collect(bundle_root)
    return TrustAuditReport(
        input=bundle_to_report_input(bundle),
        metrics={
            "citations": MetricSection.complete(
                "citations",
                counts=CITATION_COUNTS,
                coverage=CoverageInfo(items_examined=4),
                details=[{"bib_key": "k1", "verdict": "RESOLVED_EXACT"}],
            ),
            "numbers": MetricSection.complete(
                "numbers",
                counts=NUMBER_COUNTS,
                coverage=CoverageInfo(items_examined=10, items_skipped=1, skip_reasons={"fixture": 1}),
            ),
            "claims": MetricSection.not_run("claims", "not_implemented_yet"),
            "overclaim": MetricSection.not_run("overclaim", "not_implemented_yet"),
            "disclosure": MetricSection.not_run("disclosure", "NO_GROUND_TRUTH"),
            "reproducibility": MetricSection.not_run("reproducibility", "no_commands"),
        },
    )


# ----------------------------------------------------------------------
# Write / load round trip
# ----------------------------------------------------------------------


def test_write_and_load_round_trip(tmp_path: Path, bundle_root: Path) -> None:
    report = _report(bundle_root)
    path = tmp_path / "report.json"
    checksum = write_report(report, path)
    assert checksum == report.checksum()

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["schema"] == "trust-audit-report/v1"
    assert payload["report_checksum"] == checksum

    loaded = load_report(path)
    assert loaded.report_checksum == checksum
    assert loaded.input.bundle.system_id == "autoresearch"
    assert loaded.metrics["citations"] == report.metrics["citations"]
    assert loaded.metrics["disclosure"].status == "NOT_RUN"
    assert loaded.metrics["disclosure"].not_run_reason == "NO_GROUND_TRUTH"
    assert loaded.metrics["numbers"].coverage.skip_reasons == {"fixture": 1}


def test_report_id_is_32_hex_and_checksum_is_64_hex(tmp_path: Path, bundle_root: Path) -> None:
    report = _report(bundle_root)
    path = tmp_path / "report.json"
    write_report(report, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    import re

    assert re.fullmatch(r"[0-9a-f]{32}", payload["report_id"])
    assert re.fullmatch(r"[0-9a-f]{64}", payload["report_checksum"])


def test_deterministic_checksum_for_identical_reports(bundle_root: Path) -> None:
    first = _report(bundle_root)
    second = replace(first, report_id="0" * 32, generated_at="2026-10-16T00:00:00+00:00")
    third = replace(second, metrics={**second.metrics, "citations": MetricSection.complete(
        "citations", counts={**CITATION_COUNTS, "resolved_exact": 2, "unresolvable": 2},
    )})
    assert second.checksum() == replace(second, report_id="0" * 32).checksum()
    assert second.checksum() != third.checksum()


# ----------------------------------------------------------------------
# Rate derivation (§8.3 step 2)
# ----------------------------------------------------------------------


def test_rates_are_derived_from_counts(bundle_root: Path) -> None:
    section = MetricSection.complete("citations", counts=CITATION_COUNTS)
    assert section.rates["citation_trust_rate"] == pytest.approx(0.75)
    assert section.rates["unverifiable_rate"] == pytest.approx(0.25)
    assert section.rates["misattribution_rate"] == 0.0


def test_zero_denominator_yields_null_rate() -> None:
    section = MetricSection.complete("numbers", counts=NUMBER_COUNTS)
    assert section.rates["recomputation_match_rate"] is None  # recomputable = 0
    assert section.rates["numeric_provenance_rate"] == pytest.approx(0.9)


def test_not_run_sections_have_all_null_rates() -> None:
    section = MetricSection.not_run("citations", "network_down")
    assert section.status == "NOT_RUN"
    assert set(section.rates) == {"citation_trust_rate", "unverifiable_rate", "misattribution_rate"}
    assert all(value is None for value in section.rates.values())
    assert section.counts == {}


def test_hand_edited_rate_is_refused(tmp_path: Path, bundle_root: Path) -> None:
    report = _report(bundle_root)
    tampered = MetricSection(
        metric="citations", status="RUN_COMPLETE", llm_assisted=False,
        counts=dict(CITATION_COUNTS),
        rates={"citation_trust_rate": 0.99, "unverifiable_rate": 0.25, "misattribution_rate": 0.0},
        coverage=CoverageInfo(items_examined=4),
    )
    report = replace(report, metrics={**report.metrics, "citations": tampered})
    with pytest.raises(AuditRefusal) as excinfo:
        write_report(report, tmp_path / "report.json")
    assert excinfo.value.code == "RATE_NOT_DERIVED"
    assert "citation_trust_rate" in excinfo.value.detail


def test_unregistered_rate_is_refused(tmp_path: Path, bundle_root: Path) -> None:
    report = _report(bundle_root)
    section = MetricSection(
        metric="citations", status="RUN_COMPLETE", llm_assisted=False,
        counts=dict(CITATION_COUNTS),
        rates={**dict(report.metrics["citations"].rates), "mystery_rate": 0.5},
        coverage=CoverageInfo(items_examined=4),
    )
    report = replace(report, metrics={**report.metrics, "citations": section})
    with pytest.raises(AuditRefusal) as excinfo:
        write_report(report, tmp_path / "report.json")
    assert excinfo.value.code == "RATE_NOT_DERIVED"
    assert "mystery_rate" in excinfo.value.detail


def test_run_complete_with_missing_rate_is_refused(tmp_path: Path, bundle_root: Path) -> None:
    report = _report(bundle_root)
    section = MetricSection(
        metric="citations", status="RUN_COMPLETE", llm_assisted=False,
        counts=dict(CITATION_COUNTS),
        rates={"citation_trust_rate": 0.75},  # other declared rates missing
        coverage=CoverageInfo(items_examined=4),
    )
    report = replace(report, metrics={**report.metrics, "citations": section})
    with pytest.raises(AuditRefusal) as excinfo:
        write_report(report, tmp_path / "report.json")
    assert excinfo.value.code == "RATE_NOT_DECLARED"


def test_missing_count_key_is_refused(tmp_path: Path, bundle_root: Path) -> None:
    counts = {k: v for k, v in CITATION_COUNTS.items() if k != "unresolvable"}
    section = MetricSection(
        metric="citations", status="RUN_COMPLETE", llm_assisted=False,
        counts=counts, rates={"citation_trust_rate": 0.75, "unverifiable_rate": 0.0,
                              "misattribution_rate": 0.0},
        coverage=CoverageInfo(items_examined=4),
    )
    report = _report(bundle_root)
    report = replace(report, metrics={**report.metrics, "citations": section})
    with pytest.raises(AuditRefusal) as excinfo:
        write_report(report, tmp_path / "report.json")
    assert excinfo.value.code == "COUNT_MISSING"


def test_not_run_with_non_null_rate_is_refused(tmp_path: Path, bundle_root: Path) -> None:
    report = _report(bundle_root)
    section = MetricSection(
        metric="claims", status="NOT_RUN", llm_assisted=False, counts={},
        rates={"strong_support_rate": 0.5, "citation_backed_rate": None},
        not_run_reason="offline",
    )
    report = replace(report, metrics={**report.metrics, "claims": section})
    with pytest.raises(AuditRefusal) as excinfo:
        write_report(report, tmp_path / "report.json")
    assert excinfo.value.code == "NOT_RUN_WITH_RATE"


def test_invalid_count_type_is_refused_at_construction() -> None:
    with pytest.raises(AuditRefusal) as excinfo:
        MetricSection.complete("citations", counts={**CITATION_COUNTS, "total_entries": "4"})
    assert excinfo.value.code == "COUNT_INVALID"


# ----------------------------------------------------------------------
# Schema enforcement
# ----------------------------------------------------------------------


def test_missing_metric_section_is_refused(tmp_path: Path, bundle_root: Path) -> None:
    report = _report(bundle_root)
    metrics = {k: v for k, v in report.metrics.items() if k != "overclaim"}
    with pytest.raises(AuditRefusal) as excinfo:
        write_report(replace(report, metrics=metrics), tmp_path / "report.json")
    assert excinfo.value.code == "SCHEMA_INVALID"
    assert "overclaim" in excinfo.value.detail


def test_rate_above_one_is_refused_by_schema(tmp_path: Path, bundle_root: Path) -> None:
    counts = {**CITATION_COUNTS, "resolved_exact": 5, "unresolvable": 0}
    report = _report(bundle_root)
    section = MetricSection.complete("citations", counts=counts)
    report = replace(report, metrics={**report.metrics, "citations": section})
    with pytest.raises(AuditRefusal) as excinfo:
        write_report(report, tmp_path / "report.json")
    assert excinfo.value.code == "SCHEMA_INVALID"
    assert "maximum" in excinfo.value.detail


def test_llm_sole_basis_true_is_refused(tmp_path: Path, bundle_root: Path) -> None:
    report = replace(_report(bundle_root), llm_usage=(
        LlmUsageRecord(
            metric="claims", purpose="extraction", provider="p", model="m",
            prompt_version="v1", calls=1, sole_basis=True,
        ),
    ))
    with pytest.raises(AuditRefusal) as excinfo:
        write_report(report, tmp_path / "report.json")
    assert excinfo.value.code == "SCHEMA_INVALID"
    assert "sole_basis" in excinfo.value.detail


def test_invalid_file_role_is_refused(bundle_root: Path, tmp_path: Path) -> None:
    report = _report(bundle_root)
    files = list(report.input.bundle.files)
    if files:
        import dataclasses

        files[0] = dataclasses.replace(files[0], role="mystery_role")
        report = replace(report, input=dataclasses.replace(
            report.input, bundle=dataclasses.replace(report.input.bundle, files=tuple(files))))
        with pytest.raises(AuditRefusal) as excinfo:
            write_report(report, tmp_path / "report.json")
        assert excinfo.value.code == "SCHEMA_INVALID"


# ----------------------------------------------------------------------
# Loader drift refusal
# ----------------------------------------------------------------------


def test_checksum_drift_is_refused(tmp_path: Path, bundle_root: Path) -> None:
    path = tmp_path / "report.json"
    write_report(_report(bundle_root), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metrics"]["citations"]["counts"]["resolved_exact"] = 4
    payload["metrics"]["citations"]["counts"]["unresolvable"] = 0
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(AuditRefusal) as excinfo:
        load_report(path)
    assert excinfo.value.code == "CHECKSUM_MISMATCH"


def test_tampered_rate_with_forged_checksum_is_refused_by_self_check(
    tmp_path: Path, bundle_root: Path,
) -> None:
    path = tmp_path / "report.json"
    write_report(_report(bundle_root), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metrics"]["citations"]["rates"]["citation_trust_rate"] = 1.0
    # Forge the outer checksum so only the §8.3 self-check can catch it.
    from autoresearch.domain.models import content_checksum

    payload.pop("report_checksum")
    payload["report_checksum"] = content_checksum(payload)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    with pytest.raises(AuditRefusal) as excinfo:
        load_report(path)
    assert excinfo.value.code == "RATE_NOT_DERIVED"


def test_not_json_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "report.json"
    path.write_text("not json at all", encoding="utf-8")
    with pytest.raises(AuditRefusal) as excinfo:
        load_report(path)
    assert excinfo.value.code == "NOT_JSON"


def test_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(AuditRefusal) as excinfo:
        load_report(tmp_path / "missing.json")
    assert excinfo.value.code == "REPORT_UNREADABLE"


# ----------------------------------------------------------------------
# Full pipeline against an Autoresearch-style bundle
# ----------------------------------------------------------------------


def test_full_report_includes_manifest_and_failures(tmp_path: Path) -> None:
    root = tmp_path / "manuscripts" / "m1"
    root.mkdir(parents=True)
    (root / "manuscript.md").write_text("# Abstract\nHonest text.\n", encoding="utf-8")
    (root / "references.bib").write_text(
        "@article{k1, author={A, B}, title={T}, year={2020}}\n", encoding="utf-8"
    )
    (root / "evidence_bundle.json").write_text(json.dumps({
        "lineage": {"gap_id": "g", "spec_id": "s", "analysis_id": "a", "report_id": "r"},
        "analysis": {"seed_level_results": [{"seed": 7, "accuracy": 0.5}]},
    }), encoding="utf-8")
    (root / "stray.dat").write_bytes(b"\x01")

    bundle = AutoresearchAdapter().collect(root)
    report = TrustAuditReport(
        input=bundle_to_report_input(bundle),
        metrics={
            "citations": MetricSection.complete("citations", counts=CITATION_COUNTS),
            "claims": MetricSection.not_run("claims", "not_implemented_yet"),
            "numbers": MetricSection.not_run("numbers", "not_implemented_yet"),
            "overclaim": MetricSection.not_run("overclaim", "not_implemented_yet"),
            "disclosure": MetricSection.complete("disclosure", counts={
                "audited_completeness_claims": 0, "disclosed_claims": 0,
            }),
            "reproducibility": MetricSection.not_run("reproducibility", "no_commands"),
        },
        failures=(FailureRecord(metric="citations", code="SOURCE_TIMEOUT", detail="offline"),),
    )
    path = tmp_path / "report.json"
    write_report(report, path)
    loaded = load_report(path)

    roles = {f.path: f.role for f in loaded.input.bundle.files}
    assert roles["manuscript.md"] == "paper_source"
    assert roles["evidence_bundle.json"] == "artifact"
    assert loaded.input.bundle.excluded_files == ("stray.dat",)
    assert loaded.input.bundle.ground_truth_tier == "structured"
    # Zero denominator: verdict-only disclosure (§5.5), rate stays null.
    assert loaded.metrics["disclosure"].rates["disclosure_honesty_rate"] is None
    assert loaded.failures[0].code == "SOURCE_TIMEOUT"


def test_schema_of_record_matches_design_section_8_1() -> None:
    assert REPORT_SCHEMA["$id"] == "autoresearch/trust-audit-report/v1"
    assert set(REPORT_SCHEMA["required"]) == {
        "schema", "report_id", "generated_at", "harness", "input",
        "metrics", "llm_usage", "failures", "report_checksum",
    }
    assert set(REPORT_SCHEMA["properties"]["metrics"]["required"]) == {
        "citations", "claims", "numbers", "overclaim", "disclosure", "reproducibility",
    }
    bundle_props = REPORT_SCHEMA["properties"]["input"]["properties"]["bundle"]
    assert set(bundle_props["required"]) == {
        "system_id", "adapter_name", "adapter_version", "root",
        "files", "excluded_files", "text_extraction",
    }
    file_role_enum = bundle_props["properties"]["files"]["items"]["properties"]["role"]["enum"]
    assert file_role_enum == [role.value for role in FileRole]
    metric_section = REPORT_SCHEMA["$defs"]["metricSection"]
    assert metric_section["properties"]["status"]["enum"] == ["RUN_COMPLETE", "NOT_RUN"]
    assert metric_section["properties"]["llm_assisted"]["type"] == "boolean"
    assert REPORT_SCHEMA["properties"]["llm_usage"]["items"]["properties"]["sole_basis"]["const"] is False


def test_harness_defaults_are_recorded(bundle_root: Path, tmp_path: Path) -> None:
    report = _report(bundle_root)
    path = tmp_path / "report.json"
    write_report(report, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    harness = payload["harness"]
    assert harness["harness_version"] == "autoresearch-audit/0.1.0"
    assert harness["frozen"] is False
    assert harness["python_version"].count(".") == 2
    assert harness["package_version"]
    assert payload["llm_usage"] == []
