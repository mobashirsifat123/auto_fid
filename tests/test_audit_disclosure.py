"""Hermetic unit tests for autoresearch.audit.disclosure (DESIGN §5.5).

Covers: NO_GROUND_TRUTH fail-closed behavior, the injected RunLogParser
protocol, DISCLOSED / MASKED / OVERSTATED verdicts (including the
declared-but-never-run ablation soft spot), the disclosure honesty rate,
and the metricSection envelope shape. No network, no LLM.
"""

from __future__ import annotations

import json

from autoresearch.audit.disclosure import (
    DisclosureVerdict,
    FailureDisclosureAuditor,
    LogFile,
    RunLogGroundTruth,
    RunLogParser,
    TextSection,
)
from autoresearch.domain.models import content_checksum


def _log(path: str, text: str) -> LogFile:
    return LogFile(path=path, sha256=content_checksum(text), text=text)


class JsonLogParser:
    """Deterministic stub parser: one JSON run-log object per file."""

    def parse(self, logs: list[LogFile]) -> RunLogGroundTruth | None:
        if not logs:
            return None
        payload = json.loads(logs[0].text)
        return RunLogGroundTruth(
            source_files=(logs[0].path,),
            runs_total=payload.get("runs_total"),
            runs_failed=payload.get("runs_failed"),
            seeds_executed=tuple(payload["seeds_executed"])
            if payload.get("seeds_executed") is not None
            else None,
            stages_executed=tuple(payload["stages_executed"])
            if payload.get("stages_executed") is not None
            else None,
            stages_failed_or_skipped=(
                tuple(payload["stages_failed_or_skipped"])
                if payload.get("stages_failed_or_skipped") is not None
                else None
            ),
        )


class NoneParser:
    def parse(self, logs: list[LogFile]) -> RunLogGroundTruth | None:
        return None


# ----------------------------------------------------------------------
# Fail-closed: no ground truth => NOT_RUN, never a pass
# ----------------------------------------------------------------------


def test_no_parser_means_not_run() -> None:
    paper = [TextSection("Abstract", "We report all results. n = 5 seeds.")]
    result = FailureDisclosureAuditor().audit(paper)
    assert result.status == "NOT_RUN"
    assert result.not_run_reason == "NO_GROUND_TRUTH"
    assert result.verdict is None
    assert result.rates["disclosure_honesty_rate"] is None
    section = result.to_section()
    assert section["status"] == "NOT_RUN"


def test_parser_returning_none_means_not_run() -> None:
    paper = [TextSection("Abstract", "We report all results. n = 5 seeds.")]
    logs = [_log("logs/run.json", json.dumps({"runs_failed": 0}))]
    result = FailureDisclosureAuditor().audit(paper, logs=logs, parser=NoneParser())
    assert result.status == "NOT_RUN"
    assert result.not_run_reason == "NO_GROUND_TRUTH"


def test_no_paper_text_means_not_run() -> None:
    result = FailureDisclosureAuditor().audit(
        [TextSection("Abstract", "")], ground_truth=RunLogGroundTruth(runs_failed=0)
    )
    assert result.status == "NOT_RUN"
    assert result.not_run_reason == "no_paper_text"


# ----------------------------------------------------------------------
# DISCLOSED: claims match logs, failures reported
# ----------------------------------------------------------------------


def test_disclosed_when_failures_reported_honestly() -> None:
    paper = [
        TextSection("Results", "2/5 seeds failed and are reported here. n = 5."),
        TextSection("Limitations and Threats to Validity", "Coverage is limited."),
    ]
    truth = RunLogGroundTruth(
        source_files=("logs/run.json",),
        runs_total=5,
        runs_failed=2,
        seeds_executed=(1, 2, 3, 4, 5),
        stages_executed=("training", "evaluation"),
        stages_failed_or_skipped=(),
    )
    result = FailureDisclosureAuditor().audit(paper, ground_truth=truth)
    assert result.verdict is DisclosureVerdict.DISCLOSED
    assert result.counts["masked_failure_count"] == 0
    assert result.counts["discrepancy_count"] == 0
    assert (
        result.counts["audited_completeness_claims"] == 2
    )  # seed count + failure disclosure
    assert result.counts["disclosed_claims"] == 2
    assert result.rates["disclosure_honesty_rate"] == 1.0
    verdict_detail = next(d for d in result.details if d["kind"] == "verdict")
    assert verdict_detail["verdict"] == "DISCLOSED"
    ground_truth_detail = next(d for d in result.details if d["kind"] == "ground_truth")
    assert ground_truth_detail["source_files"] == ["logs/run.json"]


def test_disclosed_via_parser_protocol() -> None:
    paper = [TextSection("Results", "n = 3 seeds were executed.")]
    logs = [
        _log(
            "logs/run.json",
            json.dumps(
                {"runs_total": 3, "runs_failed": 0, "seeds_executed": [1, 2, 3]}
            ),
        )
    ]
    result = FailureDisclosureAuditor().audit(paper, logs=logs, parser=JsonLogParser())
    assert result.status == "RUN_COMPLETE"
    assert result.verdict is DisclosureVerdict.DISCLOSED


def test_explicit_incomplete_statement_counts_as_disclosure() -> None:
    paper = [
        TextSection(
            "Abstract",
            "This manuscript is explicitly INCOMPLETE: no result is fabricated to fill the gaps. n = 2.",
        )
    ]
    truth = RunLogGroundTruth(
        source_files=("logs/run.json",),
        runs_failed=3,
        seeds_executed=(1, 2),
        stages_failed_or_skipped=("ablation_stage",),
    )
    result = FailureDisclosureAuditor().audit(paper, ground_truth=truth)
    assert result.verdict is not DisclosureVerdict.MASKED
    masked_kinds = [
        d for d in result.details if d.get("discrepancy_kind", "").startswith("MASKED")
    ]
    assert masked_kinds == []


# ----------------------------------------------------------------------
# MASKED: failures present, paper silent
# ----------------------------------------------------------------------


def test_masked_failures_detected() -> None:
    paper = [
        TextSection("Abstract", "We report all results."),
        TextSection("Results", "n = 5 seeds were executed with consistent outcomes."),
    ]
    truth = RunLogGroundTruth(
        source_files=("logs/run.json",),
        runs_total=5,
        runs_failed=3,
        seeds_executed=(1, 2),
        stages_executed=("training",),
        stages_failed_or_skipped=("ablation_stage",),
    )
    result = FailureDisclosureAuditor().audit(paper, ground_truth=truth)
    assert result.verdict is DisclosureVerdict.MASKED
    assert (
        result.counts["masked_failure_count"] == 4
    )  # 3 failed runs + 1 failed/skipped stage
    assert result.counts["discrepancy_count"] >= 2
    assert result.rates["disclosure_honesty_rate"] == 0.0
    kinds = {
        d["discrepancy_kind"] for d in result.details if d["kind"] == "discrepancy"
    }
    assert "MASKED_FAILURES" in kinds
    assert "MASKED_COMPLETENESS" in kinds
    assert "OVERSTATED_SEEDS" in kinds  # claimed 5, logs executed 2
    masked = next(
        d for d in result.details if d.get("discrepancy_kind") == "MASKED_FAILURES"
    )
    assert masked["evidence"] == ["logs/run.json"]
    assert masked["ground_truth"] == "4 failed/skipped"


def test_masked_beats_overstated_in_verdict_order() -> None:
    paper = [TextSection("Abstract", "we report all")]
    truth = RunLogGroundTruth(
        runs_total=5, runs_failed=1, seeds_executed=(1,), stages_failed_or_skipped=()
    )
    result = FailureDisclosureAuditor().audit(paper, ground_truth=truth)
    assert result.verdict is DisclosureVerdict.MASKED


# ----------------------------------------------------------------------
# OVERSTATED: paper claims more than logs evidence
# ----------------------------------------------------------------------


def test_overstated_seed_count() -> None:
    paper = [TextSection("Results", "n = 5 seeds were executed.")]
    truth = RunLogGroundTruth(
        source_files=("logs/run.json",), seeds_executed=(1, 2), runs_failed=0
    )
    result = FailureDisclosureAuditor().audit(paper, ground_truth=truth)
    assert result.verdict is DisclosureVerdict.OVERSTATED
    kinds = {
        d["discrepancy_kind"] for d in result.details if d["kind"] == "discrepancy"
    }
    assert kinds == {"OVERSTATED_SEEDS"}
    discrepancy = next(d for d in result.details if d["kind"] == "discrepancy")
    assert discrepancy["claimed"] == "5"
    assert discrepancy["ground_truth"] == "2"
    assert result.rates["disclosure_honesty_rate"] == 0.0


def test_overstated_declared_but_never_run_ablation() -> None:
    # Roadmap soft spot #4: declared ablation stages that never executed.
    paper = [TextSection("Ablations", "We report 2 ablations for robustness.")]
    truth = RunLogGroundTruth(
        source_files=("logs/run.json",),
        runs_failed=0,
        stages_executed=("training", "evaluation"),
    )
    result = FailureDisclosureAuditor().audit(paper, ground_truth=truth)
    assert result.verdict is DisclosureVerdict.OVERSTATED
    kinds = {
        d["discrepancy_kind"] for d in result.details if d["kind"] == "discrepancy"
    }
    assert kinds == {"OVERSTATED_ABLATIONS"}


def test_ablation_claim_disclosed_when_stage_executed() -> None:
    paper = [TextSection("Ablations", "We report 1 ablation for robustness.")]
    truth = RunLogGroundTruth(
        runs_failed=0, stages_executed=("training", "ablation_attention")
    )
    result = FailureDisclosureAuditor().audit(paper, ground_truth=truth)
    assert result.verdict is DisclosureVerdict.DISCLOSED
    assert result.rates["disclosure_honesty_rate"] == 1.0


def test_overstated_run_count_uses_runs_total() -> None:
    paper = [TextSection("Results", "We ran 4 runs.")]
    truth = RunLogGroundTruth(
        source_files=("logs/run.json",), runs_total=2, runs_failed=0
    )
    result = FailureDisclosureAuditor().audit(paper, ground_truth=truth)
    assert result.verdict is DisclosureVerdict.OVERSTATED
    kinds = {
        d["discrepancy_kind"] for d in result.details if d["kind"] == "discrepancy"
    }
    assert kinds == {"OVERSTATED_RUNS"}


# ----------------------------------------------------------------------
# Missing ground-truth fields are skipped, never silently passed
# ----------------------------------------------------------------------


def test_missing_truth_fields_are_skips_not_passes() -> None:
    paper = [
        TextSection("Abstract", "We report all results. n = 5. We report 2 ablations.")
    ]
    truth = RunLogGroundTruth(source_files=("logs/run.json",))  # nothing derivable
    result = FailureDisclosureAuditor().audit(paper, ground_truth=truth)
    assert (
        result.verdict is DisclosureVerdict.DISCLOSED
    )  # nothing checkable => verdict-only
    assert (
        result.rates["disclosure_honesty_rate"] is None
    )  # denominator 0 => verdict-only
    assert result.coverage["items_skipped"] == 3
    assert set(result.coverage["skip_reasons"]) == {
        "no_seed_truth",
        "no_failure_truth",
        "no_stage_truth",
    }


def test_inaccurate_failure_disclosure_is_a_discrepancy() -> None:
    paper = [TextSection("Results", "1/5 seeds failed. n = 5.")]
    truth = RunLogGroundTruth(
        runs_total=5, runs_failed=2, seeds_executed=(1, 2, 3, 4, 5)
    )
    result = FailureDisclosureAuditor().audit(paper, ground_truth=truth)
    kinds = {
        d["discrepancy_kind"] for d in result.details if d["kind"] == "discrepancy"
    }
    assert "INACCURATE_FAILURE_DISCLOSURE" in kinds
    assert result.verdict is DisclosureVerdict.PARTIALLY_DISCLOSED


# ----------------------------------------------------------------------
# Envelope shape (integration contract for report.py)
# ----------------------------------------------------------------------


def test_result_to_section_metric_envelope() -> None:
    paper = [TextSection("Results", "n = 2. 1/2 runs failed.")]
    truth = RunLogGroundTruth(runs_total=2, runs_failed=1, seeds_executed=(1, 2))
    section = FailureDisclosureAuditor().audit(paper, ground_truth=truth).to_section()
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
    assert section["counts"]["masked_failure_count"] == 0
    assert set(section["coverage"]) == {
        "items_examined",
        "items_skipped",
        "skip_reasons",
    }
    for rate in section["rates"].values():
        assert rate is None or 0.0 <= rate <= 1.0
    assert section["details"][0]["kind"] == "verdict"


def test_run_log_parser_protocol_structural_check() -> None:
    parser: RunLogParser = JsonLogParser()
    logs = [_log("logs/run.json", json.dumps({"runs_total": 1}))]
    truth = parser.parse(logs)
    assert truth is not None and truth.runs_total == 1
    assert parser.parse([]) is None
