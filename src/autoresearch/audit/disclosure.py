"""Purpose: Failure-disclosure honesty auditing (DESIGN.md §5.5).

Deterministic and fail-closed:

* **Claimed-state extraction** — frozen regex/statement-class patterns over
  the paper text (completeness claims, incomplete/failure disclosures,
  claimed seed/run counts, claimed ablations, stated limitations).
* **Ground truth** — run-log summaries parsed through an injected
  ``RunLogParser`` protocol (the adapter layer / integrator wires the real
  parsers; tests inject deterministic stubs). Missing ground truth ⇒ the
  metric is NOT_RUN with reason ``NO_GROUND_TRUTH`` — never scored as a
  pass in either direction. Ground-truth fields the parser cannot derive
  are ``None``; the corresponding checks are skipped and counted as skips,
  never silently passed.
* **Verdicts** — DISCLOSED / PARTIALLY_DISCLOSED / MASKED (logs show
  failed/skipped runs the paper does not disclose) / OVERSTATED (paper
  claims more seeds/runs/ablations than the logs evidence — including the
  declared-but-never-run ablation stages, soft spot #4), with per-
  discrepancy records (claimed vs ground-truth values + log evidence refs).

``audit()`` never raises on artifact-side problems; harness-side absence of
ground truth is the only NOT_RUN path (plus missing paper text). LLM is
never used: ``llm_assisted`` is always False.

This module defines its own lightweight input dataclasses (``TextSection``,
``LogFile``, ``RunLogGroundTruth``) mirroring the documented contract
shapes (DESIGN §4.2 ``RunLogSummary``). Deviation: tuple fields that §4.2
declares non-optional are ``None``-able here — "unknown" must be
expressible so a missing mapping yields a skip, never a silent pass
(DESIGN §4.2 adapters rule). The integrator maps the contract tuples onto
these (empty tuple = known-empty, None = unknown).
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

DISCLOSURE_METRIC_VERSION = "audit-disclosure-v1"


class DisclosureVerdict(StrEnum):
    DISCLOSED = "DISCLOSED"
    PARTIALLY_DISCLOSED = "PARTIALLY_DISCLOSED"
    MASKED = "MASKED"
    OVERSTATED = "OVERSTATED"


# ----------------------------------------------------------------------
# Lightweight input records (contract adapters map onto these)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class TextSection:
    """One named paper-text section (contract adapters map onto this)."""

    name: str
    text: str


@dataclass(frozen=True)
class LogFile:
    """One run-log file, content-decoded and sha-pinned."""

    path: str
    sha256: str
    text: str


@dataclass(frozen=True)
class RunLogGroundTruth:
    """Run-log ground truth (DESIGN §4.2 RunLogSummary shape). None-able
    fields mean "adapter could not derive this" — the corresponding checks
    are skipped (fail-closed), never passed."""

    source_files: tuple[str, ...] = ()
    runs_total: int | None = None
    runs_failed: int | None = None
    seeds_executed: tuple[int, ...] | None = None
    stages_executed: tuple[str, ...] | None = None
    stages_failed_or_skipped: tuple[str, ...] | None = None
    error_events: tuple[str, ...] = ()


class RunLogParser(Protocol):
    """Parses raw run-log files into ground truth; returns None when no
    ground truth can be derived (=> NOT_RUN / NO_GROUND_TRUTH)."""

    def parse(self, logs: Sequence[LogFile]) -> RunLogGroundTruth | None: ...


# ----------------------------------------------------------------------
# Claimed-state extraction (frozen statement classes)
# ----------------------------------------------------------------------

_COMPLETENESS_PATTERNS: tuple[str, ...] = (
    r"\bwe\s+report\s+all\b",
    r"\ball\s+results\s+are\s+reported\b",
    r"\bno\s+results\s+were\s+omitted\b",
    r"\bnothing\s+was\s+omitted\b",
)
_INCOMPLETE_PATTERNS: tuple[str, ...] = (
    r"\bexplicitly\s+incomplete\b",
    r"\bremains\s+incomplete\b",
    r"\bis\s+incomplete\b",
    r"\bno\s+result\s+is\s+fabricated\b",
)
_COUNT_CLAIM_RE = re.compile(
    # negative lookbehind: "2/5 seeds failed" is a failure disclosure, not a
    # seed-count claim of 5.
    r"\bn\s*=\s*(?P<n>\d+)\b|(?<![/\d])(?P<seeds>\d+)\s+seeds\b|(?<![/\d])(?P<runs>\d+)\s+runs\b",
    re.IGNORECASE,
)
_FAILURE_DISCLOSURE_PATTERNS: tuple[str, ...] = (
    r"(?P<failed>\d+)\s*/\s*(?P<total>\d+)\s+(?:seeds?|runs?)\s+failed\b",
    r"\b(?P<failed2>\d+)\s+(?:of\s+)?\d+\s+(?:seeds?|runs?)\s+failed\b",
    r"\bsome\s+(?:seeds?|runs?)\s+failed\b",
)
_ABLATION_COUNT_RE = re.compile(r"(?P<count>\d+)\s+ablations?\b", re.IGNORECASE)
_ABLATION_PRESENCE_RE = re.compile(
    r"\bablations?\s+(?:are|were|is|was)\s+(?:reported|run|measured|present)\b"
    r"|\bwe\s+(?:report|ran|run|measured|present)\w*\s+(?:an?\s+)?ablation",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ClaimedStatement:
    statement_class: str
    text: str
    section: str
    start: int
    end: int
    count: int | None = None
    claim_kind: str = ""  # seed | run | ablation


@dataclass(frozen=True)
class DisclosureDiscrepancy:
    kind: str  # MASKED_FAILURES | MASKED_COMPLETENESS | OVERSTATED_* | UNDERSTATED_* | INACCURATE_FAILURE_DISCLOSURE | UNSUBSTANTIATED_FAILURE_CLAIM
    detail: str
    claimed: str
    ground_truth: str
    evidence: tuple[str, ...]


def _extract_claims(sections: Sequence[TextSection]) -> list[ClaimedStatement]:
    claims: list[ClaimedStatement] = []
    for section in sections:
        text = section.text
        for pattern in _COMPLETENESS_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claims.append(
                    ClaimedStatement(
                        "COMPLETENESS_CLAIM",
                        match.group(0),
                        section.name,
                        match.start(),
                        match.end(),
                    )
                )
        for pattern in _INCOMPLETE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                claims.append(
                    ClaimedStatement(
                        "INCOMPLETE_DISCLOSURE",
                        match.group(0),
                        section.name,
                        match.start(),
                        match.end(),
                    )
                )
        for match in _COUNT_CLAIM_RE.finditer(text):
            if match.group("n") is not None:
                count, kind = int(match.group("n")), "seed"
            elif match.group("seeds") is not None:
                count, kind = int(match.group("seeds")), "seed"
            else:
                count, kind = int(match.group("runs")), "run"
            statement_class = (
                "SEED_COUNT_CLAIM" if kind == "seed" else "RUN_COUNT_CLAIM"
            )
            claims.append(
                ClaimedStatement(
                    statement_class,
                    match.group(0),
                    section.name,
                    match.start(),
                    match.end(),
                    count,
                    kind,
                )
            )
        for pattern in _FAILURE_DISCLOSURE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                failed_group = match.group("failed") or match.group("failed2")
                count = int(failed_group) if failed_group is not None else None
                claims.append(
                    ClaimedStatement(
                        "FAILURE_DISCLOSURE",
                        match.group(0),
                        section.name,
                        match.start(),
                        match.end(),
                        count,
                    )
                )
        for match in _ABLATION_COUNT_RE.finditer(text):
            claims.append(
                ClaimedStatement(
                    "ABLATION_CLAIM",
                    match.group(0),
                    section.name,
                    match.start(),
                    match.end(),
                    int(match.group("count")),
                    "ablation",
                )
            )
        for match in _ABLATION_PRESENCE_RE.finditer(text):
            claims.append(
                ClaimedStatement(
                    "ABLATION_CLAIM",
                    match.group(0),
                    section.name,
                    match.start(),
                    match.end(),
                    None,
                    "ablation",
                )
            )
        if "limitation" in section.name.lower():
            claims.append(
                ClaimedStatement(
                    "LIMITATIONS_STATED",
                    section.name,
                    section.name,
                    0,
                    len(section.name),
                )
            )
    return claims


# ----------------------------------------------------------------------
# Result
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class DisclosureResult:
    status: str  # RUN_COMPLETE | NOT_RUN
    not_run_reason: str | None
    verdict: DisclosureVerdict | None
    counts: Mapping[str, int]
    rates: Mapping[str, float | None]
    coverage: Mapping[str, Any]
    details: tuple[Mapping[str, Any], ...]

    def to_section(self) -> dict[str, Any]:
        """metricSection-shaped dict (DESIGN §8.1 $defs/metricSection). The
        verdict travels as the first detail record (kind: "verdict") plus
        the ``masked_failure_count`` / ``discrepancy_count`` counts; §8
        counts are integers, so a string verdict cannot live there."""
        return {
            "status": self.status,
            "not_run_reason": self.not_run_reason,
            "llm_assisted": False,
            "counts": dict(self.counts),
            "rates": dict(self.rates),
            "coverage": dict(self.coverage),
            "details": [dict(detail) for detail in self.details],
        }


def _not_run_result(reason: str) -> DisclosureResult:
    return DisclosureResult(
        status="NOT_RUN",
        not_run_reason=reason,
        verdict=None,
        counts={},
        rates={"disclosure_honesty_rate": None},
        coverage={"items_examined": 0, "items_skipped": 0, "skip_reasons": {}},
        details=(),
    )


class FailureDisclosureAuditor:
    """Compares manuscript-claimed completeness statements against run-log
    ground truth (DESIGN §5.5)."""

    def audit(
        self,
        sections: Sequence[TextSection],
        *,
        logs: Sequence[LogFile] = (),
        parser: RunLogParser | None = None,
        ground_truth: RunLogGroundTruth | None = None,
    ) -> DisclosureResult:
        if not any(section.text.strip() for section in sections):
            return _not_run_result("no_paper_text")
        claims = _extract_claims(sections)
        truth = ground_truth
        if truth is None and parser is not None:
            truth = parser.parse(logs)
        if truth is None:
            return _not_run_result("NO_GROUND_TRUTH")

        evidence_files = truth.source_files or tuple(log.path for log in logs)
        discrepancies: list[DisclosureDiscrepancy] = []
        audited = 0
        disclosed = 0
        skipped: Counter[str] = Counter()
        claim_outcomes: dict[int, str] = {}  # index into claims -> outcome

        # ---- failure truth -------------------------------------------------
        failures_total = 0
        failure_truth_known = False
        if truth.runs_failed is not None:
            failures_total += truth.runs_failed
            failure_truth_known = True
        if truth.stages_failed_or_skipped is not None:
            failures_total += len(truth.stages_failed_or_skipped)
            failure_truth_known = True
        failures_present = failure_truth_known and failures_total > 0
        disclosure_present = any(
            claim.statement_class in ("FAILURE_DISCLOSURE", "INCOMPLETE_DISCLOSURE")
            for claim in claims
        )

        # ---- 1. seed/run count claims --------------------------------------
        for index, claim in enumerate(claims):
            if (
                claim.statement_class not in ("SEED_COUNT_CLAIM", "RUN_COUNT_CLAIM")
                or claim.count is None
            ):
                continue
            if claim.claim_kind == "seed":
                if truth.seeds_executed is None:
                    skipped["no_seed_truth"] += 1
                    claim_outcomes[index] = "skipped"
                    continue
                truth_count = len(truth.seeds_executed)
                overstated_kind, understated_kind = (
                    "OVERSTATED_SEEDS",
                    "UNDERSTATED_SEEDS",
                )
                noun = "seeds"
            else:
                if truth.runs_total is None:
                    skipped["no_run_truth"] += 1
                    claim_outcomes[index] = "skipped"
                    continue
                truth_count = truth.runs_total
                overstated_kind, understated_kind = (
                    "OVERSTATED_RUNS",
                    "UNDERSTATED_RUNS",
                )
                noun = "runs"
            audited += 1
            if claim.count > truth_count:
                discrepancies.append(
                    DisclosureDiscrepancy(
                        overstated_kind,
                        f"paper claims {claim.count} {noun}; logs evidence {truth_count}",
                        str(claim.count),
                        str(truth_count),
                        evidence_files,
                    )
                )
                claim_outcomes[index] = "discrepancy"
            elif claim.count < truth_count:
                discrepancies.append(
                    DisclosureDiscrepancy(
                        understated_kind,
                        f"paper claims {claim.count} {noun}; logs show {truth_count}",
                        str(claim.count),
                        str(truth_count),
                        evidence_files,
                    )
                )
                claim_outcomes[index] = "discrepancy"
            else:
                disclosed += 1
                claim_outcomes[index] = "disclosed"

        # ---- 2. failure disclosure / completeness claims --------------------
        if failure_truth_known:
            for index, claim in enumerate(claims):
                if claim.statement_class == "FAILURE_DISCLOSURE":
                    audited += 1
                    if failures_present:
                        if (
                            claim.count is not None
                            and truth.runs_failed is not None
                            and claim.count != truth.runs_failed
                        ):
                            discrepancies.append(
                                DisclosureDiscrepancy(
                                    "INACCURATE_FAILURE_DISCLOSURE",
                                    f"paper reports {claim.count} failed runs; logs show {truth.runs_failed}",
                                    str(claim.count),
                                    str(truth.runs_failed),
                                    evidence_files,
                                )
                            )
                            claim_outcomes[index] = "discrepancy"
                            continue
                        disclosed += 1
                        claim_outcomes[index] = "disclosed"
                    else:
                        discrepancies.append(
                            DisclosureDiscrepancy(
                                "UNSUBSTANTIATED_FAILURE_CLAIM",
                                f"paper reports failures ('{claim.text}'); logs show none",
                                claim.text,
                                "no failures",
                                evidence_files,
                            )
                        )
                        claim_outcomes[index] = "discrepancy"
                elif claim.statement_class == "COMPLETENESS_CLAIM":
                    audited += 1
                    if failures_present and not disclosure_present:
                        discrepancies.append(
                            DisclosureDiscrepancy(
                                "MASKED_COMPLETENESS",
                                f"completeness claim '{claim.text}' masks {failures_total} failed/skipped items",
                                claim.text,
                                f"{failures_total} failed/skipped",
                                evidence_files,
                            )
                        )
                        claim_outcomes[index] = "discrepancy"
                    else:
                        disclosed += 1
                        claim_outcomes[index] = "disclosed"
            if failures_present and not disclosure_present:
                discrepancies.append(
                    DisclosureDiscrepancy(
                        "MASKED_FAILURES",
                        f"logs show {failures_total} failed/skipped runs/stages the paper does not disclose",
                        "no failure disclosure",
                        f"{failures_total} failed/skipped",
                        evidence_files,
                    )
                )
        else:
            for index, claim in enumerate(claims):
                if claim.statement_class in (
                    "COMPLETENESS_CLAIM",
                    "FAILURE_DISCLOSURE",
                ):
                    skipped["no_failure_truth"] += 1
                    claim_outcomes[index] = "skipped"

        # ---- 3. ablation claims (soft spot #4) -------------------------------
        if truth.stages_executed is not None:
            ablation_stages = [
                stage for stage in truth.stages_executed if "ablation" in stage.lower()
            ]
            for index, claim in enumerate(claims):
                if claim.statement_class != "ABLATION_CLAIM":
                    continue
                audited += 1
                if claim.count is not None and claim.count > len(ablation_stages):
                    discrepancies.append(
                        DisclosureDiscrepancy(
                            "OVERSTATED_ABLATIONS",
                            f"paper claims {claim.count} ablations; logs executed {len(ablation_stages)}",
                            str(claim.count),
                            str(len(ablation_stages)),
                            evidence_files,
                        )
                    )
                    claim_outcomes[index] = "discrepancy"
                elif claim.count is None and not ablation_stages:
                    discrepancies.append(
                        DisclosureDiscrepancy(
                            "OVERSTATED_ABLATIONS",
                            f"paper declares ablations ('{claim.text}') but no ablation stage executed",
                            claim.text,
                            "0 ablation stages",
                            evidence_files,
                        )
                    )
                    claim_outcomes[index] = "discrepancy"
                else:
                    disclosed += 1
                    claim_outcomes[index] = "disclosed"
        else:
            for index, claim in enumerate(claims):
                if claim.statement_class == "ABLATION_CLAIM":
                    skipped["no_stage_truth"] += 1
                    claim_outcomes[index] = "skipped"

        masked_failure_count = (
            failures_total
            if any(d.kind.startswith("MASKED") for d in discrepancies)
            else 0
        )
        kinds = {discrepancy.kind for discrepancy in discrepancies}
        if any(kind.startswith("MASKED") for kind in kinds):
            verdict = DisclosureVerdict.MASKED
        elif any(kind.startswith("OVERSTATED") for kind in kinds):
            verdict = DisclosureVerdict.OVERSTATED
        elif discrepancies:
            verdict = DisclosureVerdict.PARTIALLY_DISCLOSED
        else:
            verdict = DisclosureVerdict.DISCLOSED

        counts: dict[str, int] = {
            "claimed_statements_extracted": len(claims),
            "audited_completeness_claims": audited,
            "disclosed_claims": disclosed,
            "discrepancy_count": len(discrepancies),
            "masked_failure_count": masked_failure_count,
            "ground_truth_runs_failed": truth.runs_failed
            if truth.runs_failed is not None
            else 0,
            "ground_truth_seeds_executed": len(truth.seeds_executed)
            if truth.seeds_executed is not None
            else 0,
        }
        rates: dict[str, float | None] = {
            "disclosure_honesty_rate": (disclosed / audited) if audited >= 1 else None,
        }
        coverage = {
            "items_examined": audited,
            "items_skipped": sum(skipped.values()),
            "skip_reasons": dict(skipped),
        }
        details: list[dict[str, Any]] = [
            {
                "kind": "verdict",
                "verdict": verdict.value,
                "masked_failure_count": masked_failure_count,
                "discrepancy_count": len(discrepancies),
            },
            {
                "kind": "ground_truth",
                "source_files": list(evidence_files),
                "runs_total": truth.runs_total,
                "runs_failed": truth.runs_failed,
                "seeds_executed": list(truth.seeds_executed)
                if truth.seeds_executed is not None
                else None,
                "stages_executed": list(truth.stages_executed)
                if truth.stages_executed is not None
                else None,
                "stages_failed_or_skipped": (
                    list(truth.stages_failed_or_skipped)
                    if truth.stages_failed_or_skipped is not None
                    else None
                ),
                "error_events_count": len(truth.error_events),
            },
        ]
        for index, claim in enumerate(claims):
            details.append(
                {
                    "kind": "claim",
                    "statement_class": claim.statement_class,
                    "section": claim.section,
                    "text": claim.text,
                    "start": claim.start,
                    "end": claim.end,
                    "count": claim.count,
                    "outcome": claim_outcomes.get(index, "recorded"),
                }
            )
        details.extend(
            {
                "kind": "discrepancy",
                "discrepancy_kind": discrepancy.kind,
                "detail": discrepancy.detail,
                "claimed": discrepancy.claimed,
                "ground_truth": discrepancy.ground_truth,
                "evidence": list(discrepancy.evidence),
            }
            for discrepancy in discrepancies
        )
        return DisclosureResult(
            status="RUN_COMPLETE",
            not_run_reason=None,
            verdict=verdict,
            counts=counts,
            rates=rates,
            coverage=coverage,
            details=tuple(details),
        )
