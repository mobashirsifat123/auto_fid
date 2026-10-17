"""Purpose: Shared Findings Memory — the cross-run feedback loop.

Adapted (mechanism only, no code copied) from DeepScientist's cumulative
Findings Memory (ICLR 2026, repo v1.6.0, Apache-2.0): every informative
outcome — successful, failed, negative, or inconclusive — is persisted as a
structured finding with its exact experimental lineage, and later stages
(novelty, experiment planning, review follow-up) retrieve relevant findings
*including failures* so the system does not repeatedly rediscover
unsuccessful approaches.

Honesty rules:
- observation and interpretation are stored in separate fields, never merged;
- ``CONFIRMED_FINDING`` means *gate-confirmed* (deterministic decision gates
  over measured seeds, optionally with recomputed metrics); it does NOT mean
  independently replicated;
- the next-action policy is a simple, inspectable, deterministic ranking
  labeled a heuristic — it is not Bayesian optimization and does not pretend
  to be.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from autoresearch.application.keywords import keyword_tokens
from autoresearch.domain.models import (
    ArtifactProvenance,
    ExperimentSpec,
    FindingStatus,
    ResearchFinding,
    StatisticalAnalysis,
)
from autoresearch.storage.repository import ResearchRepository

FINDINGS_PROMPT_VERSION = "findings-memory-v1"

_DECISION_STATUS = {
    "SUPPORTED": FindingStatus.CONFIRMED_FINDING,
    "REFUTED": FindingStatus.REFUTED_HYPOTHESIS,
    "PARTIAL_SUPPORT": FindingStatus.INCONCLUSIVE,
    "INSUFFICIENT_EVIDENCE": FindingStatus.INCONCLUSIVE,
}


def _keywords(*texts: str, limit: int = 12) -> list[str]:
    return keyword_tokens(*texts, limit=limit)


def _open_questions(block: dict[str, Any]) -> list[str]:
    """Falsification conditions normalized to a short question list."""
    raw = block.get("falsification_conditions") or []
    if isinstance(raw, str):
        return [q for q in raw.split(".") if q.strip()][:3]
    return [str(item) for item in raw][:3]


class FindingsMemory:
    def __init__(self, repository: ResearchRepository) -> None:
        self.repository = repository

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_from_analysis(
        self,
        spec: ExperimentSpec,
        analysis: StatisticalAnalysis,
        report_row: dict[str, Any],
        *,
        methodology: dict[str, Any] | None = None,
        fidelity: str = "CONFIRMATORY",
    ) -> ResearchFinding:
        """Persist one finding from a completed, analyzed experiment.

        The status derives ONLY from the deterministic decision; the
        interpretation field carries the report's narrative verbatim, kept
        separate from the observed outcome.
        """
        decision = str(report_row.get("decision") or analysis.hypothesis_support_assessment or "INSUFFICIENT_EVIDENCE")
        status = _DECISION_STATUS.get(str(analysis.hypothesis_support_assessment).upper(), FindingStatus.INCONCLUSIVE)
        block = (methodology or {}).get("methodology", methodology or {})
        effects = {name: round(float(value), 6) for name, value in dict(analysis.effect_size).items()}
        gates = {name: bool(passed) for name, passed in dict(analysis.decision_gates).items()}
        failed_gates = [name for name, passed in gates.items() if not passed]
        metrics = ", ".join(sorted(dict(analysis.proposed_metrics).keys()))

        if status is FindingStatus.CONFIRMED_FINDING:
            lesson = (
                f"Method '{spec.proposed_change}' passed its preregistered gates on {metrics} "
                f"(dataset {spec.dataset_identity}, split {spec.split_definition}); "
                f"mean differences {effects}. Applicability bounded by: {report_row.get('uncertainty') or 'unrecorded'}."
            )
            next_action = "Extend within the same mechanism: run the planned ablations/robustness tests before claiming generality."
        elif status is FindingStatus.REFUTED_HYPOTHESIS:
            lesson = (
                f"Method '{spec.proposed_change}' FAILED its preregistered gates on {metrics} "
                f"(dataset {spec.dataset_identity}, split {spec.split_definition}); failed gates: "
                f"{failed_gates or 'all'}. Negative result — do not repropose this mechanism under a new name "
                f"without a change addressing: {spec.falsification_condition}."
            )
            next_action = "Pivot: modify the causal mechanism or close this question; a renamed rerun is not a new method."
        elif status is FindingStatus.INCONCLUSIVE:
            lesson = (
                f"Method '{spec.proposed_change}' produced an inconclusive outcome on {metrics} "
                f"(dataset {spec.dataset_identity}): assessment {analysis.hypothesis_support_assessment}. "
                f"Uncertainty: {report_row.get('uncertainty') or 'unrecorded'}."
            )
            next_action = "Redesign for discriminability: stronger effect, lower variance, or a more discriminating metric/split."
        else:
            lesson = f"Outcome {decision} recorded for spec {spec.id}."
            next_action = "Review the analysis before further method work."

        finding = ResearchFinding(
            id=uuid4().hex,
            project_id=spec.project_id,
            run_id=spec.run_id,
            status=status,
            question_id=str(spec.gap_id),
            candidate_id=spec.candidate_id,
            methodology_freeze_id=str(spec.methodology_freeze_id),
            spec_id=spec.id,
            analysis_id=analysis.id,
            report_id=str(report_row.get("id") or ""),
            hypothesis=spec.hypothesis,
            expected_observation=str(block.get("predicted_observations") or ""),
            implementation_digest=str(block.get("algorithm_outline") or "")[:400],
            observed_outcome_json=json.dumps(
                {"decision": decision, "effects": effects, "gates": gates, "seed_count": len(analysis.seed_level_results)},
                sort_keys=True,
            ),
            interpretation=str(report_row.get("decision_evidence") or ""),
            uncertainty=str(report_row.get("uncertainty") or ""),
            fidelity=fidelity,
            failure_classification="none" if status is not FindingStatus.REFUTED_HYPOTHESIS else "gate_failure",
            lesson=lesson,
            applicability=f"dataset={spec.dataset_identity}; split={spec.split_definition}; seeds={list(spec.random_seeds)}",
            open_questions_json=json.dumps(_open_questions(block)),
            next_action=next_action,
            keywords_json=json.dumps(_keywords(spec.proposed_change, spec.hypothesis, metrics, spec.dataset_identity)),
            provenance=ArtifactProvenance.create(
                project_id=spec.project_id, run_id=spec.run_id,
                source_evidence=[analysis.id, str(report_row.get("id") or ""), spec.id],
                provider="deterministic", model="findings-memory",
                prompt_version=FINDINGS_PROMPT_VERSION,
                content={"status": status.value, "decision": decision},
            ),
        )
        self.repository.save_finding(finding)
        return finding

    def record_failure(
        self,
        spec: ExperimentSpec,
        reason: str,
        *,
        fidelity: str = "SMOKE",
    ) -> ResearchFinding:
        """Persist an implementation/measurement failure as a first-class finding."""
        finding = ResearchFinding(
            id=uuid4().hex,
            project_id=spec.project_id,
            run_id=spec.run_id,
            status=FindingStatus.IMPLEMENTATION_FAILURE,
            question_id=str(spec.gap_id),
            candidate_id=spec.candidate_id,
            methodology_freeze_id=str(spec.methodology_freeze_id),
            spec_id=spec.id,
            analysis_id="",
            report_id="",
            hypothesis=spec.hypothesis,
            expected_observation="",
            implementation_digest=str(spec.expected_outputs and spec.expected_outputs[0] or ""),
            observed_outcome_json=json.dumps({"reason": reason[:500]}),
            interpretation="",
            uncertainty="",
            fidelity=fidelity,
            failure_classification="measurement_ingestion_failure",
            lesson=(
                f"Execution of spec {spec.id} completed but its measurements did not pass ingestion "
                f"validation: {reason[:300]}. Repair the implementation's measurement contract before "
                f"re-proposing this method."
            ),
            applicability=f"dataset={spec.dataset_identity}",
            open_questions_json="[]",
            next_action="Repair the implementation (contract emission or identity binding), then re-execute; do not treat as a scientific result.",
            keywords_json=json.dumps(_keywords(spec.proposed_change, spec.hypothesis)),
            provenance=ArtifactProvenance.create(
                project_id=spec.project_id, run_id=spec.run_id,
                source_evidence=[spec.id],
                provider="deterministic", model="findings-memory",
                prompt_version=FINDINGS_PROMPT_VERSION,
                content={"status": "implementation_failure", "reason": reason[:200]},
            ),
        )
        self.repository.save_finding(finding)
        return finding

    # ------------------------------------------------------------------
    # Retrieval
    # ------------------------------------------------------------------

    def relevant(self, project_id: str, query: str, *, question_id: str | None = None, limit: int = 6) -> list[ResearchFinding]:
        """Question-keyed retrieval first (exact lineage — findings recorded for
        THIS research question are always relevant), then lexical supplement.
        Failures and refutations rank first so the generator sees what NOT to
        redo. Simple and inspectable by design."""
        primary = self.repository.findings_for_question(project_id, question_id, limit=limit) if question_id else []
        seen = {finding.id for finding in primary}
        lexical = [
            finding for finding in self.repository.search_findings(project_id, _keywords(query, limit=6), limit=limit)
            if finding.id not in seen
        ]
        return (primary + lexical)[:limit]

    def context_block(self, project_id: str, query: str, *, question_id: str | None = None, limit: int = 6) -> list[dict[str, Any]]:
        """Compact, prompt-ready entries for the novelty generator."""
        return [
            {
                "finding_id": finding.id,
                "status": finding.status.value,
                "hypothesis": finding.hypothesis[:200],
                "observed_outcome": json.loads(finding.observed_outcome_json or "{}"),
                "lesson": finding.lesson[:400],
                "applicability": finding.applicability[:200],
                "next_action": finding.next_action[:200],
            }
            for finding in self.relevant(project_id, query, question_id=question_id, limit=limit)
        ]

    # ------------------------------------------------------------------
    # Next-action policy (deterministic, inspectable — a HEURISTIC, not BO)
    # ------------------------------------------------------------------

    def next_actions(self, project_id: str, limit: int = 5) -> list[dict[str, Any]]:
        """Rank open follow-ups from the Findings Memory.

        Priority weights are fixed and transparent: refuted (close/pivot) >
        implementation failure (repair) > inconclusive (redesign) > confirmed
        (extend). This is a heuristic ordering inspired by exploit/explore
        balancing (DeepScientist); it is NOT Bayesian optimization and carries
        no calibrated uncertainty.
        """
        weight = {
            FindingStatus.REFUTED_HYPOTHESIS.value: 0,
            FindingStatus.IMPLEMENTATION_FAILURE.value: 1,
            FindingStatus.INCONCLUSIVE.value: 2,
            FindingStatus.CONFIRMED_FINDING.value: 3,
            FindingStatus.MEASURED_OBSERVATION.value: 4,
            FindingStatus.INTERPRETATION.value: 5,
            FindingStatus.IMPLEMENTED_METHOD.value: 6,
            FindingStatus.UNVERIFIED_IDEA.value: 7,
        }
        findings = self.repository.findings_for_project(project_id, limit=200)
        ranked = sorted(findings, key=lambda f: (weight.get(f.status.value, 9), f.created_at), reverse=False)
        return [
            {
                "question_id": finding.question_id,
                "candidate_id": finding.candidate_id,
                "status": finding.status.value,
                "priority": "heuristic",
                "next_action": finding.next_action,
            }
            for finding in ranked[:limit]
        ]
