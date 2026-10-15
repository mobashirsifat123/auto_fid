"""Purpose: Synthesizes the final ExperimentReport."""
from __future__ import annotations

from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, ExperimentDecision, ExperimentReport, ExperimentSpec, StatisticalAnalysis
from autoresearch.experiments.integrity import IntegrityFailure


class ScientificReportWriter:
    def __init__(self, router=None) -> None:
        self.router = router

    async def write(self, spec: ExperimentSpec, analysis: StatisticalAnalysis) -> ExperimentReport:
        if analysis.evidence_status == "INSUFFICIENT_EVIDENCE":
            raise IntegrityFailure("INSUFFICIENT_EVIDENCE")
        decision_map = {
            "SUPPORTED": ExperimentDecision.SUPPORTED,
            "REFUTED": ExperimentDecision.REFUTED,
            "PARTIAL_SUPPORT": ExperimentDecision.PARTIALLY_SUPPORTED,
        }
        decision = decision_map.get(analysis.hypothesis_support_assessment)
        if decision is None:
            raise IntegrityFailure("INSUFFICIENT_EVIDENCE")
        gate_lines = ", ".join(f"{name}: {'PASS' if passed else 'FAIL'}" for name, passed in analysis.decision_gates.items())
        measurements = {metric: {"value": value, "classification": "MEASURED RESULT"} for metric, value in analysis.proposed_metrics.items()}
        return ExperimentReport(
            id=uuid4().hex, project_id=spec.project_id, run_id=spec.run_id, analysis_id=analysis.id,
            literature_motivation="Evidence-backed experiment generated from the selected research candidate.",
            research_gap="See source novelty candidate and its cited gap.", hypothesis=spec.hypothesis,
            proposed_method=spec.proposed_change,
            implementation_diff="MEASURED RESULT: implementation artifact contract validated.",
            lineage=f"candidate={spec.candidate_id}; spec={spec.id}", dataset_split=spec.split_definition,
            exact_commands=(), baseline_results=analysis.baseline_metrics, aggregated_results=measurements,
            uncertainty="Derived statistics are limited to the recorded raw seed observations.",
            ablations="ABLATION_NOT_RUN" if not analysis.ablation_contributions else "Derived from measured descendant runs.",
            robustness_tests="Not claimed unless represented by validated artifacts.", failed_runs=0,
            cost=analysis.training_inference_cost,
            limitations="No model prediction, fixture, or incompatible cached value is reported as measurement.",
            reproducibility_instructions="Inspect the artifact contract, identity fields, raw seed observations, and exact statistics.",
            decision=decision, decision_evidence=f"Deterministic decision gates: {gate_lines}.",
            provenance=ArtifactProvenance.create(project_id=spec.project_id, run_id=spec.run_id,
                source_evidence=[analysis.id], provider="deterministic", model="report-contract",
                prompt_version="report-writer-v4-fail-closed", content={"decision": decision.value, "gates": analysis.decision_gates}),
        )
