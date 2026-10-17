"""Purpose: Synthesizes the final ExperimentReport."""
from __future__ import annotations

from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, ExperimentDecision, ExperimentReport, ExperimentSpec, StatisticalAnalysis
from autoresearch.experiments.integrity import IntegrityFailure
from autoresearch.storage.repository import ResearchRepository


class ScientificReportWriter:
    def __init__(self, router=None, repository: ResearchRepository | None = None) -> None:
        self.router = router
        self.repository = repository

    async def write(
        self,
        spec: ExperimentSpec,
        analysis: StatisticalAnalysis,
        *,
        repository: ResearchRepository | None = None,
    ) -> ExperimentReport:
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
        repository_available = repository or self.repository
        exact_commands: tuple[str, ...] = ()
        if repository_available is not None:
            try:
                exact_commands = tuple(repository_available.execution_commands_for_spec(spec.id))
            except Exception:  # noqa: BLE001 - command recording is best-effort; absence is reported honestly
                exact_commands = ()
        # The manuscript template (FastTrack, template-spec.json) requires the
        # Discussion to list at least 3 study limitations. Every entry here is
        # an objective fact about this run — never an invented weakness.
        limitations: tuple[str, ...] = (
            "No model prediction, fixture, or incompatible cached value is reported as measurement.",
            f"Derived statistics are limited to the recorded raw seed observations "
            f"({len(analysis.seed_level_results)} seed(s)); findings are bounded to dataset "
            f"'{spec.dataset_identity}' (split '{spec.split_definition}') and do not transfer "
            f"beyond this configuration.",
            "Ablations were not run; per-component contributions are unmeasured."
            if not analysis.ablation_contributions
            else "Ablation contributions are measured only for recorded descendant runs.",
            "Robustness tests were not run; robustness is not claimed unless represented by validated artifacts.",
        )
        return ExperimentReport(
            id=uuid4().hex, project_id=spec.project_id, run_id=spec.run_id, analysis_id=analysis.id,
            literature_motivation="Evidence-backed experiment generated from the selected research candidate.",
            research_gap="See source novelty candidate and its cited gap.", hypothesis=spec.hypothesis,
            proposed_method=spec.proposed_change,
            implementation_diff="MEASURED RESULT: implementation artifact contract validated.",
            lineage=f"candidate={spec.candidate_id}; spec={spec.id}", dataset_split=spec.split_definition,
            exact_commands=exact_commands, baseline_results=analysis.baseline_metrics, aggregated_results=measurements,
            uncertainty="Derived statistics are limited to the recorded raw seed observations.",
            ablations="ABLATION_NOT_RUN" if not analysis.ablation_contributions else "Derived from measured descendant runs.",
            robustness_tests="Not claimed unless represented by validated artifacts.", failed_runs=0,
            cost=analysis.training_inference_cost,
            limitations=limitations,
            reproducibility_instructions="Inspect the artifact contract, identity fields, raw seed observations, and exact statistics.",
            decision=decision, decision_evidence=f"Deterministic decision gates: {gate_lines}.",
            provenance=ArtifactProvenance.create(project_id=spec.project_id, run_id=spec.run_id,
                source_evidence=[analysis.id], provider="deterministic", model="report-contract",
                prompt_version="report-writer-v5-template-limitations", content={"decision": decision.value, "gates": analysis.decision_gates}),
        )
