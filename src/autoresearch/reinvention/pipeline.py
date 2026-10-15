"""Scientific Reinvention Pipeline conditioning on experiment reports to advance research iterations."""

from __future__ import annotations

import json
import logging
from uuid import uuid4

from autoresearch.domain.models import (
    ArtifactProvenance,
    CandidateStatus,
    ExperimentDecision,
    NoveltyCandidate,
    ReinventionAction,
    ReinventionAnalysis,
)
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.router import ModelRouter
from autoresearch.storage.repository import ResearchRepository

logger = logging.getLogger(__name__)

REINVENTION_PROMPT_VERSION = "reinvention-analysis-v1"

REINVENTION_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["refine", "pivot", "rebuild", "abort"],
            "description": "Strategic next action based on empirical results: refine (extrapolate gains), pivot (change core mechanism), rebuild (fix execution/optimization defects), abort (unviable direction)."
        },
        "root_cause_diagnosis": {
            "type": "string",
            "description": "Exhaustive scientific diagnosis of why the experiment succeeded or failed."
        },
        "empirical_lessons": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Concrete empirical findings extracted from metric observations and ablations."
        },
        "next_generation_hypothesis": {
            "type": "string",
            "description": "Refined or pivoted hypothesis for the next research iteration."
        },
        "revised_mechanism": {
            "type": "string",
            "description": "Modified causal mechanism addressing documented failure modes or limitations."
        },
        "actionable_next_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Specific experimental protocol adjustments for Cycle N+1."
        },
        "revised_method_name": {
            "type": "string",
            "description": "Updated name/title for the next-generation candidate method."
        },
        "revised_difference_statement": {
            "type": "string",
            "description": "How the revised method differs from both prior art and the parent candidate."
        },
        "expected_effect": {
            "type": "string",
            "description": "Targeted empirical effect size for the next iteration."
        },
    },
    "required": [
        "action", "root_cause_diagnosis", "empirical_lessons",
        "next_generation_hypothesis", "revised_mechanism", "actionable_next_steps",
        "revised_method_name", "revised_difference_statement", "expected_effect"
    ],
    "additionalProperties": False,
}


class ReinventionPipeline:
    def __init__(self, repository: ResearchRepository, router: ModelRouter) -> None:
        self.repository = repository
        self.router = router

    async def analyze_and_reinvent(self, candidate_id: str, *, run_id: str) -> tuple[ReinventionAnalysis, NoveltyCandidate | None]:
        """Perform result-conditioned reinvention for a completed experiment candidate."""
        candidate = self.repository.get_novelty_candidate(candidate_id)
        if not candidate:
            raise ValueError(f"Novelty candidate '{candidate_id}' not found.")

        # Locate experiment spec, analysis, and report for this candidate
        spec = self.repository.get_experiment_spec(candidate.id)
        if not spec:
            raise ValueError(f"No experiment specification found for candidate '{candidate_id}'. Run experiment pipeline first.")

        analysis = self.repository.get_statistical_analysis(spec.id)
        if not analysis:
            raise ValueError(f"No statistical analysis found for experiment spec '{spec.id}'. Execute experiment first.")

        report = self.repository.get_experiment_report(analysis.id)
        if not report:
            raise ValueError(f"No experiment report found for analysis '{analysis.id}'. Complete report generation first.")

        logger.info("Starting reinvention analysis for candidate %s (decision: %s)", candidate.id, report.decision)

        # Step 1: Query LLM for structured root-cause analysis and strategic action
        payload = await self._run_reinvention_reasoning(candidate, spec, analysis, report, run_id=run_id)

        action_str = payload.get("action", "pivot")
        action = ReinventionAction(action_str)

        provider, model = self.router.identity(ModelRole.RESEARCH_REASONING)
        provenance = ArtifactProvenance.create(
            project_id=candidate.project_id,
            run_id=run_id,
            source_evidence=[candidate.id, spec.id, analysis.id, report.id],
            provider=provider,
            model=model,
            prompt_version=REINVENTION_PROMPT_VERSION,
            content=payload,
        )

        next_candidate: NoveltyCandidate | None = None
        next_candidate_id: str | None = None

        # Step 2: If action is not ABORT, synthesize the next-generation NoveltyCandidate for Cycle N+1
        if action != ReinventionAction.ABORT:
            next_candidate = self._synthesize_next_candidate(
                candidate=candidate,
                payload=payload,
                run_id=run_id,
                provider=provider,
                model=model,
            )
            next_candidate_id = next_candidate.id
            self.repository.save_novelty_candidate(next_candidate)
            logger.info("Synthesized Cycle N+1 candidate %s (action: %s)", next_candidate.id, action)

        # Step 3: Save ReinventionAnalysis
        reinvention = ReinventionAnalysis(
            id=uuid4().hex,
            project_id=candidate.project_id,
            run_id=run_id,
            spec_id=spec.id,
            report_id=report.id,
            parent_candidate_id=candidate.id,
            action=action,
            root_cause_diagnosis=payload.get("root_cause_diagnosis", ""),
            empirical_lessons=tuple(payload.get("empirical_lessons", [])),
            next_generation_hypothesis=payload.get("next_generation_hypothesis", ""),
            revised_mechanism=payload.get("revised_mechanism", ""),
            actionable_next_steps=tuple(payload.get("actionable_next_steps", [])),
            provenance=provenance,
            next_candidate_id=next_candidate_id,
        )

        self.repository.save_reinvention_analysis(reinvention)
        logger.info("Reinvention analysis %s complete for project %s", reinvention.id, candidate.project_id)

        return reinvention, next_candidate

    async def _run_reinvention_reasoning(
        self,
        candidate: NoveltyCandidate,
        spec,
        analysis,
        report,
        *,
        run_id: str,
    ) -> dict:
        """Query LLM for scientific diagnosis and strategic direction."""
        provenance_data = getattr(analysis.provenance, "content", {}) if isinstance(getattr(analysis.provenance, "content", None), dict) else {}
        bayesian_probs = provenance_data.get("bayesian_p_positive", {})
        cohens_d_vals = provenance_data.get("cohens_d", {})
        holm_sig = provenance_data.get("holm_bonferroni_significant", {})

        metrics_summary = json.dumps({
            "effect_size": analysis.effect_size,
            "cohens_d": cohens_d_vals,
            "bayesian_p_positive": bayesian_probs,
            "holm_bonferroni_significant": holm_sig,
        }, indent=2)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an elite principal AI scientist conducting result-conditioned research iteration. "
                    "Analyze the empirical evidence from the previous experiment cycle. "
                    "Determine whether to REFINE (extrapolate gains), PIVOT (change causal mechanism to resolve failure), "
                    "REBUILD (fix execution/optimization defects), or ABORT (unviable direction). "
                    "Return a precise, highly-structured reinvention plan."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Prompt version: {REINVENTION_PROMPT_VERSION}\n"
                    f"Parent Method: {candidate.method}\n"
                    f"Parent Causal Mechanism: {candidate.causal_mechanism}\n"
                    f"Hypothesis: {spec.hypothesis}\n"
                    f"Falsification Condition: {spec.falsification_condition}\n\n"
                    f"Empirical Decision: {report.decision}\n"
                    f"Decision Evidence: {report.decision_evidence}\n"
                    f"Statistical Analysis:\n{metrics_summary}\n\n"
                    f"Limitations: {report.limitations}\n"
                    f"Uncertainty: {report.uncertainty}\n\n"
                    "Provide your structured reinvention analysis."
                ),
            },
        ]

        payload = await self.router.structured_output(
            ModelRole.RESEARCH_REASONING,
            messages=messages,
            schema={"name": "reinvention_analysis", "schema": REINVENTION_SCHEMA},
            project_id=candidate.project_id,
            run_id=run_id,
            operation="reinvention_analysis",
        )
        return payload

    def _synthesize_next_candidate(
        self,
        *,
        candidate: NoveltyCandidate,
        payload: dict,
        run_id: str,
        provider: str,
        model: str,
    ) -> NoveltyCandidate:
        """Build a new NoveltyCandidate pre-populated with reinvention lessons for Cycle N+1."""
        revised_method = payload.get("revised_method_name") or f"{candidate.method} (v2-{payload.get('action', 'refined')})"
        revised_mechanism = payload.get("revised_mechanism") or candidate.causal_mechanism
        revised_diff = payload.get("revised_difference_statement") or candidate.difference_from_existing
        expected_eff = payload.get("expected_effect") or candidate.expected_effect

        provenance = ArtifactProvenance.create(
            project_id=candidate.project_id,
            run_id=run_id,
            source_evidence=[candidate.id],
            provider=provider,
            model=model,
            prompt_version=f"{REINVENTION_PROMPT_VERSION}-next-gen",
            content=payload,
        )

        return NoveltyCandidate(
            id=uuid4().hex,
            project_id=candidate.project_id,
            run_id=run_id,
            gap_id=candidate.gap_id,
            strategy=f"reinvention-{payload.get('action', 'refine')}",
            method=revised_method,
            causal_mechanism=revised_mechanism,
            target_limitation=candidate.target_limitation,
            difference_from_existing=f"Cycle N+1 iteration over '{candidate.method}': {revised_diff}",
            expected_effect=expected_eff,
            falsification_condition=f"Falsified if {revised_method} fails to outperform baseline on primary metric.",
            minimum_validating_experiment=f"Ablated comparison of {revised_method} vs parent method {candidate.method}.",
            required_ablations=(f"Parent method baseline ({candidate.method})", "No mechanism modification"),
            implementation_complexity=candidate.implementation_complexity,
            expected_compute=candidate.expected_compute,
            risks_and_objections=tuple(payload.get("actionable_next_steps", [])),
            status=CandidateStatus.GENERATED,
            provenance=provenance,
        )
