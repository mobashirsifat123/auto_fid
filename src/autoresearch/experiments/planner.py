"""Purpose: Translates a candidate method into a strict ExperimentSpec. Planning never runs code."""

from __future__ import annotations

import math
from pathlib import Path
from uuid import uuid4

from autoresearch.domain.models import (
    ArtifactProvenance,
    ExperimentSpec,
    NoveltyCandidate,
)
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.router import ModelRouter
from autoresearch.review.evidence_context import format_evidence_context

SPEC_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "causal_mechanism": {"type": "string"},
        "falsification_condition": {"type": "string"},
        "repository": {"type": "string"},
        "baseline_commit": {"type": "string"},
        "dataset_identity": {"type": "string"},
        "dataset_checksum": {"type": "string"},
        "split_definition": {"type": "string"},
        "baseline_configuration": {"type": "string"},
        "proposed_change": {"type": "string"},
        "independent_variables": {"type": "array", "items": {"type": "string"}},
        "controlled_variables": {"type": "array", "items": {"type": "string"}},
        "metrics": {"type": "array", "items": {"type": "string"}},
        "random_seeds": {"type": "array", "items": {"type": "integer"}},
        "ablations": {"type": "array", "items": {"type": "string"}},
        "robustness_tests": {"type": "array", "items": {"type": "string"}},
        "statistical_plan": {"type": "string"},
        "compute_budget": {"type": "string"},
        "stopping_conditions": {"type": "array", "items": {"type": "string"}},
        "expected_outputs": {"type": "array", "items": {"type": "string"}},
        "artifact_contract": {"type": "string"},
        "metric_direction": {"type": "object", "additionalProperties": {"type": "string", "enum": ["maximize", "minimize"]}},
        "minimum_meaningful_effect": {"type": "object", "additionalProperties": {"type": "number"}},
        "required_seed_count": {"type": "integer", "minimum": 1},
        "network_policy": {"type": "string", "enum": ["disabled", "allowlisted"]},
        "target_worker": {"type": "string"},
        "expected_effect_size": {"type": "number", "description": "Expected Cohen's d effect size for power analysis"},
        "power_target": {"type": "number", "minimum": 0.5, "maximum": 0.99, "description": "Target statistical power (e.g., 0.8)"},
        "interpretation_rules": {
            "type": "array", "items": {"type": "string"}, "minItems": 1,
            "description": "Pre-registered rules for interpreting positive, negative, null, failed, and unavailable outcomes",
        },
    },
    "required": [
        "hypothesis", "causal_mechanism", "falsification_condition",
        "repository", "baseline_commit", "dataset_identity", "dataset_checksum",
        "split_definition", "baseline_configuration", "proposed_change",
        "independent_variables", "controlled_variables", "metrics",
        "random_seeds", "ablations", "robustness_tests", "statistical_plan",
        "compute_budget", "stopping_conditions", "expected_outputs",
        "artifact_contract", "interpretation_rules"
    ],
    "additionalProperties": False,
}

ADVERSARIAL_SCHEMA = {
    "type": "object",
    "properties": {
        "what_would_disprove": {"type": "string"},
        "confounding_factors": {"type": "array", "items": {"type": "string"}},
        "minimal_viable_experiment": {"type": "string"},
        "expected_null_outcome": {"type": "string"},
        "failure_modes": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["what_would_disprove", "confounding_factors", "minimal_viable_experiment", "expected_null_outcome", "failure_modes"],
    "additionalProperties": False,
}

SPEC_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_index": {"type": "integer", "minimum": 0, "maximum": 2},
        "selection_rationale": {"type": "string"},
    },
    "required": ["selected_index", "selection_rationale"],
    "additionalProperties": False,
}

PROMPT_VERSION = "experiment-planner-v4"

# Baseline anchoring adapted from HKUDS/AI-Researcher @ f9a6f8480860c193afff600eeffe3defcee8a978
# (research_agent/inno/agents/inno_agent/idea_agent.py): designs must be tied to
# the existing methods surveyed from the papers. Mechanism reused; no upstream
# code copied (the upstream repository carries no license file).
_BASELINE_CONTRACT = (
    "Ground the design in the reviewed literature supplied below when present. "
    "baseline_configuration MUST name at least one established method from the "
    "supplied corpus (by [paper_id]) or an explicitly named standard baseline; "
    "do not invent baselines. Prefer metrics that make results comparable with "
    "the supplied claims. Do not invent papers, claims, or datasets."
)

# json_object mode drops the schema, so the prompt must enumerate the contract.
_FIELD_CONTRACT = (
    "Return ONLY a JSON object with exactly these keys:\n"
    "- hypothesis: string\n"
    "- causal_mechanism: string\n"
    "- falsification_condition: string\n"
    "- repository: string\n"
    "- baseline_commit: string\n"
    "- dataset_identity: string\n"
    "- dataset_checksum: string\n"
    "- split_definition: string\n"
    "- baseline_configuration: string\n"
    "- proposed_change: string\n"
    "- independent_variables: array of strings\n"
    "- controlled_variables: array of strings\n"
    "- metrics: array of strings\n"
    "- random_seeds: array of integers\n"
    "- ablations: array of strings\n"
    "- robustness_tests: array of strings\n"
    "- statistical_plan: string\n"
    "- compute_budget: string\n"
    "- stopping_conditions: array of strings\n"
    "- expected_outputs: array of strings containing the Python experiment script (e.g. 'scripts/run_experiment.py') and metric log files\n"
    "- artifact_contract: string\n"
    "- metric_direction: object mapping metric name to 'maximize' or 'minimize'\n"
    "- minimum_meaningful_effect: object mapping metric name to a number\n"
    "- required_seed_count: integer\n"
    "- network_policy: 'disabled' or 'allowlisted'\n"
    "- target_worker: string\n"
    "- expected_effect_size: number (expected Cohen's d)\n"
    "- power_target: number (0.5-0.99)\n"
    "- interpretation_rules: array of strings (pre-registered rules for positive, negative, null, failed, and unavailable outcomes)\n"
    "Every required key must be present with concrete, non-empty values. No other keys."
)


import asyncio


class ExperimentSpecGenerator:
    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    async def plan(
        self,
        candidate: NoveltyCandidate,
        *,
        evidence_context: dict | None = None,
        gap_id: str = "",
        methodology_freeze_id: str = "",
    ) -> ExperimentSpec:
        # Step 1 & 2: Generate 3 experiment designs & run adversarial analysis concurrently (E6)
        focuses = [
            "default: a standard rigorous experiment design",
            "efficient: a minimal viable experiment that can quickly falsify the hypothesis",
            "comprehensive: an exhaustive experiment with many ablations and robustness tests",
        ]
        grounding_block = format_evidence_context(evidence_context)
        designs = list(await asyncio.gather(*[self._generate_single(candidate, focus, grounding_block) for focus in focuses]))

        adversarials = await asyncio.gather(*[self._adversarial_analysis(candidate, design) for design in designs])
        for design, adv in zip(designs, adversarials):
            design["_adversarial"] = adv

        # Step 3: Multi-metric power analysis to determine required seed count (E2)
        for design in designs:
            design["_required_seeds"] = self._power_analysis(design)

        # Step 4: Select the best design
        selected = await self._select_best_design(candidate, designs)

        # Build the final spec
        provider, model = self.router.identity(ModelRole.RESEARCH_REASONING)
        provenance = ArtifactProvenance.create(
            project_id=candidate.project_id, run_id=candidate.run_id,
            source_evidence=[candidate.id],
            provider=provider, model=model,
            prompt_version=PROMPT_VERSION,
            content={**selected, "adversarial": selected.get("_adversarial", {}),
                     "required_seeds": selected.get("_required_seeds", 5)},
        )

        # Compute required_seed_count from power analysis or fall back
        required_seeds = selected.get("_required_seeds", 5)
        power_target = selected.get("power_target", 0.8)

        # The repository is an execution target: never trust the LLM to invent
        # it (it emits URLs and hallucinated paths). Default to the process CWD
        # when the claimed path does not exist on disk.
        repository = str(selected.get("repository") or "").strip()
        if not repository or not Path(repository).expanduser().is_dir():
            repository = str(Path.cwd().expanduser().resolve())

        return ExperimentSpec(
            id=uuid4().hex, project_id=candidate.project_id, run_id=candidate.run_id,
            candidate_id=candidate.id,
            hypothesis=selected.get("hypothesis", ""),
            causal_mechanism=selected.get("causal_mechanism", ""),
            falsification_condition=selected.get("falsification_condition", ""),
            repository=repository,
            baseline_commit=selected.get("baseline_commit", ""),
            dataset_identity=selected.get("dataset_identity", ""),
            dataset_checksum=selected.get("dataset_checksum", ""),
            split_definition=selected.get("split_definition", ""),
            baseline_configuration=selected.get("baseline_configuration", ""),
            proposed_change=selected.get("proposed_change", ""),
            independent_variables=tuple(selected.get("independent_variables", [])),
            controlled_variables=tuple(selected.get("controlled_variables", [])),
            metrics=tuple(selected.get("metrics", [])),
            random_seeds=tuple(selected.get("random_seeds", [])),
            ablations=tuple(selected.get("ablations", [])),
            robustness_tests=tuple(selected.get("robustness_tests", [])),
            statistical_plan=selected.get("statistical_plan", ""),
            compute_budget=selected.get("compute_budget", ""),
            stopping_conditions=tuple(selected.get("stopping_conditions", [])),
            expected_outputs=tuple(selected.get("expected_outputs", [])),
            artifact_contract=selected.get("artifact_contract", ""),
            provenance=provenance,
            metric_direction=selected.get("metric_direction", {}),
            minimum_meaningful_effect=selected.get("minimum_meaningful_effect", {}),
            required_seed_count=required_seeds,
            network_policy=selected.get("network_policy", "disabled"),
            target_worker=selected.get("target_worker", "local"),
            approval_required=True,
            gap_id=gap_id or candidate.gap_id,
            methodology_freeze_id=methodology_freeze_id,
            interpretation_rules=tuple(selected.get("interpretation_rules", []) or ("Interpret outcomes against the pre-specified falsification condition.",)),
        )

    async def _generate_single(self, candidate: NoveltyCandidate, focus: str, grounding_block: str = "") -> dict:
        """Generate one experiment design with a specific focus."""
        literature_section = f"\n\n{grounding_block}\n" if grounding_block else ""
        messages = [
            {"role": "system", "content": f"You are a meticulous experimental designer. Focus: {focus}. Translate the proposed novelty candidate into a highly-detailed, reproducible experiment specification. Specify random seeds, ablations, exact stopping conditions, and effect size estimates." + (" " + _BASELINE_CONTRACT if grounding_block else "")},
            {"role": "user", "content": f"Prompt version: {PROMPT_VERSION}\nCandidate Method: {candidate.method}\nExpected Effect: {candidate.expected_effect}\nAblations: {candidate.required_ablations}\nObjections: {candidate.risks_and_objections}{literature_section}\n\n{_FIELD_CONTRACT}"}
        ]

        payload = await self.router.structured_output(
            ModelRole.RESEARCH_REASONING, messages=messages,
            schema={"name": "experiment_spec", "schema": SPEC_SCHEMA},
            project_id=candidate.project_id, run_id=candidate.run_id,
            operation="experiment_planning",
        )
        return payload

    async def _adversarial_analysis(self, candidate: NoveltyCandidate, design: dict) -> dict:
        """Generate an adversarial analysis: what would disprove the hypothesis."""
        messages = [
            {"role": "system", "content": "You are an adversarial reviewer whose job is to find what would disprove this experiment's hypothesis. Be thorough and concrete."},
            {"role": "user", "content": f"Hypothesis: {design.get('hypothesis', '')}\nProposed Change: {design.get('proposed_change', '')}\nCandidate Method: {candidate.method}\nFalsification Condition: {candidate.falsification_condition}\n\nReturn ONLY a JSON object with exactly these keys:\n- what_would_disprove: string\n- confounding_factors: array of strings\n- minimal_viable_experiment: string\n- expected_null_outcome: string\n- failure_modes: array of strings\nEvery key must be present. No other keys."}
        ]

        payload = await self.router.structured_output(
            ModelRole.SCIENTIFIC_CRITIC, messages=messages,
            schema={"name": "adversarial_analysis", "schema": ADVERSARIAL_SCHEMA},
            project_id=candidate.project_id, run_id=candidate.run_id,
            operation="adversarial_experiment_analysis",
        )
        return payload

    def _power_analysis(self, design: dict) -> int:
        """Compute required seed count as max across all metrics (E2)."""
        minimum_effects = design.get("minimum_meaningful_effect", {})
        power_target = design.get("power_target", 0.8)
        z_scores = {0.5: 0.0, 0.6: 0.25, 0.7: 0.52, 0.8: 0.84, 0.9: 1.28, 0.95: 1.64, 0.99: 2.33}
        z_beta = z_scores.get(power_target, 0.84)
        z_alpha_half = 1.96

        if isinstance(minimum_effects, dict) and minimum_effects:
            required = []
            for metric, min_effect in minimum_effects.items():
                try:
                    eff = float(min_effect)
                    if eff > 0:
                        n = math.ceil(2 * (z_alpha_half + z_beta) ** 2 / (eff ** 2))
                        required.append(n)
                except (TypeError, ValueError):
                    pass
            if required:
                return max(3, min(max(required), 30))

        effect_size = design.get("expected_effect_size")
        if effect_size is None or not isinstance(effect_size, (int, float)) or effect_size <= 0:
            effect_size = 0.5
        n_per_group = math.ceil(2 * (z_alpha_half + z_beta) ** 2 / (effect_size ** 2))
        return max(3, min(n_per_group, 30))

    async def _select_best_design(self, candidate: NoveltyCandidate, designs: list[dict]) -> dict:
        """Select the best experiment design via meta-evaluation."""
        if len(designs) == 1:
            return designs[0]

        designs_text = "\n\n---\n\n".join(
            f"Design {i}:\nHypothesis: {d.get('hypothesis', '')[:200]}\n"
            f"Seeds required: {d.get('_required_seeds', 'N/A')}\n"
            f"Adversarial falsification: {d.get('_adversarial', {}).get('what_would_disprove', '')[:200]}"
            for i, d in enumerate(designs)
        )
        selection = await self.router.structured_output(
            ModelRole.RESEARCH_REASONING,
            messages=[
                {"role": "system", "content": "Select the best experiment design. Prefer designs that are maximally falsifiable, have appropriate seed counts from power analysis, and have clear adversarial falsification conditions."},
                {"role": "user", "content": f"Candidate: {candidate.method}\n\n{designs_text}\n\nReturn selected_index and selection_rationale."},
            ],
            schema={"name": "design_selection", "schema": SPEC_SELECTION_SCHEMA},
            project_id=candidate.project_id, run_id=candidate.run_id,
            operation="experiment_design_selection",
        )
        idx = max(0, min(selection.get("selected_index", 0), len(designs) - 1))
        return designs[idx]
