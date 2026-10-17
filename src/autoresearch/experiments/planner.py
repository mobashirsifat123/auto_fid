"""Purpose: Translates a candidate method into a strict ExperimentSpec. Planning never runs code."""

from __future__ import annotations

import math
from pathlib import Path
from statistics import NormalDist
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
        "required_artifacts": {"type": "array", "items": {"type": "string"}, "description": "Artifact types this protocol actually produces (subset of the standard contract); omit for the full standard set"},
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

PROMPT_VERSION = "experiment-planner-v5"

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
    "- required_artifacts: array of strings (artifact types this protocol actually produces; omit for the full standard set)\n"
    "- network_policy: 'disabled' or 'allowlisted'\n"
    "- target_worker: string\n"
    "- expected_effect_size: number (expected Cohen's d)\n"
    "- power_target: number (0.5-0.99)\n"
    "- interpretation_rules: array of strings (pre-registered rules for positive, negative, null, failed, and unavailable outcomes)\n"
    "Every required key must be present with concrete, non-empty values. No other keys."
)


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
        grounding_block = format_evidence_context(evidence_context)
        selected = await self._generate_single(
            candidate, "efficient: the smallest controlled experiment that can decisively falsify the hypothesis",
            grounding_block,
        )
        adversarial = await self._adversarial_analysis(candidate, selected)
        selected["_adversarial"] = adversarial
        selected["_required_seeds"] = self._power_analysis(selected)
        # The adversarial review must affect the final plan, not just hidden metadata.
        if adversarial.get("what_would_disprove"):
            selected["falsification_condition"] = (
                f"{selected.get('falsification_condition', '')} "
                f"Adversarial check: {adversarial['what_would_disprove']}"
            ).strip()
        selected["controlled_variables"] = list(dict.fromkeys(
            [*selected.get("controlled_variables", []), *adversarial.get("confounding_factors", [])]
        ))
        selected["interpretation_rules"] = [
            *selected.get("interpretation_rules", []),
            *(f"Investigate failure mode before interpreting the effect: {item}"
              for item in adversarial.get("failure_modes", [])),
        ]

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
        if selected.get("expected_effect_size"):
            power_note = "Prospective power calculation uses the supplied standardized effect size assumption; verify it using pilot variance."
        else:
            power_note = "Power is not established without a standardized effect size or pilot variance; the seed count is a proposed minimum."

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
            statistical_plan=f"{selected.get('statistical_plan', '')} {power_note}".strip(),
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
            # Artifact evidence follows the method: the planner may declare the
            # artifact types this protocol actually produces; the analyzer and
            # implementation generator honor the subset. Empty keeps the full
            # universal contract (the analyzer's default).
            required_artifacts=tuple(str(item) for item in selected.get("required_artifacts", []) or ()),
        )

    async def _generate_single(self, candidate: NoveltyCandidate, focus: str, grounding_block: str = "") -> dict:
        """Generate one experiment design with a specific focus."""
        literature_section = f"\n\n{grounding_block}\n" if grounding_block else ""
        messages = [
            {"role": "system", "content": f"You are a meticulous experimental designer. Focus: {focus}. Produce a concise, reproducible specification with one primary metric, a matched baseline, essential ablations, explicit stopping conditions, and missing prerequisites. Treat effect sizes as prospective assumptions, never measurements; use 0 when no standardized effect is justified. Do not invent dataset checksums, commits, available scripts, or observed metrics." + (" " + _BASELINE_CONTRACT if grounding_block else "")},
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
        """Raw metric deltas are not standardized effect sizes (delta / sigma)."""
        effect_size = design.get("expected_effect_size")
        proposed = max(3, int(design.get("required_seed_count") or len(design.get("random_seeds", []))))
        if not isinstance(effect_size, (int, float)) or not math.isfinite(effect_size) or effect_size <= 0:
            return proposed
        power = float(design.get("power_target", 0.8))
        if not 0.5 <= power < 1:
            raise ValueError("power_target must be between 0.5 and 1 (exclusive).")
        n_per_group = math.ceil(2 * (NormalDist().inv_cdf(0.975) + NormalDist().inv_cdf(power)) ** 2 / effect_size ** 2)
        return max(proposed, n_per_group)
