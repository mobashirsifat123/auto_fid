"""Purpose: Evaluates candidate methods using a multi-perspective reviewer committee."""

from __future__ import annotations

import asyncio
import json
from statistics import mean
from uuid import uuid4

from autoresearch.domain.models import (
    ArtifactProvenance,
    NoveltyCandidate,
    NoveltyCritique,
    PriorArtAssessment,
)
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.router import ModelRouter

CRITIQUE_SCHEMA = {
    "type": "object",
    "properties": {
        "novelty_score": {"type": "number", "minimum": 0, "maximum": 1},
        "significance_score": {"type": "number", "minimum": 0, "maximum": 1},
        "causal_plausibility_score": {"type": "number", "minimum": 0, "maximum": 1},
        "falsifiability_score": {"type": "number", "minimum": 0, "maximum": 1},
        "experimental_discriminability_score": {"type": "number", "minimum": 0, "maximum": 1},
        "feasibility_score": {"type": "number", "minimum": 0, "maximum": 1},
        "implementation_risk_score": {"type": "number", "minimum": 0, "maximum": 1},
        "expected_information_gain_score": {"type": "number", "minimum": 0, "maximum": 1},
        "compute_efficiency_score": {"type": "number", "minimum": 0, "maximum": 1},
        "publication_potential_score": {"type": "number", "minimum": 0, "maximum": 1},
        "objections": {"type": "array", "items": {"type": "string"}},
        "failure_scenarios": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "novelty_score", "significance_score", "causal_plausibility_score",
        "falsifiability_score", "experimental_discriminability_score",
        "feasibility_score", "implementation_risk_score",
        "expected_information_gain_score", "compute_efficiency_score",
        "publication_potential_score", "objections", "failure_scenarios"
    ],
    "additionalProperties": False,
}

PROMPT_VERSION = "novelty-critic-v2"

# Distinct reviewer perspectives for ensemble critique
_CRITIC_PERSPECTIVES = [
    {
        "system": "You are a harsh but fair area chair for a top-tier machine learning conference. You value rigorous experimental validation above all. Score strictly. Provide explicit objections and failure scenarios.",
        "tag": "area_chair",
    },
    {
        "system": "You are a skeptical senior reviewer who has seen many claimed novelties fail to reproduce. Focus on whether the claimed effect is real and whether the falsification condition is meaningful. Score conservatively.",
        "tag": "skeptic",
    },
    {
        "system": "You are an optimistic junior reviewer excited by new ideas. You value creativity and potential impact, but still demand sound reasoning. Be fair but generous where the idea has merit.",
        "tag": "enthusiast",
    },
]


from statistics import mean, stdev

_ROLE_WEIGHTS = {
    "area_chair":  {"novelty_score": 1.5, "significance_score": 1.5, "publication_potential_score": 1.5, "falsifiability_score": 1.2},
    "skeptic":     {"feasibility_score": 1.5, "implementation_risk_score": 1.5, "compute_efficiency_score": 1.3, "causal_plausibility_score": 1.3},
    "enthusiast":  {"expected_information_gain_score": 1.3},
}


class ScientificCritic:
    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    async def evaluate(self, candidate: NoveltyCandidate, *, prior_art: PriorArtAssessment | None = None) -> NoveltyCritique:
        """Evaluate using ensemble of 3 reviewer perspectives and aggregate."""
        critiques = await asyncio.gather(*(
            self._single_evaluate(candidate, perspective, prior_art=prior_art)
            for perspective in _CRITIC_PERSPECTIVES
        ))

        # Aggregate scores: weighted mean across ensemble (N5) & agreement (N6)
        score_keys = [
            "novelty_score", "significance_score", "causal_plausibility_score",
            "falsifiability_score", "experimental_discriminability_score",
            "feasibility_score", "implementation_risk_score",
            "expected_information_gain_score", "compute_efficiency_score",
            "publication_potential_score",
        ]
        aggregated = {}
        agreement = {}
        for key in score_keys:
            vals = [c[key] for c in critiques]
            # Role-weighted average
            w_sum = sum(c[key] * _ROLE_WEIGHTS.get(c["tag"], {}).get(key, 1.0) for c in critiques)
            w_total = sum(_ROLE_WEIGHTS.get(c["tag"], {}).get(key, 1.0) for c in critiques)
            aggregated[key] = round(w_sum / max(w_total, 1e-9), 6)

            # Inter-reviewer agreement score (N6): 1.0 = perfect agreement, 0.0 = high disagreement
            sd = stdev(vals) if len(vals) > 1 else 0.0
            agreement[key] = round(max(0.0, 1.0 - 2.0 * sd), 4)

        # Collect objections and failure scenarios from all perspectives
        all_objections: list[str] = []
        seen_obj: set[str] = set()
        for c in critiques:
            for obj in c.get("objections", []):
                if obj not in seen_obj:
                    all_objections.append(obj)
                    seen_obj.add(obj)

        all_failures: list[str] = []
        seen_fail: set[str] = set()
        for c in critiques:
            for fail in c.get("failure_scenarios", []):
                if fail not in seen_fail:
                    all_failures.append(fail)
                    seen_fail.add(fail)

        provider, model = self.router.identity(ModelRole.SCIENTIFIC_CRITIC)
        provenance = ArtifactProvenance.create(
            project_id=candidate.project_id, run_id=candidate.run_id,
            source_evidence=[prior_art.id] if prior_art else [],
            provider=provider, model=model,
            prompt_version=PROMPT_VERSION,
            content={**aggregated, "agreement": agreement, "perspectives": [c["tag"] for c in critiques],
                     "ensemble_size": len(critiques)},
        )

        return NoveltyCritique(
            id=uuid4().hex, project_id=candidate.project_id, run_id=candidate.run_id,
            candidate_id=candidate.id,
            novelty_score=aggregated["novelty_score"],
            significance_score=aggregated["significance_score"],
            causal_plausibility_score=aggregated["causal_plausibility_score"],
            falsifiability_score=aggregated["falsifiability_score"],
            experimental_discriminability_score=aggregated["experimental_discriminability_score"],
            feasibility_score=aggregated["feasibility_score"],
            implementation_risk_score=aggregated["implementation_risk_score"],
            expected_information_gain_score=aggregated["expected_information_gain_score"],
            compute_efficiency_score=aggregated["compute_efficiency_score"],
            publication_potential_score=aggregated["publication_potential_score"],
            objections=tuple(all_objections),
            failure_scenarios=tuple(all_failures),
            provenance=provenance,
        )

    async def _single_evaluate(self, candidate: NoveltyCandidate, perspective: dict, *, prior_art: PriorArtAssessment | None = None) -> dict:
        """Run one critic evaluation from a specific perspective."""
        score_keys = [
            "novelty_score", "significance_score", "causal_plausibility_score",
            "falsifiability_score", "experimental_discriminability_score",
            "feasibility_score", "implementation_risk_score",
            "expected_information_gain_score", "compute_efficiency_score",
            "publication_potential_score",
        ]
        # json_object mode drops the schema, so the contract must be in the
        # prompt itself: enumerate every expected key with its range.
        instruction = (
            "Score the idea from your perspective. Return ONLY a JSON object with "
            "exactly these keys, each score a number between 0.0 and 1.0:\n"
            + "\n".join(f"- {key}: number 0.0-1.0" for key in score_keys)
            + "\n- objections: array of strings (explicit objections)\n"
            + "- failure_scenarios: array of strings (concrete ways the idea fails)\n"
            + "Every score key must be present. No other keys."
        )
        messages = [
            {"role": "system", "content": perspective["system"]},
            {
                "role": "user",
                "content": (
                    f"Prompt version: {PROMPT_VERSION}\n"
                    f"Method: {candidate.method}\n"
                    f"Mechanism: {candidate.causal_mechanism}\n"
                    f"Falsification condition: {candidate.falsification_condition}\n"
                    f"Expected effect: {candidate.expected_effect}\n"
                    f"Closest-method difference: {candidate.difference_from_existing}\n"
                    f"Minimum experiment: {candidate.minimum_validating_experiment}\n"
                    + (f"Retrieved prior-art assessment (bounded, not proof of novelty):\n"
                       f"Overlap: {prior_art.overlap_status.value}\n"
                       f"Comparison: {prior_art.mechanism_comparison}\n"
                       f"Unresolved: {prior_art.unresolved_overlap}\n" if prior_art else "")
                    + f"\n{instruction}"
                ),
            },
        ]

        payload = await self.router.structured_output(
            ModelRole.SCIENTIFIC_CRITIC,
            messages=messages,
            schema={"name": "novelty_critique", "schema": CRITIQUE_SCHEMA},
            project_id=candidate.project_id, run_id=candidate.run_id,
            operation=f"novelty_criticism_{perspective['tag']}",
        )

        # json_object mode does not enforce the schema, so validate: a missing
        # score key must fail loudly rather than silently persist all-zero.
        parsed: dict[str, float] = {}
        for key in score_keys:
            value = payload.get(key)
            if value is None:
                raise ValueError(
                    f"Novelty critic omitted required score key '{key}'. "
                    f"Raw payload keys: {sorted(payload)}"
                )
            parsed[key] = _to_score(value, key)

        return {
            "tag": perspective["tag"],
            **parsed,
            "objections": list(payload.get("objections", [])),
            "failure_scenarios": list(payload.get("failure_scenarios", [])),
        }


def _to_score(value: object, key: str) -> float:
    """Coerce a 0-1 score from number or numeric string, clamped to [0, 1]."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Novelty critic returned non-numeric '{key}': {value!r}")
    if not 0.0 <= score <= 1.0:
        raise ValueError(f"Novelty critic returned '{key}' outside 0-1: {value!r}")
    return score
