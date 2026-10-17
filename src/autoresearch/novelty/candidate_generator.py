"""Purpose: Generates innovative candidate solutions targeting a specific research gap."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from uuid import uuid4

from autoresearch.domain.models import (
    ArtifactProvenance,
    CandidateStatus,
    NoveltyCandidate,
    ResearchGap,
    CandidateRevision,
    NoveltyCritique,
    PriorArtAssessment,
)
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.router import ModelRouter

logger = logging.getLogger(__name__)
from autoresearch.review.evidence_context import format_evidence_context

CANDIDATE_SCHEMA = {
    "type": "object",
    "properties": {
        "strategy": {"type": "string"},
        "method": {"type": "string"},
        "causal_mechanism": {"type": "string"},
        "target_limitation": {"type": "string"},
        "difference_from_existing": {"type": "string"},
        "expected_effect": {"type": "string"},
        "falsification_condition": {"type": "string"},
        "minimum_validating_experiment": {"type": "string"},
        "required_ablations": {"type": "array", "items": {"type": "string"}},
        "implementation_complexity": {"type": "string"},
        "expected_compute": {"type": "string"},
        "risks_and_objections": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "strategy", "method", "causal_mechanism", "target_limitation",
        "difference_from_existing", "expected_effect", "falsification_condition",
        "minimum_validating_experiment", "required_ablations",
        "implementation_complexity", "expected_compute", "risks_and_objections"
    ],
    "additionalProperties": False,
}

PROMPT_VERSION = "candidate-gen-v3"

# Literature grounding adapted from HKUDS/AI-Researcher @ f9a6f8480860c193afff600eeffe3defcee8a978
# (research_agent/inno/agents/inno_agent/idea_agent.py): the Idea Generation
# Agent must summarize existing methods from the actual papers before
# proposing, and tie its proposal to them. Mechanism reused; no upstream code
# copied (the upstream repository carries no license file).
_GROUNDING_CONTRACT = (
    "Before proposing, ground yourself in the reviewed literature supplied below: "
    "the papers' methods and the verified claim excerpts. "
    "Do not invent papers or claims that are not supplied."
)
# The citation clause is appended only when the context actually supplies
# papers; demanding a [paper_id] reference with no papers in the prompt would
# force either a contract violation or a hallucinated ID.
_CITATION_CONTRACT = (
    "difference_from_existing MUST name the closest prior method by its paper_id "
    "(format [paper_id]) and state the mechanism-level difference."
)

# json_object mode drops the schema, so the prompt must carry the contract:
# every expected key, enumerated. Without this the model returns renamed keys
# and every field defaults to "".
_FIELD_CONTRACT = (
    "Return ONLY a JSON object with exactly these keys:\n"
    "- strategy: string\n"
    "- method: string (the proposed method, specific and detailed)\n"
    "- causal_mechanism: string (why it works, mechanistic account)\n"
    "- target_limitation: string\n"
    "- difference_from_existing: string (how it differs from prior work)\n"
    "- expected_effect: string (quantified where possible)\n"
    "- falsification_condition: string (a concrete condition that would refute it)\n"
    "- minimum_validating_experiment: string (the smallest decisive experiment)\n"
    "- required_ablations: array of strings\n"
    "- implementation_complexity: string\n"
    "- expected_compute: string\n"
    "- risks_and_objections: array of strings\n"
    "Every key must be present with non-empty values. No other keys."
)

_STRATEGY_SYSTEM_PROMPTS = {
    "limitation root cause elimination": (
        "You are a world-class AI researcher proposing a novel method to address a validated research gap. "
        "You MUST propose a method that directly removes the root cause of the documented limitation. "
        "Do NOT propose a method that merely mitigates symptoms or adds post-processing/regularization."
    ),
    "cross-domain mechanism transfer": (
        "You are a world-class AI researcher proposing a novel method to address a validated research gap. "
        "You MUST borrow a causal mechanism from a DIFFERENT scientific domain (e.g., control theory, biology, physics, signal processing). "
        "Do NOT propose mechanisms standard in the target domain."
    ),
    "contradiction resolution": (
        "You are a world-class AI researcher proposing a novel method to address a validated research gap. "
        "You MUST resolve a fundamental trade-off or contradiction between two opposing paradigms."
    ),
    "theoretical boundary extension": (
        "You are a world-class AI researcher proposing a novel method to address a validated research gap. "
        "You MUST extend a theoretical formulation beyond its current failure boundary or assumption set."
    ),
    "computational bottleneck bypass": (
        "You are a world-class AI researcher proposing a novel method to address a validated research gap. "
        "You MUST replace an expensive computational bottleneck with an algorithmic or statistical proxy."
    ),
}


class LLMCandidateGenerator:
    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    async def generate(self, gap: ResearchGap, count: int = 5, *, evidence_context: dict | None = None, on_candidate=None) -> tuple[NoveltyCandidate, ...]:
        """Generate candidates concurrently.

        ``on_candidate`` (optional) is invoked with each candidate the moment
        its model call completes — the pipeline uses it to persist partial
        progress at this expensive boundary, so an interrupted run leaves
        recoverable state instead of losing every finished generation.
        """
        strategies = [
            "limitation root cause elimination",
            "cross-domain mechanism transfer",
            "contradiction resolution",
            "theoretical boundary extension",
            "computational bottleneck bypass",
        ]
        selected_strategies = strategies[:min(count, len(strategies))]
        grounding_block = format_evidence_context(evidence_context)
        candidates: list[NoveltyCandidate] = []
        saved: set[str] = set()
        for coro in asyncio.as_completed([
            self._generate_single(gap, strategy, grounding_block, evidence_context)
            for strategy in selected_strategies
        ]):
            candidate = await coro
            candidates.append(candidate)
            if on_candidate is not None:
                try:
                    on_candidate(candidate)
                    saved.add(candidate.id)
                except Exception:  # noqa: BLE001 — persistence must not kill generation
                    logger.exception("on_candidate persistence failed for %s", candidate.id)
        # Deterministic strategy order (as_completed yields in completion
        # order; downstream dedup/selection must not depend on it).
        order = {strategy: index for index, strategy in enumerate(selected_strategies)}
        ordered = sorted(candidates, key=lambda c: order.get(c.strategy, len(order)))
        for candidate in ordered:
            if candidate.id not in saved and on_candidate is not None:
                try:
                    on_candidate(candidate)
                except Exception:  # noqa: BLE001
                    logger.exception("candidate persistence failed for %s", candidate.id)
        return tuple(ordered)

    async def _generate_single(
        self, gap: ResearchGap, strategy: str, grounding_block: str = "", evidence_context: dict | None = None,
    ) -> NoveltyCandidate:
        system_prompt = _STRATEGY_SYSTEM_PROMPTS.get(
            strategy,
            "You are a world-class AI researcher proposing a novel method to address a validated research gap. Return a precise, highly-detailed structured specification of your proposed method. Reject vague claims."
        )
        literature_section = f"\n\n{grounding_block}\n" if grounding_block else ""
        grounding_contract = ""
        if grounding_block:
            grounding_contract = " " + _GROUNDING_CONTRACT
            if evidence_context and evidence_context.get("papers"):
                grounding_contract += " " + _CITATION_CONTRACT
        messages = [
            {
                "role": "system",
                "content": system_prompt + grounding_contract,
            },
            {
                "role": "user",
                "content": (
                    f"Prompt version: {PROMPT_VERSION}\nResearch Gap:\n{gap.statement}\n"
                    f"Why Unsolved: {gap.why_unsolved}\n{literature_section}\n"
                    f"Generate a candidate using strategy: {strategy}.\n\n{_FIELD_CONTRACT}"
                ),
            },
        ]

        payload = await self.router.structured_output(
            ModelRole.RESEARCH_REASONING,
            messages=messages,
            schema={"name": "novelty_candidate", "schema": CANDIDATE_SCHEMA},
            project_id=gap.project_id,
            run_id=gap.run_id,
            operation="novelty_generation",
        )

        provider, model = self.router.identity(ModelRole.RESEARCH_REASONING)
        provenance = ArtifactProvenance.create(
            project_id=gap.project_id,
            run_id=gap.run_id,
            source_evidence=gap.supporting_paper_ids,
            provider=provider,
            model=model,
            prompt_version=PROMPT_VERSION,
            content={
                **payload,
                "evidence_linked_paper_ids": cited_paper_ids(
                    payload.get("difference_from_existing", ""), evidence_context,
                ),
            },
        )

        return NoveltyCandidate(
            id=uuid4().hex,
            project_id=gap.project_id,
            run_id=gap.run_id,
            gap_id=gap.id,
            strategy=payload.get("strategy", strategy),
            method=payload.get("method", ""),
            causal_mechanism=payload.get("causal_mechanism", ""),
            target_limitation=payload.get("target_limitation", ""),
            difference_from_existing=payload.get("difference_from_existing", ""),
            expected_effect=_model_hypothesis(payload.get("expected_effect", "")),
            falsification_condition=payload.get("falsification_condition", ""),
            minimum_validating_experiment=payload.get("minimum_validating_experiment", ""),
            required_ablations=tuple(payload.get("required_ablations", [])),
            implementation_complexity=payload.get("implementation_complexity", ""),
            expected_compute=payload.get("expected_compute", ""),
            risks_and_objections=tuple(payload.get("risks_and_objections", [])),
            status=CandidateStatus.GENERATED,
            provenance=provenance,
        )

    async def revise(self, candidate: NoveltyCandidate, assessment: PriorArtAssessment, critique: NoveltyCritique, revision_count: int = 1, *, evidence_context: dict | None = None) -> tuple[NoveltyCandidate, CandidateRevision]:
        if revision_count not in (1, 2):
            raise ValueError("Candidate revision is bounded to two rounds")

        score_items = [
            ("falsifiability", critique.falsifiability_score),
            ("experimental_discriminability", critique.experimental_discriminability_score),
            ("feasibility", critique.feasibility_score),
            ("causal_plausibility", critique.causal_plausibility_score),
            ("compute_efficiency", critique.compute_efficiency_score),
        ]
        weakest = sorted(score_items, key=lambda x: x[1])[:3]
        weakest_str = ", ".join(f"{name}={score:.2f}" for name, score in weakest)

        payload = await self.router.structured_output(
            ModelRole.RESEARCH_REASONING,
            messages=[
                {"role": "system", "content": "Revise the candidate to address documented overlap and critic objections. Narrow claims when needed. Do not claim novelty. Return the complete candidate schema."},
                {"role": "user", "content": f"Prompt version: candidate-revision-v2\nMethod: {candidate.method}\nOverlap: {assessment.mechanism_comparison}\nUncertainty: {assessment.unresolved_overlap}\nObjections: {critique.objections}\nWeakest dimensions to improve: {weakest_str}\nRevision round: {revision_count}\n\n{_FIELD_CONTRACT}"},
            ],
            schema={"name": "candidate_revision", "schema": CANDIDATE_SCHEMA},
            project_id=candidate.project_id, run_id=candidate.run_id, operation="candidate_revision",
        )
        provider, model = self.router.identity(ModelRole.RESEARCH_REASONING)
        provenance = ArtifactProvenance.create(project_id=candidate.project_id, run_id=candidate.run_id,
            source_evidence=[candidate.id, assessment.id, critique.id], provider=provider, model=model,
            prompt_version="candidate-revision-v2",
            content={**payload, "evidence_linked_paper_ids": cited_paper_ids(payload.get("difference_from_existing", ""), evidence_context)})
        revised = NoveltyCandidate(
            uuid4().hex, candidate.project_id, candidate.run_id, candidate.gap_id,
            payload.get("strategy", candidate.strategy), payload.get("method", ""), payload.get("causal_mechanism", ""),
            payload.get("target_limitation", ""), payload.get("difference_from_existing", ""), _model_hypothesis(payload.get("expected_effect", "")),
            payload.get("falsification_condition", ""), payload.get("minimum_validating_experiment", ""),
            tuple(payload.get("required_ablations", [])), payload.get("implementation_complexity", ""), payload.get("expected_compute", ""),
            tuple(payload.get("risks_and_objections", [])), CandidateStatus.GENERATED, provenance,
        )
        revision = CandidateRevision(uuid4().hex, candidate.project_id, candidate.run_id, candidate.id, revised.id,
            critique.id, revision_count, assessment.mechanism_comparison or "Revised to reduce documented overlap.", provenance)
        return revised, revision


def cited_paper_ids(text: str, evidence_context: dict | None) -> list[str]:
    """Record which supplied paper IDs the model actually referenced.

    This is provenance bookkeeping, not a validation gate: an empty list means
    the model did not cite the corpus, which reviewers can see in the stored
    provenance content and the novelty-grounding artifact instead of trusting
    a difference_from_existing claim.
    """
    if not evidence_context:
        return []
    supplied = {str(p.get("paper_id")) for p in evidence_context.get("papers", [])}
    cited = set(re.findall(r"\[([A-Za-z0-9_-]+)\]", str(text or "")))
    return sorted(cited & supplied)


def _model_hypothesis(value: object) -> str:
    text = str(value or "").strip()
    if not text or text.startswith("MODEL_HYPOTHESIS:"):
        return text
    return f"MODEL_HYPOTHESIS: {text}"
