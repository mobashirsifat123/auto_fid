"""Purpose: Freezes a selected novelty candidate into a versioned, structured
experimental methodology.

"Finalize methodology" means: freeze a concrete, testable specification of
what will be implemented and compared. It does NOT certify novelty or
scientific truth — the frozen record always carries an explicit uncertainty
statement and evidence linkage is bookkeeping, not validation.
"""

from __future__ import annotations

import json
from uuid import uuid4

from autoresearch.domain.models import (
    ArtifactProvenance,
    ArtifactReference,
    MethodologyFreeze,
    NoveltyCandidate,
    content_checksum,
    utc_now,
)
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.router import ModelRouter
from autoresearch.review.evidence_context import build_evidence_context
from autoresearch.storage.repository import ResearchRepository

PROMPT_VERSION = "methodology-freeze-v1"

METHODOLOGY_SCHEMA = {
    "type": "object",
    "properties": {
        "hypothesis_statement": {"type": "string"},
        "predicted_observations": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "proposed_mechanism": {"type": "string"},
        "competing_explanations": {"type": "array", "items": {"type": "string"}},
        "falsification_conditions": {"type": "array", "items": {"type": "string"}, "minItems": 1},
        "closest_prior_methods": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "paper_id": {"type": "string"},
                    "method_summary": {"type": "string"},
                    "proposed_difference": {"type": "string"},
                },
                "required": ["paper_id", "method_summary", "proposed_difference"],
                "additionalProperties": False,
            },
        },
        "algorithm_outline": {"type": "string", "description": "Concrete algorithm: pseudocode, equations, or step-by-step procedure"},
        "inputs": {"type": "string"},
        "outputs": {"type": "string"},
        "components": {"type": "array", "items": {"type": "string"}},
        "procedure": {"type": "array", "items": {"type": "string"}, "minItems": 1, "description": "Training/analysis steps in order"},
        "implementation_requirements": {"type": "string"},
        "expected_resource_demands": {"type": "string"},
        "supporting_evidence_claim_ids": {"type": "array", "items": {"type": "string"}},
        "uncertainty_statement": {"type": "string", "description": "What remains uncertain about novelty and scientific validity; never empty"},
    },
    "required": [
        "hypothesis_statement", "predicted_observations", "proposed_mechanism",
        "competing_explanations", "falsification_conditions", "closest_prior_methods",
        "algorithm_outline", "inputs", "outputs", "components", "procedure",
        "implementation_requirements", "expected_resource_demands",
        "supporting_evidence_claim_ids", "uncertainty_statement",
    ],
    "additionalProperties": False,
}

_FIELD_CONTRACT = (
    "Return ONLY a JSON object with exactly these keys:\n"
    "- hypothesis_statement: string (precise, testable)\n"
    "- predicted_observations: array of strings (what we expect to measure)\n"
    "- proposed_mechanism: string\n"
    "- competing_explanations: array of strings\n"
    "- falsification_conditions: array of strings\n"
    "- closest_prior_methods: array of {paper_id, method_summary, proposed_difference}\n"
    "- algorithm_outline: string (pseudocode/equations/steps; concrete)\n"
    "- inputs: string\n- outputs: string\n- components: array of strings\n"
    "- procedure: array of strings (ordered training/analysis steps)\n"
    "- implementation_requirements: string\n"
    "- expected_resource_demands: string\n"
    "- supporting_evidence_claim_ids: array of claim IDs from the supplied evidence only\n"
    "- uncertainty_statement: string (mandatory, non-empty)\n"
    "Every key must be present. Do not invent paper or claim IDs that were not supplied."
)


class MethodologyFreezer:
    def __init__(self, repository: ResearchRepository, router: ModelRouter) -> None:
        self.repository = repository
        self.router = router

    async def freeze(
        self,
        candidate: NoveltyCandidate,
        *,
        selection_rationale: str,
        portfolio_id: str | None = None,
    ) -> MethodologyFreeze:
        """Freeze the candidate's methodology as the next version.

        Grounded in the same verified evidence context as generation; the
        prior-art closest methods come from the stored assessment.
        """
        gap_row = (
            self.repository.row("SELECT * FROM research_gaps WHERE id=? AND project_id=?", (candidate.gap_id, candidate.project_id))
            if candidate.gap_id else None
        )
        try:
            import json as _json
            supporting_paper_ids = tuple(_json.loads(gap_row["supporting_paper_ids_json"])) if gap_row else tuple(candidate.provenance.source_evidence)
            supporting_claim_ids = tuple(_json.loads(gap_row["supporting_claim_ids_json"])) if gap_row else ()
            counter_ids = tuple(_json.loads(gap_row.get("counter_evidence_claim_ids_json") or "[]")) if gap_row else ()
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"methodology_invalid: malformed gap JSON for gap {candidate.gap_id}"
            ) from exc
        evidence_context = build_evidence_context(
            self.repository,
            project_id=candidate.project_id,
            supporting_paper_ids=supporting_paper_ids,
            supporting_claim_ids=supporting_claim_ids,
            counter_claim_ids=counter_ids,
            candidate_id=candidate.id,
            gap_row=gap_row,
        )
        from autoresearch.review.evidence_context import format_evidence_context
        grounding = format_evidence_context(evidence_context)

        payload = await self.router.structured_output(
            ModelRole.RESEARCH_REASONING,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You convert a selected research candidate into a frozen, versioned experimental "
                        "methodology. Be concrete: algorithmic steps, equations or pseudocode, inputs/outputs, "
                        "and resource demands. Freezing does not certify novelty: the uncertainty_statement "
                        "must state what remains unverified."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Prompt version: {PROMPT_VERSION}\n"
                        f"Candidate strategy: {candidate.strategy}\n"
                        f"Candidate method: {candidate.method}\n"
                        f"Causal mechanism: {candidate.causal_mechanism}\n"
                        f"Difference from existing: {candidate.difference_from_existing}\n"
                        f"Falsification condition: {candidate.falsification_condition}\n"
                        f"Expected effect: {candidate.expected_effect}\n\n{grounding}\n\n{_FIELD_CONTRACT}"
                    ),
                },
            ],
            schema={"name": "methodology", "schema": METHODOLOGY_SCHEMA},
            project_id=candidate.project_id,
            run_id=candidate.run_id,
            operation="methodology_freeze",
        )
        payload = _validate(payload, evidence_context)

        previous = self.repository.latest_methodology_freeze_for_candidate(candidate.id)
        version = (previous.version + 1) if previous else 1

        provider, model = self.router.identity(ModelRole.RESEARCH_REASONING)
        provenance = ArtifactProvenance.create(
            project_id=candidate.project_id,
            run_id=candidate.run_id,
            source_evidence=(candidate.id, *(c["paper_id"] for c in evidence_context.get("papers", []))),
            provider=provider, model=model,
            prompt_version=PROMPT_VERSION,
            content=payload,
        )
        record = {
            "methodology_freeze_id": uuid4().hex,
            "candidate_id": candidate.id,
            "portfolio_id": portfolio_id,
            "version": version,
            "selection_rationale": selection_rationale,
            "methodology": payload,
            "provenance": {
                "provider": provenance.provider, "model": provenance.model,
                "prompt_version": PROMPT_VERSION, "created_at": utc_now(),
                "checksum": content_checksum(payload), "source_evidence": list(provenance.source_evidence),
            },
        }
        text = json.dumps(record, indent=2, ensure_ascii=False, default=str)
        root = self.repository.artifact_root / candidate.project_id / candidate.run_id
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"methodology-v{version}.json"
        path.write_text(text, encoding="utf-8")

        freeze = MethodologyFreeze(
            id=record["methodology_freeze_id"],
            project_id=candidate.project_id,
            run_id=candidate.run_id,
            candidate_id=candidate.id,
            version=version,
            selection_rationale=selection_rationale,
            prompt_version=PROMPT_VERSION,
            artifact_path=str(path),
            checksum=content_checksum(text),
        )
        self.repository.save_methodology_freeze(freeze)
        self.repository.save_artifact(ArtifactReference(
            content_checksum(f"methodology|{freeze.id}")[:32],
            project_id=candidate.project_id, run_id=candidate.run_id,
            artifact_type=f"methodology_v{version}_json", path=str(path),
            source_evidence=(candidate.id,),
            provider=provider, model=model,
            prompt_version=PROMPT_VERSION, checksum=freeze.checksum,
        ))
        return freeze


def _validate(payload: dict, evidence_context: dict) -> dict:
    """Deterministic schema/enforcement checks — prompt instructions alone do
    not enforce this contract. Raises ValueError on malformed output."""
    required = METHODOLOGY_SCHEMA["required"]
    missing = [key for key in required if not payload.get(key)]
    if missing:
        raise ValueError(f"methodology_invalid: missing required fields {missing}")
    for key in ("predicted_observations", "falsification_conditions", "procedure"):
        value = payload.get(key)
        if not isinstance(value, list) or not value:
            raise ValueError(f"methodology_invalid: '{key}' must be a non-empty array")
    supplied_claims = {str(c.get("claim_id")) for c in evidence_context.get("claims", [])}
    supplied_claims |= {str(c.get("claim_id")) for c in evidence_context.get("counter_evidence_claims", [])}
    payload["supporting_evidence_claim_ids"] = [
        str(cid) for cid in payload.get("supporting_evidence_claim_ids", []) if str(cid) in supplied_claims
    ]
    supplied_papers = {str(p.get("paper_id")) for p in evidence_context.get("papers", [])}
    payload["closest_prior_methods"] = [
        entry for entry in payload.get("closest_prior_methods", [])
        if isinstance(entry, dict) and str(entry.get("paper_id")) in supplied_papers
    ]
    # Filtering must not silently empty the linkage: when the evidence context
    # supplied claims, a freeze citing none of them is ungrounded, not valid.
    if supplied_claims and not payload["supporting_evidence_claim_ids"]:
        raise ValueError(
            "methodology_ungrounded: supporting_evidence_claim_ids cites none of the supplied "
            "verified claims; regenerate with the evidence context or freeze without one"
        )
    if supplied_papers and not payload["closest_prior_methods"]:
        raise ValueError(
            "methodology_ungrounded: closest_prior_methods cites none of the supplied papers; "
            "the frozen methodology must state its concrete difference from the reviewed corpus"
        )
    uncertainty = " ".join(str(payload.get("uncertainty_statement") or "").split())
    if len(uncertainty) < 20:
        raise ValueError("methodology_invalid: uncertainty_statement must be a substantive statement")
    return payload
