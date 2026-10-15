"""Purpose: Identifies unaddressed limitations across the top-ranked papers and synthesizes a concrete, falsifiable research gap."""

from __future__ import annotations

import asyncio
import json
from typing import Sequence
from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, EvidenceClaim, GapClassification, PaperRecord, PaperReview, PriorityScore, ResearchGap
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.router import ModelRouter

GAP_PROMPT_VERSION = "evidence-gap-v2"
COUNTER_EVIDENCE_PROMPT_VERSION = "counter-evidence-v1"
LIMITATION_CLUSTER_PROMPT_VERSION = "limitation-cluster-v1"
GAP_SELECTION_PROMPT_VERSION = "gap-selection-v1"
_ID_ARRAY = {"type": "array", "items": {"type": "string"}}

GAP_SCHEMA = {
    "type": "object",
    "properties": {
        "gap_statement": {"type": "string"},
        "supporting_paper_ids": _ID_ARRAY,
        "supporting_claim_ids": _ID_ARRAY,
        "why_existing_work_does_not_solve_it": {"type": "string"},
        "missing_experiment": {"type": "string"},
        "falsification_test": {"type": "string"},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "counter_evidence_claim_ids": _ID_ARRAY,
    },
    "required": ["gap_statement", "supporting_paper_ids", "supporting_claim_ids", "why_existing_work_does_not_solve_it", "missing_experiment", "falsification_test", "confidence", "counter_evidence_claim_ids"],
    "additionalProperties": False,
}

GAP_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "selected_index": {"type": "integer", "minimum": 0, "maximum": 2},
        "selection_rationale": {"type": "string"},
    },
    "required": ["selected_index", "selection_rationale"],
    "additionalProperties": False,
}

LIMITATION_CLUSTER_SCHEMA = {
    "type": "object",
    "properties": {
        "clusters": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "theme": {"type": "string"},
                    "frequency": {"type": "integer", "minimum": 1},
                    "paper_ids": _ID_ARRAY,
                    "description": {"type": "string"},
                },
                "required": ["theme", "frequency", "paper_ids", "description"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["clusters"],
    "additionalProperties": False,
}


class GapAnalyzer:
    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    async def analyze(
        self, *,
        project_id: str, run_id: str,
        papers: Sequence[PaperRecord], reviews: Sequence[PaperReview],
        claims: Sequence[EvidenceClaim], scores: Sequence[PriorityScore],
    ) -> ResearchGap:
        top_ids = {score.paper_id for score in scores}
        paper_map = {paper.id: paper for paper in papers if paper.id in top_ids}
        review_map = {review.paper_id: review for review in reviews if review.paper_id in top_ids}
        claim_map = {claim.id: claim for claim in claims if claim.paper_id in top_ids}
        if not paper_map or not claim_map:
            raise ValueError("Gap analysis requires ranked papers with validated evidence claims.")

        context = self._build_context(paper_map, review_map, claim_map)
        provider, model = self.router.identity(ModelRole.SCIENTIFIC_CRITIC)

        # Step 1: Cluster limitations across papers for structured insight
        limitation_clusters = await self._cluster_limitations(
            project_id, run_id, paper_map, review_map, provider, model,
        )

        # Step 2: Generate 3 gap candidates independently
        gap_candidates = await self._generate_gap_candidates(
            project_id, run_id, context, claim_map, provider, model,
        )

        # Step 3: For each candidate, search for counter-evidence explicitly
        # (parallel — these are independent adversarial reads of the corpus).
        counters = await asyncio.gather(*[
            self._find_counter_evidence(
                project_id, run_id, candidate_payload, context, claim_map, provider, model,
            )
            for candidate_payload in gap_candidates
        ])
        enhanced_candidates = []
        for candidate_payload, counter in zip(gap_candidates, counters):
            candidate_payload["counter_evidence_claim_ids"] = counter
            enhanced_candidates.append(candidate_payload)

        # Step 4: Select the best gap candidate using a meta-evaluation
        selected = await self._select_best_gap(
            project_id, run_id, enhanced_candidates, limitation_clusters, provider, model,
        )

        return validate_gap_payload(
            selected, project_id=project_id, run_id=run_id,
            valid_papers=paper_map, valid_claims=claim_map,
            provider=provider, model=model,
        )

    def _build_context(self, paper_map, review_map, claim_map) -> dict:
        return {
            "papers": [
                {
                    "paper_id": paper.id, "title": paper.title, "year": paper.year,
                    "review": {
                        "method": review_map[paper.id].method,
                        "limitations": review_map[paper.id].limitations,
                        "failure_modes": review_map[paper.id].failure_modes,
                    },
                }
                for paper in paper_map.values() if paper.id in review_map
            ],
            "claims": [
                {
                    "claim_id": claim.id, "paper_id": claim.paper_id,
                    "statement": claim.statement, "excerpt": claim.excerpt,
                    "page": claim.page, "section": claim.section,
                }
                for claim in claim_map.values()
            ],
        }

    async def _cluster_limitations(self, project_id, run_id, paper_map, review_map, provider, model) -> list[dict]:
        """Cluster limitations across papers to identify recurring themes."""
        all_limitations = []
        for paper in paper_map.values():
            if paper.id in review_map:
                review = review_map[paper.id]
                for lim in review.limitations:
                    all_limitations.append({"paper_id": paper.id, "limitation": lim, "title": paper.title})
                for fm in review.failure_modes:
                    all_limitations.append({"paper_id": paper.id, "limitation": fm, "title": paper.title, "is_failure_mode": True})

        if len(all_limitations) < 3:
            return []

        payload = await self.router.structured_output(
            ModelRole.SCIENTIFIC_CRITIC,
            messages=[
                {"role": "system", "content": "Cluster the following paper limitations and failure modes into recurring themes. Group them by common root causes. Each cluster should have at least 2 papers."},
                {"role": "user", "content": f"Prompt version: {LIMITATION_CLUSTER_PROMPT_VERSION}\n{json.dumps(all_limitations, ensure_ascii=False)}"},
            ],
            schema={"name": "limitation_clusters", "schema": LIMITATION_CLUSTER_SCHEMA},
            project_id=project_id, run_id=run_id, operation="limitation_clustering",
        )
        return payload.get("clusters", [])

    async def _generate_gap_candidates(self, project_id, run_id, context, claim_map, provider, model) -> list[dict]:
        """Generate 3 distinct gap candidates concurrently (R6)."""
        perspectives = [
            "methodological: a gap in the core methods/architectures used across these papers",
            "evaluation: a gap in how results are measured or compared, suggesting a missing benchmark or protocol",
            "theoretical: a gap in the underlying theory or assumptions that limits generalization",
        ]
        tasks = [
            self.router.structured_output(
                ModelRole.SCIENTIFIC_CRITIC,
                messages=[
                    {"role": "system", "content": f"Identify one precise research gap from the {perspective} perspective. Use only the supplied paper and claim IDs. Unsupported statements must be labeled as hypotheses."},
                    {"role": "user", "content": f"Prompt version: {GAP_PROMPT_VERSION}\n{json.dumps(context, ensure_ascii=False)}\nReturn gap_statement, supporting_paper_ids[], supporting_claim_ids[], why_existing_work_does_not_solve_it, missing_experiment, falsification_test, confidence."},
                ],
                schema={"name": "research_gap", "schema": GAP_SCHEMA},
                project_id=project_id, run_id=run_id, operation=f"gap_analysis_{perspective.split(':')[0]}",
            )
            for perspective in perspectives
        ]
        return list(await asyncio.gather(*tasks))

    async def _find_counter_evidence(self, project_id, run_id, candidate, context, claim_map, provider, model) -> list[str]:
        """Explicitly search for evidence that contradicts or weakens the gap claim."""
        payload = await self.router.structured_output(
            ModelRole.SCIENTIFIC_CRITIC,
            messages=[
                {"role": "system", "content": "You are an adversarial reviewer. Given a proposed research gap, actively search the supplied evidence for claims, findings, or statements that CONTRADICT or weaken the gap. If no counter-evidence exists, return an empty array. Never return supporting claim IDs as counter-evidence."},
                {"role": "user", "content": f"Prompt version: {COUNTER_EVIDENCE_PROMPT_VERSION}\nProposed gap: {candidate.get('gap_statement', '')}\nSupporting papers: {candidate.get('supporting_paper_ids', [])}\n\nEvidence corpus:\n{json.dumps(context, ensure_ascii=False)}\n\nReturn counter_evidence_claim_ids[] — only claim IDs that directly undermine the gap statement."},
            ],
            schema={"name": "counter_evidence_search", "schema": {"type": "object", "properties": {"counter_evidence_claim_ids": _ID_ARRAY, "analysis": {"type": "string"}}, "required": ["counter_evidence_claim_ids", "analysis"], "additionalProperties": False}},
            project_id=project_id, run_id=run_id, operation="counter_evidence_search",
        )
        return [str(cid) for cid in payload.get("counter_evidence_claim_ids", []) if str(cid) in claim_map]

    async def _select_best_gap(self, project_id, run_id, candidates, limitation_clusters, provider, model) -> dict:
        """Select the best gap candidate via meta-evaluation."""
        cluster_summary = ""
        if limitation_clusters:
            cluster_summary = "\n".join(
                f"- Theme '{c.get('theme', 'unlabeled')}' (appears in {c.get('n', c.get('frequency', 0))} papers): {c.get('description', '')}"
                for c in limitation_clusters
            )
        candidates_text = "\n\n---\n\n".join(
            f"Candidate {i}:\nGap: {c.get('gap_statement', '')}\nConfidence: {c.get('confidence', 0.5)}\nCounter-evidence count: {len(c.get('counter_evidence_claim_ids', []))}"
            for i, c in enumerate(candidates)
        )
        prompt = (
            f"Prompt version: {GAP_SELECTION_PROMPT_VERSION}\n"
            f"Recurring limitation themes across papers:\n{cluster_summary or 'None identified'}\n\n"
            f"Candidate gaps to evaluate:\n{candidates_text}\n\n"
            "Select the best gap candidate by index (0, 1, or 2). The best gap is the one that:\n"
            "1. Is most precisely stated and falsifiable\n"
            "2. Has the strongest support from paper evidence\n"
            "3. Has the weakest counter-evidence\n"
            "4. Addresses a recurring limitation theme\n"
            "Return selected_index and selection_rationale."
        )
        selection = await self.router.structured_output(
            ModelRole.SCIENTIFIC_CRITIC,
            messages=[
                {"role": "system", "content": "Evaluate the three gap candidates and select the best one based on precision, evidence support, counter-evidence weakness, and alignment with recurring limitation themes."},
                {"role": "user", "content": prompt},
            ],
            schema={"name": "gap_selection", "schema": GAP_SELECTION_SCHEMA},
            project_id=project_id, run_id=run_id, operation="gap_selection",
        )
        idx = max(0, min(selection.get("selected_index", 0), len(candidates) - 1))
        selected = candidates[idx]
        selected["_selection_rationale"] = selection.get("selection_rationale", "")
        selected["_limitation_clusters"] = limitation_clusters
        return selected


def validate_gap_payload(payload: dict, *, project_id: str, run_id: str, valid_papers: dict[str, PaperRecord], valid_claims: dict[str, EvidenceClaim], provider: str, model: str) -> ResearchGap:
    paper_ids = tuple(dict.fromkeys(value for value in _ids(payload.get("supporting_paper_ids")) if value in valid_papers))
    claim_ids = tuple(dict.fromkeys(value for value in _ids(payload.get("supporting_claim_ids")) if value in valid_claims and valid_claims[value].paper_id in paper_ids))
    counter_ids = tuple(dict.fromkeys(value for value in _ids(payload.get("counter_evidence_claim_ids")) if value in valid_claims))
    classification = GapClassification.EVIDENCE_SUPPORTED if paper_ids and claim_ids else GapClassification.HYPOTHESIS

    # Enrich the gap with limitation cluster context if available
    meta = {}
    if payload.get("_selection_rationale"):
        meta["selection_rationale"] = payload["_selection_rationale"]
    if payload.get("_limitation_clusters"):
        meta["limitation_clusters"] = payload["_limitation_clusters"]

    content = {
        "statement": _required(payload, "gap_statement"), "supporting_paper_ids": paper_ids, "supporting_claim_ids": claim_ids,
        "why_unsolved": _required(payload, "why_existing_work_does_not_solve_it"), "missing_experiment": _required(payload, "missing_experiment"),
        "falsification_test": _required(payload, "falsification_test"), "counter_evidence_claim_ids": counter_ids, "classification": classification.value,
        **meta,
    }
    source_evidence = (*claim_ids, *counter_ids)
    provenance = ArtifactProvenance.create(project_id=project_id, run_id=run_id, source_evidence=source_evidence, provider=provider, model=model, prompt_version=GAP_PROMPT_VERSION, content=content)

    # R7: Calibrated confidence score based on LLM rating, evidence coverage, claim density, and counter-evidence penalty
    raw_conf = _confidence(payload.get("confidence"))
    coverage = len(paper_ids) / max(1, len(valid_papers))
    claim_density = min(1.0, len(claim_ids) / 5.0)
    counter_penalty = 0.15 * len(counter_ids)
    calibrated_conf = round(max(0.0, min(1.0, raw_conf * 0.4 + coverage * 0.3 + claim_density * 0.3 - counter_penalty)), 6)

    return ResearchGap(uuid4().hex, project_id, run_id, content["statement"], paper_ids, claim_ids, content["why_unsolved"], content["missing_experiment"], content["falsification_test"], calibrated_conf, counter_ids, classification, provenance)


def _ids(value: object) -> tuple[str, ...]:
    return tuple(str(item) for item in value) if isinstance(value, list) else ()


def _required(payload: dict, key: str) -> str:
    value = " ".join(str(payload.get(key) or "").split())
    if not value:
        raise ValueError(f"Gap field '{key}' is required.")
    return value


def _confidence(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
