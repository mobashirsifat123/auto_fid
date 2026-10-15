"""Purpose: Compares candidate methods against prior literature to detect overlap."""

from __future__ import annotations

import re
from uuid import uuid4

from autoresearch.domain.models import (
    ArtifactProvenance,
    NoveltyCandidate,
    PriorArtAssessment,
    PriorArtStatus,
    utc_now,
)
from autoresearch.papers.search_pipeline import StudySearchPipeline
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.router import ModelRouter

PRIOR_ART_SCHEMA = {
    "type": "object",
    "properties": {
        "overlap_status": {
            "type": "string",
            "enum": ["unsupported", "potentially_novel", "partial_overlap", "incremental", "known_method", "expert_review_required"]
        },
        "mechanism_comparison": {"type": "string"},
        "supporting_excerpts": {"type": "array", "items": {"type": "string"}},
        "unresolved_overlap": {"type": "string"},
    },
    "required": ["overlap_status", "mechanism_comparison", "supporting_excerpts", "unresolved_overlap"],
    "additionalProperties": False,
}

PROMPT_VERSION = "prior-art-v2"


class EvidenceBackedPriorArtChecker:
    def __init__(self, router: ModelRouter, search_pipeline: StudySearchPipeline) -> None:
        self.router = router
        self.search_pipeline = search_pipeline

    async def check(self, candidate: NoveltyCandidate) -> PriorArtAssessment:
        # Step 1: Search the project's verified evidence corpus (internal)
        repository = getattr(self.search_pipeline, "repository", None)
        terms = set(re.findall(r"[a-z0-9]{3,}", f"{candidate.method} {candidate.causal_mechanism} {candidate.difference_from_existing}".lower()))
        internal_evidence: list[dict] = []
        if repository is not None:
            for claim in repository.rows(
                "SELECT e.*,p.title,p.year,p.venue FROM evidence_claims e JOIN papers p ON p.id=e.paper_id WHERE e.project_id=?",
                (candidate.project_id,),
            ):
                evidence_terms = set(re.findall(r"[a-z0-9]{3,}", f"{claim['statement']} {claim['excerpt']} {claim.get('section') or ''}".lower()))
                internal_evidence.append((len(terms & evidence_terms), claim))
        internal_selected = [claim for score, claim in sorted(internal_evidence, key=lambda item: (-item[0], item[1]["id"])) if score > 0][:12]

        # Step 2: External prior-art search using search pipeline sources directly
        import asyncio
        external_results: list[dict] = []
        search_queries = [
            candidate.method[:200],
            f"{candidate.causal_mechanism[:200]} machine learning",
            f"{candidate.difference_from_existing[:200]} method",
        ]
        if self.search_pipeline and hasattr(self.search_pipeline, 'sources') and self.search_pipeline.sources:
            external_tasks = []
            for source in self.search_pipeline.sources[:2]:
                for query in search_queries:
                    if not query.strip():
                        continue
                    external_tasks.append(source.search(query, limit=5))
            if external_tasks:
                ext_results = await asyncio.gather(*external_tasks, return_exceptions=True)
                for result in ext_results:
                    if isinstance(result, Exception):
                        continue
                    for paper in getattr(result, 'papers', [])[:5]:
                        paper_id = getattr(paper, 'id', None) or str(uuid4().hex)[:24]
                        external_results.append({
                            "paper_id": paper_id,
                            "title": getattr(paper, 'title', ''),
                            "year": getattr(paper, 'year', None),
                            "abstract": (getattr(paper, 'abstract', '') or '')[:2000],
                            "source": getattr(result, 'source', 'external_search'),
                        })

        # Combine evidence contexts
        closest_paper_ids = list(dict.fromkeys(
            claim["paper_id"] for claim in internal_selected
        ))

        evidence_parts = []
        if internal_selected:
            evidence_parts.append("=== Project Evidence Claims ===")
            for claim in internal_selected:
                evidence_parts.append(
                    f"[{claim['paper_id']}] {claim['title']} ({claim.get('year') or 'n.d.'}), "
                    f"section={claim.get('section') or 'unknown'}, page={claim.get('page') or 'unknown'}: "
                    f"{claim['excerpt']}"
                )
        if external_results:
            evidence_parts.append("\n=== External Search Results (for prior-art comparison) ===")
            for paper in external_results:
                evidence_parts.append(
                    f"[{paper['paper_id']}] {paper['title']} ({paper.get('year') or 'n.d.'}): "
                    f"{paper['abstract'][:500]}"
                )
            closest_paper_ids.extend(
                p["paper_id"] for p in external_results if p["paper_id"] not in closest_paper_ids
            )

        evidence_context = "\n".join(evidence_parts) if evidence_parts else (
            "No matching verified evidence was retrieved from the project corpus "
            "or from external search. Retrieval coverage is incomplete."
        )

        # Step 3: Query the LLM to assess overlap (with external context)
        messages = [
            {"role": "system", "content": "You are a stringent prior-art analyst. Compare mechanisms only against the supplied verified excerpts. Pay special attention to external search results that may represent concurrent or prior work. Preserve ambiguity and counter-evidence. Never claim exhaustive novelty."},
            {"role": "user", "content": f"Proposed Method: {candidate.method}\nMechanism: {candidate.causal_mechanism}\nDifference from existing: {candidate.difference_from_existing}\n\nPrior Art Abstracts and Section Evidence (including external search results):\n{evidence_context}"}
        ]

        payload = await self.router.structured_output(
            ModelRole.RESEARCH_REASONING,
            messages=messages,
            schema={"name": "prior_art_assessment", "schema": PRIOR_ART_SCHEMA},
            project_id=candidate.project_id,
            run_id=candidate.run_id,
            operation="prior_art_check",
        )

        # A corpus-only check cannot legitimately establish broad novelty.  The
        # explicit status preserves what was searched instead of turning an LLM
        # absence-of-evidence judgement into a novelty claim.
        if not external_results and payload.get("overlap_status") == PriorArtStatus.POTENTIALLY_NOVEL.value:
            payload["overlap_status"] = PriorArtStatus.EXPERT_REVIEW_REQUIRED.value
            payload["unresolved_overlap"] = (
                "CORPUS_BOUNDED_ASSESSMENT: not found in the reviewed corpus; "
                "external prior-art retrieval was unavailable."
            )

        provider, model = self.router.identity(ModelRole.RESEARCH_REASONING)
        provenance = ArtifactProvenance.create(
            project_id=candidate.project_id,
            run_id=candidate.run_id,
            source_evidence=closest_paper_ids,
            provider=provider,
            model=model,
            prompt_version=PROMPT_VERSION,
            content={**payload, "external_search_included": bool(external_results),
                     "queries": search_queries, "retrieval_timestamp": utc_now(),
                     "retrieved_papers": external_results},
        )

        return PriorArtAssessment(
            id=uuid4().hex,
            project_id=candidate.project_id,
            run_id=candidate.run_id,
            candidate_id=candidate.id,
            closest_paper_ids=tuple(closest_paper_ids),
            overlap_status=PriorArtStatus(payload.get("overlap_status", "potentially_novel")),
            mechanism_comparison=payload.get("mechanism_comparison", ""),
            supporting_excerpts=tuple(payload.get("supporting_excerpts", [])),
            unresolved_overlap=payload.get("unresolved_overlap", ""),
            provenance=provenance,
        )
