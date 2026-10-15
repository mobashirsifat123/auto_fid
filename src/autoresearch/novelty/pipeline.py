"""Purpose: Ties generation, prior-art checking, critique, revision, and selection into an end-to-end workflow."""

from __future__ import annotations

import logging

from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, NoveltyClaimLedger, ResearchGap, NoveltyPortfolio
from autoresearch.novelty.candidate_generator import LLMCandidateGenerator
from autoresearch.novelty.novelty_critic import ScientificCritic
from autoresearch.novelty.prior_art_checker import EvidenceBackedPriorArtChecker
from autoresearch.novelty.selection import PortfolioSelector
from autoresearch.storage.repository import ResearchRepository
from autoresearch.review.evidence_context import build_evidence_context

logger = logging.getLogger(__name__)

# Maximum revision rounds for candidates with partial overlap
MAX_REVISION_ROUNDS = 2


import asyncio


class NoveltyPipeline:
    def __init__(
        self,
        repository: ResearchRepository,
        generator: LLMCandidateGenerator,
        prior_art_checker: EvidenceBackedPriorArtChecker,
        critic: ScientificCritic,
        selector: PortfolioSelector,
    ) -> None:
        self.repository = repository
        self.generator = generator
        self.prior_art_checker = prior_art_checker
        self.critic = critic
        self.selector = selector

    async def generate_portfolio(self, gap: ResearchGap) -> NoveltyPortfolio:
        logger.info(f"Starting novelty generation for gap {gap.id}")

        # Literature grounding (HKUDS mechanism, adapted): the generator sees
        # the gap's verified claims, the papers' methods, and counter-evidence
        # instead of only the two-sentence gap statement. A sparse corpus
        # degrades to the previous behaviour.
        evidence_context = build_evidence_context(
            self.repository,
            project_id=gap.project_id,
            supporting_paper_ids=gap.supporting_paper_ids,
            supporting_claim_ids=gap.supporting_claim_ids,
            counter_claim_ids=gap.counter_evidence_claim_ids,
        )
        logger.info(
            "Evidence context for gap %s: %d papers, %d claims, %d counter-claims.",
            gap.id,
            len(evidence_context.get("papers", [])),
            len(evidence_context.get("claims", [])),
            len(evidence_context.get("counter_evidence_claims", [])),
        )

        # 1. Generate Candidates
        raw_candidates = await self.generator.generate(gap, count=5, evidence_context=evidence_context)
        for candidate in raw_candidates:
            self.repository.save_novelty_candidate(candidate)

        # N3: Semantic deduplication before prior-art check
        candidates = _deduplicate_candidates(list(raw_candidates))
        logger.info(f"Generated {len(raw_candidates)} candidates, {len(candidates)} unique after deduplication.")

        valid_candidates = []
        critiques = []

        # N4: Concurrent Prior-Art + Critique Processing
        async def process_candidate(candidate):
            assessment = await self.prior_art_checker.check(candidate)
            self.repository.save_prior_art_assessment(assessment)
            if assessment.overlap_status.value == "known_method":
                logger.info(f"Candidate {candidate.id} rejected due to overlap: {assessment.overlap_status}")
                return None

            critique = await self.critic.evaluate(candidate)
            self.repository.save_novelty_critique(critique)

            # Iterative refinement: up to MAX_REVISION_ROUNDS for partial overlap
            for revision_round in range(1, MAX_REVISION_ROUNDS + 1):
                if assessment.overlap_status.value != "partial_overlap":
                    break

                logger.info(f"Revision round {revision_round}/{MAX_REVISION_ROUNDS} for candidate {candidate.id}")
                original = candidate
                revised, revision = await self.generator.revise(
                    original, assessment, critique, revision_count=revision_round,
                    evidence_context=evidence_context,
                )
                self.repository.save_novelty_candidate(revised)
                self.repository.save_candidate_revision(revision)
                reassessment = await self.prior_art_checker.check(revised)
                self.repository.save_prior_art_assessment(reassessment)
                candidate = revised
                assessment = reassessment
                critique = await self.critic.evaluate(revised)
                self.repository.save_novelty_critique(critique)

                if assessment.overlap_status.value != "partial_overlap":
                    break

            # N7: Enhanced Scientific Gates with Composite Score
            gate_score = (
                0.25 * critique.falsifiability_score
                + 0.20 * critique.experimental_discriminability_score
                + 0.20 * critique.feasibility_score
                + 0.15 * critique.causal_plausibility_score
                + 0.10 * critique.novelty_score
                + 0.10 * critique.significance_score
            )
            hard_gates = (
                bool(candidate.difference_from_existing.strip()),
                bool(candidate.falsification_condition.strip()),
                bool(candidate.minimum_validating_experiment.strip()),
                # Corpus-bounded assessments may remain visible to a human as
                # candidates, but their ledger retains EXPERT_REVIEW_REQUIRED;
                # they never become a broad POTENTIALLY_NOVEL claim.
                assessment.overlap_status.value not in ("unsupported", "known_method"),
                critique.falsifiability_score > 0,
                critique.experimental_discriminability_score > 0,
                critique.feasibility_score > 0,
                gate_score >= 0.20,
            )
            if not all(hard_gates):
                logger.info("Candidate %s did not pass scientific gates (gate_score=%.3f)", candidate.id, gate_score)
                return None

            return (candidate, critique, assessment)

        results = await asyncio.gather(*[process_candidate(c) for c in candidates])
        for res in results:
            if res is not None:
                candidate, critique, assessment = res
                valid_candidates.append(candidate)
                critiques.append(critique)
                safe_wording = "Appears distinct under the current retrieved evidence; requires expert verification."
                ledger = NoveltyClaimLedger(
                    id=uuid4().hex, project_id=candidate.project_id, run_id=candidate.run_id,
                    candidate_id=candidate.id, contribution_claim=candidate.difference_from_existing,
                    closest_methods=assessment.closest_paper_ids, overlap_status=assessment.overlap_status,
                    supporting_evidence=assessment.supporting_excerpts,
                    unresolved_uncertainty=assessment.unresolved_overlap,
                    safe_wording=safe_wording,
                    prohibited_wording=("proven novel", "definitively novel", "first-ever"),
                    review_status="expert_review_required" if assessment.overlap_status.value == "expert_review_required" else "retrieval_assessed",
                    provenance=ArtifactProvenance.create(
                        project_id=candidate.project_id, run_id=candidate.run_id,
                        source_evidence=[assessment.id, critique.id],
                        provider="deterministic", model="claim-policy",
                        prompt_version="claim-ledger-v1",
                        content={"safe_wording": safe_wording, "status": assessment.overlap_status.value},
                    ),
                )
                self.repository.save_novelty_claim_ledger(ledger)

        if not valid_candidates:
            self._save_grounding_record(gap, evidence_context)
            raise RuntimeError("Novelty pipeline produced no candidate that passed the scientific gates.")

        # 3. Portfolio Selection (diversity-aware)
        portfolio = self.selector.select(gap.project_id, gap.run_id, valid_candidates, critiques)
        self.repository.save_novelty_portfolio(portfolio)

        # Written after gating and revision so revised candidates — the ones
        # that can actually be selected and planned — carry linkage too.
        self._save_grounding_record(gap, evidence_context)

        logger.info("Portfolio selected successfully.")
        return portfolio

    def _save_grounding_record(self, gap: ResearchGap, evidence_context: dict) -> None:
        """Persist which supplied evidence every candidate of this run cited.

        Provenance stores only a content checksum, so this checksummed JSON
        artifact keeps the literature linkage inspectable after the run. It
        covers every saved candidate of the run (generated, revised, and
        rejected alike) by reading current rows, so revisions are included.
        Bookkeeping must never fail the run: an OSError is logged and the
        portfolio result still returns.
        """
        try:
            import json as _json

            from autoresearch.domain.models import ArtifactReference, content_checksum
            from autoresearch.novelty.candidate_generator import PROMPT_VERSION, cited_paper_ids

            candidate_rows = self.repository.rows(
                "SELECT id,strategy,difference_from_existing FROM novelty_candidates WHERE run_id=? ORDER BY rowid",
                (gap.run_id,),
            )
            record = {
                "gap_id": gap.id,
                "prompt_version": PROMPT_VERSION,
                "supplied": {
                    "papers": [str(p.get("paper_id")) for p in evidence_context.get("papers", [])],
                    "claims": [str(c.get("claim_id")) for c in evidence_context.get("claims", [])],
                    "counter_evidence_claims": [str(c.get("claim_id")) for c in evidence_context.get("counter_evidence_claims", [])],
                },
                "candidates": [
                    {
                        "id": str(row["id"]),
                        "strategy": str(row["strategy"]),
                        "evidence_linked_paper_ids": cited_paper_ids(str(row["difference_from_existing"]), evidence_context),
                    }
                    for row in candidate_rows
                ],
            }
            text = _json.dumps(record, indent=2, ensure_ascii=False, default=str)
            root = self.repository.artifact_root / gap.project_id / gap.run_id
            root.mkdir(parents=True, exist_ok=True)
            path = root / "novelty-grounding.json"
            path.write_text(text, encoding="utf-8")
            self.repository.save_artifact(ArtifactReference(
                content_checksum(f"novelty_grounding|{gap.run_id}")[:32],
                project_id=gap.project_id, run_id=gap.run_id,
                artifact_type="novelty_grounding_json", path=str(path),
                source_evidence=gap.supporting_claim_ids,
                provider="deterministic", model="record-writer",
                prompt_version=PROMPT_VERSION, checksum=content_checksum(text),
            ))
        except OSError:
            logger.exception("Failed to persist the novelty grounding record for run %s; continuing.", gap.run_id)


def _deduplicate_candidates(candidates: list, threshold: float = 0.65) -> list:
    """Drop candidates whose method+mechanism word overlap exceeds threshold, preserving strategy diversity (N3)."""
    if len(candidates) <= 3:
        return candidates
    unique = [candidates[0]]
    for idx, candidate in enumerate(candidates[1:], start=1):
        words_new = set(f"{candidate.method} {candidate.causal_mechanism}".lower().split())
        is_duplicate = False
        for u in unique:
            if candidate.strategy == u.strategy:
                words_u = set(f"{u.method} {u.causal_mechanism}".lower().split())
                if not words_new or not words_u:
                    continue
                ratio = len(words_new & words_u) / max(len(words_new | words_u), 1)
                if ratio >= threshold:
                    is_duplicate = True
                    break
        if not is_duplicate or (len(unique) + (len(candidates) - idx) <= 3):
            unique.append(candidate)
    return unique
