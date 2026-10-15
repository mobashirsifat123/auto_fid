"""Purpose: Selects 3 candidates (Low-Risk, Balanced, High-Risk) from the evaluated candidates."""

from __future__ import annotations

from uuid import uuid4

from autoresearch.domain.models import (
    ArtifactProvenance,
    CandidateStatus,
    NoveltyCandidate,
    NoveltyCritique,
    NoveltyPortfolio,
)
from autoresearch.storage.repository import ResearchRepository


import math


class PortfolioSelector:
    def __init__(self, repository: ResearchRepository) -> None:
        self.repository = repository

    def select(self, project_id: str, run_id: str, candidates: list[NoveltyCandidate], critiques: list[NoveltyCritique]) -> NoveltyPortfolio:
        if not candidates:
            raise ValueError("Portfolio selection requires at least one candidate.")

        # Match candidates to their critiques
        scored_candidates = []
        for candidate in candidates:
            critique = next((c for c in critiques if c.candidate_id == candidate.id), None)
            if critique:
                scored_candidates.append((candidate, critique))

        if not scored_candidates:
            raise ValueError("No candidate passed the scientific gates.")

        # Diversity-aware selection (N8): IDF-weighted methodological diversity
        all_cand_objs = [sc[0] for sc in scored_candidates]
        def method_diversity(c1: NoveltyCandidate, c2: NoveltyCandidate) -> float:
            words1 = set(c1.method.lower().split()) | set(c1.causal_mechanism.lower().split())
            words2 = set(c2.method.lower().split()) | set(c2.causal_mechanism.lower().split())
            if not words1 or not words2:
                return 1.0

            all_docs = [set(c.method.lower().split()) | set(c.causal_mechanism.lower().split()) for c in all_cand_objs]
            doc_count = len(all_docs)
            doc_freq = {}
            for doc in all_docs:
                for w in doc:
                    doc_freq[w] = doc_freq.get(w, 0) + 1
            idf = {w: math.log((doc_count + 1.0) / (df + 1.0)) + 1.0 for w, df in doc_freq.items()}

            inter_weight = sum(idf.get(w, 1.0) for w in (words1 & words2))
            union_weight = sum(idf.get(w, 1.0) for w in (words1 | words2))
            return round(1.0 - (inter_weight / max(union_weight, 1e-9)), 4)

        # Sort by expected compute efficiency (proxy for low risk/fast execution)
        low_risk = sorted(scored_candidates, key=lambda x: x[1].compute_efficiency_score + x[1].feasibility_score, reverse=True)[0]

        # Balance diversity with the best balanced candidate
        remaining_for_balanced = [c for c in scored_candidates if c[0].id != low_risk[0].id]
        if remaining_for_balanced:
            # Score: publication_potential + feasibility + diversity_from_low_risk
            balanced = max(
                remaining_for_balanced,
                key=lambda x: (
                    x[1].publication_potential_score + x[1].feasibility_score
                    + method_diversity(low_risk[0], x[0]) * 0.3  # diversity bonus
                ),
            )
        else:
            balanced = None

        # High risk: novelty + significance - feasibility + diversity from selected
        selected_ids = {c[0].id for c in [low_risk, balanced] if c is not None}
        remaining_for_high = [c for c in scored_candidates if c[0].id not in selected_ids]
        if remaining_for_high:
            # Compute average diversity from already-selected candidates
            def avg_diversity(c: NoveltyCandidate) -> float:
                selected = [s[0] for s in [low_risk, balanced] if s is not None and s[0].id != c.id]
                if not selected:
                    return 0.0
                return sum(method_diversity(c, s) for s in selected) / len(selected)

            high_risk = max(
                remaining_for_high,
                key=lambda x: (
                    x[1].novelty_score + x[1].significance_score - x[1].feasibility_score
                    + avg_diversity(x[0]) * 0.5  # strong diversity bonus
                ),
            )
        else:
            high_risk = None

        provenance = ArtifactProvenance.create(
            project_id=project_id, run_id=run_id,
            source_evidence=[c[0].id for c in scored_candidates],
            provider="deterministic", model="heuristic",
            prompt_version="portfolio-selector-v2",
            content={"strategy": "diversity_aware_risk_balancing"},
        )

        # Build selection reasons
        selection_reasons = {}
        if low_risk:
            selection_reasons["low_risk"] = "Highest feasibility and compute efficiency."
        if balanced:
            selection_reasons["balanced"] = (
                f"Strong trade-off between publication potential, feasibility, "
                f"and methodological diversity from low-risk candidate "
                f"(diversity={method_diversity(low_risk[0], balanced[0]):.2f})."
            )
        if high_risk:
            selection_reasons["high_risk"] = (
                f"High information gain with explicitly higher risk, "
                f"maximizing methodological diversity from other selected candidates."
            )

        portfolio = NoveltyPortfolio(
            id=uuid4().hex, project_id=project_id, run_id=run_id,
            low_risk_candidate_id=low_risk[0].id,
            balanced_candidate_id=balanced[0].id if balanced else None,
            high_risk_candidate_id=high_risk[0].id if high_risk else None,
            selection_reasons=selection_reasons,
            provenance=provenance,
        )

        for candidate, _ in tuple(item for item in (low_risk, balanced, high_risk) if item is not None):
            self.repository.update_novelty_candidate_status(candidate.id, CandidateStatus.PORTFOLIO_SELECTED)

        return portfolio
