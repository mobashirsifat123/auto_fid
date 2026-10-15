"""Purpose: Ranks papers in a database relative to the project's core research question."""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Sequence
from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, PaperRecord, PaperReview, PriorityScore, SeedPaper
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.errors import ProviderError
from autoresearch.providers.router import ModelRouter

RANKING_VERSION = "priority-ranking-v1"
WEIGHTS = {
    "semantic_similarity": 0.35,
    "methodological_similarity": 0.20,
    "research_question_similarity": 0.15,
    "recency": 0.15,
    "influence": 0.10,
    "evidence_quality": 0.05,
}


class ReviewRanker:
    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    async def rank(
        self,
        *,
        seed: SeedPaper,
        papers: Sequence[PaperRecord],
        reviews: Sequence[PaperReview],
        run_id: str,
        embedded_vectors: Sequence[Sequence[float]] | None = None,
        embedding_sink: Callable[[tuple[tuple[float, ...], ...]], None] | None = None,
    ) -> list[PriorityScore]:
        review_by_paper = {review.paper_id: review for review in reviews}
        usable = [paper for paper in papers if paper.id in review_by_paper]
        if not usable:
            return []
        semantic_texts = [f"{seed.title} {seed.topic} {seed.text[:5000]}"] + [f"{paper.title} {paper.abstract} {review_by_paper[paper.id].problem}" for paper in usable]
        method_texts = [" ".join(seed.methods)] + [review_by_paper[paper.id].method for paper in usable]
        question_texts = [seed.research_question] + [review_by_paper[paper.id].research_question for paper in usable]
        # Embeddings are delegatable: when no embeddings provider is configured
        # (or cached vectors were never produced), ranking falls back to
        # deterministic lexical similarity so the review pipeline still works
        # on text-only providers.
        lexical = embedded_vectors is None
        if embedded_vectors is None:
            try:
                embedded = await self.router.embed([*semantic_texts, *method_texts, *question_texts], project_id=seed.project_id, run_id=run_id, operation="review_priority_embeddings")
                vectors = embedded.vectors
                lexical = False
                if embedding_sink:
                    embedding_sink(vectors)
            except ProviderError:
                vectors = ()
        else:
            vectors = tuple(tuple(float(value) for value in vector) for vector in embedded_vectors)
        count = len(usable) + 1
        if vectors:
            if len(vectors) != count * 3:
                raise ValueError(f"Expected {count * 3} cached ranking vectors, received {len(vectors)}.")
            semantic = vectors[:count]
            methods = vectors[count:count * 2]
            questions = vectors[count * 2:count * 3]
        else:
            semantic = methods = questions = ()
        maximum_citations = max((paper.citation_count for paper in usable), default=1) or 1
        current_year = datetime.now(timezone.utc).year
        if lexical:
            provider, model = "deterministic", "lexical-ranking"
        else:
            provider, model = self.router.identity(ModelRole.EMBEDDINGS)
        years = [p.year for p in usable if p.year]
        year_spread = (max(years) - min(years)) if len(years) >= 2 else 0
        active_weights = _adaptive_weights(len(usable), year_spread)

        scores: list[PriorityScore] = []
        for index, paper in enumerate(usable, start=1):
            review = review_by_paper[paper.id]
            if lexical:
                semantic_score = _lexical_similarity(semantic_texts[0], semantic_texts[index])
                method_score = _lexical_similarity(method_texts[0], method_texts[index])
                question_score = _lexical_similarity(question_texts[0], question_texts[index])
            else:
                semantic_score = _unit_cosine(semantic[0], semantic[index])
                method_score = _unit_cosine(methods[0], methods[index])
                question_score = _unit_cosine(questions[0], questions[index])
            components = {
                "semantic_similarity": semantic_score,
                "methodological_similarity": method_score,
                "research_question_similarity": question_score,
                "recency": math.exp(-max(0, current_year - (paper.year or current_year - 10)) / 4.0),
                "influence": math.log1p(paper.citation_count) / math.log1p(maximum_citations),
                "evidence_quality": min(1.0, review.confidence * (0.6 + min(len(review.evidence_claim_ids), 4) * 0.1)),
            }
            components = {key: round(max(0.0, min(1.0, value)), 6) for key, value in components.items()}
            total = round(sum(components[key] * active_weights[key] for key in active_weights), 6)
            explanation = {"weights": active_weights, "components": components, "formula": "weighted_sum", "similarity_basis": "lexical" if lexical else "embeddings"}
            provenance = ArtifactProvenance.create(project_id=seed.project_id, run_id=run_id, source_evidence=review.evidence_claim_ids, provider=provider, model=model, prompt_version=f"{RANKING_VERSION}{'-lexical' if lexical else ''}", content={**components, "total": total})
            scores.append(PriorityScore(uuid4().hex, seed.project_id, run_id, paper.id, components["semantic_similarity"], components["methodological_similarity"], components["research_question_similarity"], components["recency"], components["influence"], components["evidence_quality"], total, explanation, provenance))
        return sorted(scores, key=lambda score: (-score.total, score.paper_id))


def _adaptive_weights(paper_count: int, max_year_spread: int) -> dict[str, float]:
    """Shift weight emphasis dynamically based on corpus size and year spread (R5)."""
    base = dict(WEIGHTS)
    if paper_count <= 5:
        base["recency"] = 0.08
        base["influence"] = 0.05
        base["evidence_quality"] = 0.12
        base["semantic_similarity"] = 0.40
    elif paper_count >= 30:
        base["semantic_similarity"] = 0.25
        base["methodological_similarity"] = 0.25
        base["research_question_similarity"] = 0.20
    if max_year_spread <= 2:
        base["recency"] = 0.05
        base["evidence_quality"] += 0.10
    total = sum(base.values())
    return {k: round(v / total, 6) for k, v in base.items()}


def _unit_cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return (max(-1.0, min(1.0, dot / (left_norm * right_norm))) + 1.0) / 2.0


_STOPWORDS = frozenset(
    """a an and are as at be but by for from has have in is it its of on or that the this to was were
    will with over under between among into within across through using used based approach methods results
    study our their we they you your us them""".split()
)


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(
        token
        for token in re.findall(r"[a-z0-9][a-z0-9'_+\-]*", text.lower())
        if token not in _STOPWORDS and len(token) > 1
    )


def _lexical_similarity(left: str, right: str) -> float:
    """Deterministic word-overlap cosine used when no embeddings provider is configured.

    Uses square-root term frequencies so content words dominate but function-word
    noise stays bounded. Deterministic for a fixed pair of inputs.
    """
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    left_counts: dict[str, int] = {}
    for token in left_tokens:
        left_counts[token] = left_counts.get(token, 0) + 1
    right_counts: dict[str, int] = {}
    for token in right_tokens:
        right_counts[token] = right_counts.get(token, 0) + 1
    vocabulary = set(left_counts) | set(right_counts)
    left_vector = [math.sqrt(left_counts.get(token, 0)) for token in vocabulary]
    right_vector = [math.sqrt(right_counts.get(token, 0)) for token in vocabulary]
    return _unit_cosine(left_vector, right_vector)
