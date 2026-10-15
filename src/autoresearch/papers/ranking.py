"""Transparent pre-review candidate ranking."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone

from .deduplication import DeduplicatedCandidate


@dataclass(frozen=True)
class CandidateScore:
    relevance: float
    method_overlap: float
    recency: float
    influence: float
    source_agreement: float
    open_access: float
    total: float

    def components(self) -> dict[str, float]:
        return {"relevance": self.relevance, "method_overlap": self.method_overlap, "recency": self.recency, "influence": self.influence, "source_agreement": self.source_agreement, "open_access": self.open_access, "total": self.total}


def rank_candidates(candidates: list[DeduplicatedCandidate], *, seed_text: str, seed_methods: tuple[str, ...], current_year: int | None = None) -> list[tuple[DeduplicatedCandidate, CandidateScore]]:
    year = current_year or datetime.now(timezone.utc).year
    maximum_citations = max((item.paper.citation_count for item in candidates), default=1) or 1
    ranked: list[tuple[DeduplicatedCandidate, CandidateScore]] = []
    for candidate in candidates:
        paper = candidate.paper
        relevance = _jaccard(seed_text, f"{paper.title} {paper.abstract}")
        method_overlap = _jaccard(" ".join(seed_methods), f"{paper.title} {paper.abstract}") if seed_methods else relevance
        age = max(0, year - (paper.year or year - 10))
        recency = math.exp(-age / 4.0)
        influence = math.log1p(paper.citation_count) / math.log1p(maximum_citations) if maximum_citations else 0.0
        source_agreement = min(1.0, len({item.source for item in candidate.provenance}) / 3.0)
        open_access = 1.0 if paper.is_open_access or paper.open_access_pdf_url else 0.0
        total = 0.40 * relevance + 0.20 * method_overlap + 0.18 * recency + 0.10 * influence + 0.07 * source_agreement + 0.05 * open_access
        score = CandidateScore(*[round(value, 6) for value in (relevance, method_overlap, recency, influence, source_agreement, open_access, total)])
        ranked.append((candidate, score))
    return sorted(ranked, key=lambda item: (-item[1].total, -(item[0].paper.year or 0), item[0].paper.title.lower()))


def _jaccard(left: str, right: str) -> float:
    first = set(re.findall(r"[a-z0-9]+", left.lower()))
    second = set(re.findall(r"[a-z0-9]+", right.lower()))
    if not first or not second:
        return 0.0
    return len(first & second) / len(first | second)
