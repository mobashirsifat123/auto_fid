"""Exemplar-grounded manuscript calibration (2026-10-16).

Owner requirement: manuscripts produced by the manuscript skill must reach at
least the structural/rigor level of the top-5 papers the review stage
selected (their PDFs are already downloaded). This module makes that
requirement measurable and honest:

- :class:`ExemplarExtractor` ranks reviewed papers (confidence desc, evidence
  claims as tie-break) and extracts **deterministic structural features**
  from their text: section coverage, table/figure reference counts, rigor
  markers (± / p-values / seed reporting), citation density.
- The SAME extractor runs on our generated manuscript, so parity is computed
  feature-for-feature — no LLM judgment involved.
- :func:`parity_check` emits a :class:`ParityReport` with per-feature gaps
  and exemplar medians, plus writer guidance (which sections/markers to add).
  Parity gaps are recorded as INCOMPLETE findings — matching exemplar
  *structure and rigor* NEVER licenses fabricating content: a missing
  ablation is closed by running one, not by writing one.

Feature extraction works on plain text from either source (parsed PDF pages
joined, or generated markdown) so both sides are measured identically.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["ExemplarExtractor", "ExemplarProfile", "ParityReport", "parity_check"]

_SECTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("abstract", re.compile(r"\babstract\b", re.IGNORECASE)),
    ("introduction", re.compile(r"\bintroduction\b", re.IGNORECASE)),
    ("related_work", re.compile(r"\brelated\s+work\b|\bbackground\b", re.IGNORECASE)),
    (
        "method",
        re.compile(r"\bmethod(s)?\b|\bapproach\b|\bmethodology\b", re.IGNORECASE),
    ),
    (
        "experiments",
        re.compile(r"\bexperiments?\b|\bresults\b|\bevaluation\b", re.IGNORECASE),
    ),
    ("ablation", re.compile(r"\bablations?\b", re.IGNORECASE)),
    ("limitations", re.compile(r"\blimitations?\b", re.IGNORECASE)),
    ("conclusion", re.compile(r"\bconclusions?\b", re.IGNORECASE)),
    ("reproducibility", re.compile(r"\breproducib\b", re.IGNORECASE)),
)

_TABLE_REF = re.compile(r"\bTable\s+(\d+)", re.IGNORECASE)
_FIGURE_REF = re.compile(r"\b(?:Figure|Fig\.)\s+(\d+)", re.IGNORECASE)
_PM_STD = re.compile(r"±|\\pm")
_P_VALUE = re.compile(r"\bp\s*[<=]\s*[\d.]+|\bp-value", re.IGNORECASE)
_SEED = re.compile(r"\bseeds?\b|\bruns?\s+with\b", re.IGNORECASE)
_BRACKET_CITE = re.compile(r"\[(\d{1,3})\]")
_ET_AL = re.compile(r"\bet\s+al\.")


@dataclass(frozen=True)
class PaperFeatures:
    paper_id: str
    sections: tuple[str, ...]
    max_table_ref: int
    max_figure_ref: int
    pm_std_count: int
    p_value_count: int
    seed_mention_count: int
    citation_estimate: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "paper_id": self.paper_id,
            "sections": list(self.sections),
            "max_table_ref": self.max_table_ref,
            "max_figure_ref": self.max_figure_ref,
            "pm_std_count": self.pm_std_count,
            "p_value_count": self.p_value_count,
            "seed_mention_count": self.seed_mention_count,
            "citation_estimate": self.citation_estimate,
        }


@dataclass(frozen=True)
class ExemplarProfile:
    top_papers: tuple[PaperFeatures, ...]
    section_frequency: Mapping[str, float]
    medians: Mapping[str, float]
    n_papers: int


class ExemplarExtractor:
    """Rank reviewed papers and extract structural features deterministically."""

    def rank(self, reviews: Sequence[Any], *, limit: int = 5) -> list[Any]:
        """Top-``limit`` reviews by confidence; ties broken by evidence-claim
        count, then paper id for determinism."""
        ranked = sorted(
            reviews,
            key=lambda r: (
                -float(r.confidence),
                -len(r.evidence_claim_ids or ()),
                str(r.paper_id),
            ),
        )
        return ranked[:limit]

    def features(self, paper_id: str, text: str) -> PaperFeatures:
        sections = tuple(
            name
            for name, pattern in _SECTION_PATTERNS
            if pattern.search(text) is not None
        )
        table_refs = [int(m) for m in _TABLE_REF.findall(text)]
        figure_refs = [int(m) for m in _FIGURE_REF.findall(text)]
        bracket_cites = [int(m) for m in _BRACKET_CITE.findall(text)]
        citation_estimate = (
            max(bracket_cites) if bracket_cites else len(_ET_AL.findall(text))
        )
        return PaperFeatures(
            paper_id=paper_id,
            sections=sections,
            max_table_ref=max(table_refs, default=0),
            max_figure_ref=max(figure_refs, default=0),
            pm_std_count=len(_PM_STD.findall(text)),
            p_value_count=len(_P_VALUE.findall(text)),
            seed_mention_count=len(_SEED.findall(text)),
            citation_estimate=citation_estimate,
        )

    def profile(self, papers: Sequence[PaperFeatures]) -> ExemplarProfile:
        if not papers:
            return ExemplarProfile(
                top_papers=(),
                section_frequency={},
                medians={},
                n_papers=0,
            )
        n = len(papers)
        section_freq: dict[str, float] = {}
        for name, _pattern in _SECTION_PATTERNS:
            section_freq[name] = sum(1 for p in papers if name in p.sections) / n

        def median(values: Sequence[int]) -> float:
            ordered = sorted(values)
            mid = len(ordered) // 2
            if len(ordered) % 2:
                return float(ordered[mid])
            return (ordered[mid - 1] + ordered[mid]) / 2.0

        medians = {
            "max_table_ref": median([p.max_table_ref for p in papers]),
            "max_figure_ref": median([p.max_figure_ref for p in papers]),
            "pm_std_count": median([p.pm_std_count for p in papers]),
            "p_value_count": median([p.p_value_count for p in papers]),
            "seed_mention_count": median([p.seed_mention_count for p in papers]),
            "citation_estimate": median([p.citation_estimate for p in papers]),
        }
        return ExemplarProfile(
            top_papers=tuple(papers),
            section_frequency=section_freq,
            medians=medians,
            n_papers=n,
        )


@dataclass(frozen=True)
class ParityReport:
    parity: bool
    gaps: tuple[Mapping[str, Any], ...] = ()
    guidance: tuple[str, ...] = ()
    manuscript_features: PaperFeatures | None = None
    profile: ExemplarProfile | None = None
    notes: tuple[str, ...] = field(default=())


def parity_check(
    manuscript_text: str,
    profile: ExemplarProfile,
    *,
    require_sections_present_in: float = 0.6,
    numeric_floor: float = 0.5,
) -> ParityReport:
    """Compare one generated manuscript against the exemplar profile.

    - Sections present in ≥ ``require_sections_present_in`` of exemplars are
      required (abstract/intro/method/experiments/conclusion effectively
      always; limitations/ablation when exemplars do them consistently).
    - Numeric features must reach ``numeric_floor`` × exemplar median.
    - Empty profile (no exemplars) → parity inconclusive (recorded, never a
      pass): the report notes the absence and returns parity=False honestly.
    """
    if profile.n_papers == 0:
        return ParityReport(
            parity=False,
            notes=(
                "no_exemplars: review stage produced no ranked papers; parity cannot be assessed",
            ),
        )

    manuscript = ExemplarExtractor().features("manuscript", manuscript_text)
    gaps: list[Mapping[str, Any]] = []
    guidance: list[str] = []

    for name, frequency in profile.section_frequency.items():
        required = frequency >= require_sections_present_in
        present = name in manuscript.sections
        if required and not present:
            gaps.append(
                {
                    "feature": f"section:{name}",
                    "exemplar_share": round(frequency, 2),
                    "manuscript": "absent",
                    "required": True,
                }
            )
            guidance.append(
                f"add the '{name.replace('_', ' ')}' section — present in "
                f"{frequency:.0%} of the top-{profile.n_papers} exemplar papers"
            )

    numeric_map = {
        "max_table_ref": manuscript.max_table_ref,
        "max_figure_ref": manuscript.max_figure_ref,
        "pm_std_count": manuscript.pm_std_count,
        "p_value_count": manuscript.p_value_count,
        "seed_mention_count": manuscript.seed_mention_count,
        "citation_estimate": manuscript.citation_estimate,
    }
    for name, value in numeric_map.items():
        median = float(profile.medians.get(name, 0.0))
        floor = median * numeric_floor
        if value < floor:
            gaps.append(
                {
                    "feature": name,
                    "exemplar_median": median,
                    "required_floor": round(floor, 2),
                    "manuscript": value,
                    "required": True,
                }
            )
            guidance.append(
                f"increase '{name}' (manuscript {value} vs exemplar median "
                f"{median:.0f}; floor {floor:.0f}) — by producing the underlying "
                "evidence, never by decorative text"
            )

    return ParityReport(
        parity=not gaps,
        gaps=tuple(gaps),
        guidance=tuple(guidance),
        manuscript_features=manuscript,
        profile=profile,
    )
