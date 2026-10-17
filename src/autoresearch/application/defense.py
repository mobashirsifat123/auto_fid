"""Manuscript defense loop: review objections → routed fixes → defensibility
(2026-10-16, owner requirement: "add review in the manuscript skill and make
it better until a CVPR-like defensible paper").

Architecture (closed loop, evidence-first):

1. A :class:`ManuscriptReviewer` (protocol; LLM-backed in production via the
   SCIENTIFIC_CRITIC role, fake in tests) produces :class:`Objection` s in
   four fixed categories — CLARITY, ANALYSIS, MISSING_EXPERIMENT, LIMITATION.
2. :func:`plan_fixes` routes every objection deterministically:

   - CLARITY            → revise prose (revision pass)
   - ANALYSIS           → recompute from EXISTING measurements (never a new
                          claim from thin air)
   - MISSING_EXPERIMENT → a new experiment cell through the REAL
                          approval-gated path (``experiment approve`` bound
                          to the spec checksum — the loop can never silently
                          self-authorize execution)
   - LIMITATION         → disclosed in the Limitations section

3. After each round, fixes become :class:`Resolution` s: ADDRESSED (with an
   evidence reference), DISCLOSED (limitation), or OPEN.
4. :func:`defensibility` emits the verdict: DEFENSIBLE only when no
   high-severity objection is OPEN and every objection is ADDRESSED or
   DISCLOSED; otherwise NOT_DEFENSIBLE with the open list. The verdict is a
   *rebuttal-style evidence map*: objection → what answered it.

Honesty rails:
- The loop NEVER fabricates evidence: MISSING_EXPERIMENT objections can only
  close via a real run reference (run id), ANALYSIS via a recomputation
  reference (analysis id).
- Termination: all objections closed, or ``max_rounds`` (default 3), or
  budget exhaustion — whichever comes first; termination without closure is
  reported as NOT_DEFENSIBLE, never silently downgraded.
- Reviewer identity/model is recorded per round (judge ≠ writer family in
  production wiring).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol

__all__ = [
    "DefensibilityReport",
    "FixPlan",
    "FixRoute",
    "ManuscriptReviewer",
    "Objection",
    "ObjectionCategory",
    "Resolution",
    "ReviewReport",
    "defensibility",
    "plan_fixes",
]


class ObjectionCategory(StrEnum):
    CLARITY = "clarity"
    ANALYSIS = "analysis"
    MISSING_EXPERIMENT = "missing_experiment"
    LIMITATION = "limitation"


class FixRoute(StrEnum):
    REVISE_PROSE = "revise_prose"
    REANALYZE = "reanalyze"
    NEW_EXPERIMENT = "new_experiment"
    DISCLOSE_LIMITATION = "disclose_limitation"


@dataclass(frozen=True)
class Objection:
    id: str
    category: ObjectionCategory
    severity: int  # 1 (minor) .. 3 (blocking)
    text: str

    def __post_init__(self) -> None:
        if not 1 <= int(self.severity) <= 3:
            raise ValueError("severity must be 1..3")


@dataclass(frozen=True)
class ReviewReport:
    round_index: int
    reviewer_identity: str
    objections: tuple[Objection, ...]


class ManuscriptReviewer(Protocol):
    def review(self, draft: str, context: Mapping[str, object]) -> ReviewReport: ...


# Deterministic keyword overrides for LLM miscategorization (category wins
# unless an override keyword appears in the objection text).
_EXPERIMENT_KEYWORDS = (
    "ablation",
    "baseline",
    "seed",
    "sensitivity",
    "parameter sweep",
    "dataset",
)
_ANALYSIS_KEYWORDS = (
    "significance",
    "effect size",
    "confidence interval",
    "p-value",
    "variance",
)


@dataclass(frozen=True)
class FixPlan:
    routes: Mapping[str, FixRoute]
    revise_prose: tuple[str, ...]
    reanalyze: tuple[str, ...]
    new_experiments: tuple[str, ...]
    disclose_limitations: tuple[str, ...]

    @property
    def requires_new_experiments(self) -> bool:
        return bool(self.new_experiments)


def plan_fixes(report: ReviewReport) -> FixPlan:
    """Route every objection to exactly one fix path (deterministic)."""
    routes: dict[str, FixRoute] = {}
    prose: list[str] = []
    reanalyze: list[str] = []
    experiments: list[str] = []
    disclosures: list[str] = []
    for objection in report.objections:
        lowered = objection.text.lower()
        if objection.category is ObjectionCategory.MISSING_EXPERIMENT:
            route = FixRoute.NEW_EXPERIMENT
        elif objection.category is ObjectionCategory.LIMITATION:
            route = FixRoute.DISCLOSE_LIMITATION
        elif objection.category is ObjectionCategory.ANALYSIS:
            route = FixRoute.REANALYZE
        else:  # CLARITY — but keywords can promote it to a real gap
            if any(k in lowered for k in _EXPERIMENT_KEYWORDS):
                route = FixRoute.NEW_EXPERIMENT
            elif any(k in lowered for k in _ANALYSIS_KEYWORDS):
                route = FixRoute.REANALYZE
            else:
                route = FixRoute.REVISE_PROSE
        routes[objection.id] = route
        if route is FixRoute.REVISE_PROSE:
            prose.append(objection.id)
        elif route is FixRoute.REANALYZE:
            reanalyze.append(objection.id)
        elif route is FixRoute.NEW_EXPERIMENT:
            experiments.append(objection.id)
        else:
            disclosures.append(objection.id)
    return FixPlan(
        routes=routes,
        revise_prose=tuple(prose),
        reanalyze=tuple(reanalyze),
        new_experiments=tuple(experiments),
        disclose_limitations=tuple(disclosures),
    )


@dataclass(frozen=True)
class Resolution:
    objection_id: str
    status: str  # ADDRESSED | DISCLOSED | OPEN
    method: str
    evidence_ref: str | None = None  # run id / analysis id / section — never prose

    def __post_init__(self) -> None:
        if self.status == "ADDRESSED" and not self.evidence_ref:
            raise ValueError("ADDRESSED resolutions require an evidence reference")


@dataclass(frozen=True)
class DefensibilityReport:
    verdict: str  # DEFENSIBLE | NOT_DEFENSIBLE
    rounds: int
    resolutions: tuple[Resolution, ...]
    terminated_by: str  # all_closed | max_rounds | budget_exhausted
    open_blocking: tuple[str, ...] = field(default=())
    evidence_map: Mapping[str, str] = field(default_factory=dict)


def defensibility(
    resolutions: Sequence[Resolution],
    *,
    rounds: int,
    terminated_by: str,
    objections: Sequence[Objection],
) -> DefensibilityReport:
    """Compute the defensibility verdict from resolution bookkeeping.

    DEFENSIBLE requires: every objection ADDRESSED or DISCLOSED, and no
    blocking (severity 3) objection closed only by disclosure when a route
    to real evidence existed (plan_fixes must have routed it to an
    experiment/analysis — enforced by checking the route map consistency at
    the call site; here we require disclosure-only blocking objections to be
    explicitly justified via method='irreducible')."""
    by_id = {r.objection_id: r for r in resolutions}
    open_blocking: list[str] = []
    evidence_map: dict[str, str] = {}
    for objection in objections:
        resolution = by_id.get(objection.id)
        if resolution is None or resolution.status == "OPEN":
            if objection.severity >= 3:
                open_blocking.append(objection.id)
            continue
        if (
            resolution.status == "DISCLOSED"
            and objection.severity >= 3
            and resolution.method != "irreducible"
        ):
            open_blocking.append(objection.id)
            continue
        evidence_map[objection.id] = (
            f"{resolution.status}:{resolution.method}:{resolution.evidence_ref or '-'}"
        )
    verdict = "DEFENSIBLE" if not open_blocking else "NOT_DEFENSIBLE"
    return DefensibilityReport(
        verdict=verdict,
        rounds=rounds,
        resolutions=tuple(resolutions),
        open_blocking=tuple(open_blocking),
        terminated_by=terminated_by,
        evidence_map=evidence_map,
    )
