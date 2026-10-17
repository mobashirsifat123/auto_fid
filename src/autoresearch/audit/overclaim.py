"""Purpose: Overclaim detection for the trust-audit harness (DESIGN.md §5.4).

Two passes:

1. **Deterministic pattern pass (the primary score):** a frozen substring
   pattern set — a guarded duplicate of
   ``application/manuscript.py::_OVERCLAIM_PATTERNS`` (a unit test asserts
   tuple equality so the two cannot drift silently; single-source refactor
   deferred per DESIGN §5.4) — plus a small frozen set of
   comparative-superlative regexes. Every hit records span + pattern id.
   The ``SIGNIFICANTLY_OUTPERFORMS`` regex only fires when the artifact set
   carries no statistical evidence (injected flag; default False so the
   pass stays conservative).
2. **Optional LLM entailment second pass (advisory-only, never sole
   basis):** an injected ``EntailmentJudge`` callable — the runner wires
   ``ModelRouter.structured_output`` under ``ModelRole.SCIENTIFIC_CRITIC``
   later; tests inject a stub. Its results are recorded with
   ``advisory_only: true`` / ``sole_basis: false`` and are excluded from
   the primary overclaim rate. A judge error is recorded as an error and
   never kills the audit.

Frozen scope rules: the rate denominator is the sentence set of
results/discussion/abstract sections. Sentence segmentation is naive and
deterministic (``[.!?]`` boundaries; markdown headings and table rows are
excluded); hits outside the scope sections are recorded with
``in_scope: false`` and excluded from both numerator and denominator. The
rate counts hit *sentences*, so it can never exceed 1.

This module defines its own lightweight ``TextSection`` input dataclass
mirroring the documented contract shape so it stays importable while
``audit/contract.py`` is under construction.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

OVERCLAIM_METRIC_VERSION = "audit-overclaim-v1"
OVERCLAIM_PATTERN_VERSION = "audit-overclaim-patterns-v1"

# Guarded duplicate of application/manuscript.py::_OVERCLAIM_PATTERNS.
# tests/test_audit_overclaim.py asserts equality with the manuscript module.
_OVERCLAIM_PATTERNS: tuple[str, ...] = (
    "state-of-the-art",
    "sota",
    "first ever",
    "first-ever",
    "groundbreaking",
    "unprecedented",
    "outperforms all",
    "best model",
    "novel state",
)


@dataclass(frozen=True)
class ComparativePattern:
    """A frozen comparative-superlative regex addition (DESIGN §5.4.1)."""

    pattern_id: str
    regex: str
    requires_statistical_evidence: bool = False


_COMPARATIVE_PATTERNS: tuple[ComparativePattern, ...] = (
    ComparativePattern(
        "SIGNIFICANTLY_OUTPERFORMS",
        r"\bsignificantly\s+(?:outperforms?|beats?|surpasses?)\b",
        requires_statistical_evidence=True,
    ),
    ComparativePattern(
        "SUPERLATIVE_MODEL",
        r"\b(?:best|strongest|superior|dominant)\s+(?:model|method|approach|system|baseline)\b",
    ),
    ComparativePattern(
        "UNMATCHED_RESULTS",
        r"\b(?:unmatched|unrivaled|incomparable)\s+(?:performance|results?|accuracy)\b",
    ),
)

_COMPARATIVE_COMPILED: dict[str, re.Pattern[str]] = {
    pattern.pattern_id: re.compile(pattern.regex, re.IGNORECASE)
    for pattern in _COMPARATIVE_PATTERNS
}

_SCOPE_SECTION_TOKENS = ("abstract", "results", "discussion")
# Sentence boundary: sentence-ending punctuation followed by inline
# whitespace, or a newline. A "." between digits ("0.9") is NOT a boundary.
_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])[ \t]+|\n+")
_COMPARATIVE_SENTENCE_RE = re.compile(
    r"\b(?:outperform\w*|improve\w*|beat\w*|surpass\w*|exceed\w*|reduce\w*|"
    r"better|worse|best|worst|superior|inferior)\b",
    re.IGNORECASE,
)


class EntailmentVerdict(StrEnum):
    ENTAILED = "entails"
    NOT_ENTAILED = "not_entailed"
    UNVERIFIABLE = "unverifiable"


@dataclass(frozen=True)
class TextSection:
    """One named paper-text section (contract adapters map onto this)."""

    name: str
    text: str


@dataclass(frozen=True)
class EntailmentResult:
    """Verdict of the optional LLM entailment pass (advisory only)."""

    verdict: EntailmentVerdict
    evidence: tuple[str, ...] = ()
    model: str | None = None


class EntailmentJudge(Protocol):
    """LLM entailment second pass (DESIGN §5.4.2). The runner wires this to
    ``ModelRouter.structured_output`` under ``ModelRole.SCIENTIFIC_CRITIC``;
    tests inject a deterministic stub. Results are flagged advisory-only
    and never the sole basis for a score."""

    def __call__(
        self, claim_text: str, stated_results: Sequence[float]
    ) -> EntailmentResult: ...


@dataclass(frozen=True)
class Sentence:
    section: str
    text: str
    start: int
    end: int
    in_scope: bool


@dataclass(frozen=True)
class OverclaimHit:
    pattern_id: str
    pattern_kind: str  # substring | regex
    section: str
    start: int
    end: int
    matched_text: str
    in_scope: bool
    sentence: str | None = None


@dataclass(frozen=True)
class EntailmentFlag:
    section: str
    sentence: str
    verdict: EntailmentVerdict | None
    evidence: tuple[str, ...]
    model: str | None
    error: str | None = None


@dataclass(frozen=True)
class OverclaimResult:
    status: str  # RUN_COMPLETE | NOT_RUN
    not_run_reason: str | None
    llm_assisted: bool
    counts: Mapping[str, int]
    rates: Mapping[str, float | None]
    coverage: Mapping[str, Any]
    details: tuple[Mapping[str, Any], ...]

    def to_section(self) -> dict[str, Any]:
        """metricSection-shaped dict (DESIGN §8.1 $defs/metricSection)."""
        return {
            "status": self.status,
            "not_run_reason": self.not_run_reason,
            "llm_assisted": self.llm_assisted,
            "counts": dict(self.counts),
            "rates": dict(self.rates),
            "coverage": dict(self.coverage),
            "details": [dict(detail) for detail in self.details],
        }


def _not_run_result(reason: str) -> OverclaimResult:
    return OverclaimResult(
        status="NOT_RUN",
        not_run_reason=reason,
        llm_assisted=False,
        counts={},
        rates={"overclaim_pattern_rate": None, "entailment_flag_rate": None},
        coverage={"items_examined": 0, "items_skipped": 0, "skip_reasons": {}},
        details=(),
    )


def _section_in_scope(name: str) -> bool:
    lowered = name.lower()
    return any(token in lowered for token in _SCOPE_SECTION_TOKENS)


def _append_sentence(
    chunk: str, start: int, end: int, out: list[tuple[str, int, int]]
) -> None:
    stripped = chunk.strip()
    if not stripped:
        return
    if stripped.startswith(("#", "|")):  # markdown headings / table rows
        return
    lead = len(chunk) - len(chunk.lstrip())
    sentence_start = start + lead
    out.append((stripped, sentence_start, sentence_start + len(stripped)))


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    sentences: list[tuple[str, int, int]] = []
    start = 0
    for match in _SENTENCE_BOUNDARY_RE.finditer(text):
        _append_sentence(text[start : match.start()], start, match.start(), sentences)
        start = match.end()
    _append_sentence(text[start:], start, len(text), sentences)
    return sentences


def _scan_hits(
    section: TextSection,
    section_sentences: Sequence[Sentence],
    *,
    has_statistical_evidence: bool,
) -> list[OverclaimHit]:
    in_scope = _section_in_scope(section.name)
    lowered = section.text.lower()
    hits: list[OverclaimHit] = []
    for pattern in _OVERCLAIM_PATTERNS:
        index = lowered.find(pattern)
        while index != -1:
            hit = OverclaimHit(
                pattern_id=pattern,
                pattern_kind="substring",
                section=section.name,
                start=index,
                end=index + len(pattern),
                matched_text=section.text[index : index + len(pattern)],
                in_scope=in_scope,
            )
            hits.append(_with_sentence(hit, section_sentences))
            index = lowered.find(pattern, index + len(pattern))
    for comparative in _COMPARATIVE_PATTERNS:
        if comparative.requires_statistical_evidence and has_statistical_evidence:
            continue
        for match in _COMPARATIVE_COMPILED[comparative.pattern_id].finditer(
            section.text
        ):
            hit = OverclaimHit(
                pattern_id=comparative.pattern_id,
                pattern_kind="regex",
                section=section.name,
                start=match.start(),
                end=match.end(),
                matched_text=match.group(0),
                in_scope=in_scope,
            )
            hits.append(_with_sentence(hit, section_sentences))
    return hits


def _with_sentence(
    hit: OverclaimHit, section_sentences: Sequence[Sentence]
) -> OverclaimHit:
    for sentence in section_sentences:
        if sentence.start <= hit.start < sentence.end:
            return OverclaimHit(
                pattern_id=hit.pattern_id,
                pattern_kind=hit.pattern_kind,
                section=hit.section,
                start=hit.start,
                end=hit.end,
                matched_text=hit.matched_text,
                in_scope=hit.in_scope,
                sentence=sentence.text,
            )
    return hit


class OverclaimAuditor:
    """Deterministic overclaim pattern pass + optional advisory entailment
    pass (DESIGN §5.4)."""

    def __init__(self, *, has_statistical_evidence: bool = False) -> None:
        self._has_statistical_evidence = has_statistical_evidence

    def audit(
        self,
        sections: Sequence[TextSection],
        *,
        judge: EntailmentJudge | None = None,
        stated_results: Sequence[float] = (),
    ) -> OverclaimResult:
        if not any(section.text.strip() for section in sections):
            return _not_run_result("no_paper_text")

        sentences: list[Sentence] = []
        hits: list[OverclaimHit] = []
        in_scope_sentences = 0
        out_of_scope_sentences = 0
        hit_sentences_in_scope = 0
        for section in sections:
            in_scope = _section_in_scope(section.name)
            section_sentences = [
                Sentence(section.name, text, start, end, in_scope)
                for text, start, end in _split_sentences(section.text)
            ]
            sentences.extend(section_sentences)
            if in_scope:
                in_scope_sentences += len(section_sentences)
            else:
                out_of_scope_sentences += len(section_sentences)
            section_hits = _scan_hits(
                section,
                section_sentences,
                has_statistical_evidence=self._has_statistical_evidence,
            )
            hits.extend(section_hits)
            if in_scope:
                marked: set[int] = set()
                for hit in section_hits:
                    for index, sentence in enumerate(section_sentences):
                        if sentence.start <= hit.start < sentence.end:
                            marked.add(index)
                            break
                hit_sentences_in_scope += len(marked)

        entailment_flags: list[EntailmentFlag] = []
        entailment_calls = 0
        entailment_flagged = 0
        entailment_unverifiable = 0
        entailment_errors = 0
        entailment_entailed = 0
        if judge is not None:
            for sentence in sentences:
                if not sentence.in_scope:
                    continue
                if _COMPARATIVE_SENTENCE_RE.search(sentence.text) is None:
                    continue
                entailment_calls += 1
                try:
                    result = judge(sentence.text, list(stated_results))
                except Exception as exc:  # noqa: BLE001 — advisory pass must never kill the audit
                    entailment_errors += 1
                    entailment_flags.append(
                        EntailmentFlag(
                            section=sentence.section,
                            sentence=sentence.text,
                            verdict=None,
                            evidence=(),
                            model=None,
                            error=type(exc).__name__,
                        )
                    )
                    continue
                if result.verdict is EntailmentVerdict.NOT_ENTAILED:
                    entailment_flagged += 1
                elif result.verdict is EntailmentVerdict.UNVERIFIABLE:
                    entailment_unverifiable += 1
                else:
                    entailment_entailed += 1
                entailment_flags.append(
                    EntailmentFlag(
                        section=sentence.section,
                        sentence=sentence.text,
                        verdict=result.verdict,
                        evidence=tuple(result.evidence),
                        model=result.model,
                    )
                )

        in_scope_hits = [hit for hit in hits if hit.in_scope]
        counts: dict[str, int] = {
            "sentences_examined": in_scope_sentences,
            "pattern_hits": len(in_scope_hits),
            "hit_sentences": hit_sentences_in_scope,
            "out_of_scope_hits": len(hits) - len(in_scope_hits),
            "entailment_calls": entailment_calls,
            "entailment_flagged": entailment_flagged,
            "entailment_unverifiable": entailment_unverifiable,
            "entailment_entailed": entailment_entailed,
            "entailment_errors": entailment_errors,
        }
        rates: dict[str, float | None] = {
            "overclaim_pattern_rate": (
                hit_sentences_in_scope / in_scope_sentences
                if in_scope_sentences
                else None
            ),
            "entailment_flag_rate": (
                entailment_flagged / entailment_calls if entailment_calls else None
            ),
        }
        coverage = {
            "items_examined": in_scope_sentences,
            "items_skipped": out_of_scope_sentences,
            "skip_reasons": {"section_out_of_scope": out_of_scope_sentences},
        }
        details: list[dict[str, Any]] = [
            {
                "kind": "pattern_hit",
                "pattern_id": hit.pattern_id,
                "pattern_kind": hit.pattern_kind,
                "section": hit.section,
                "start": hit.start,
                "end": hit.end,
                "matched_text": hit.matched_text,
                "in_scope": hit.in_scope,
                "sentence": hit.sentence,
            }
            for hit in hits
        ]
        details.extend(
            {
                "kind": "entailment",
                "advisory_only": True,
                "sole_basis": False,
                "section": flag.section,
                "sentence": flag.sentence,
                "verdict": flag.verdict.value if flag.verdict is not None else None,
                "evidence": list(flag.evidence),
                "model": flag.model,
                "error": flag.error,
            }
            for flag in entailment_flags
        )
        return OverclaimResult(
            status="RUN_COMPLETE",
            not_run_reason=None,
            llm_assisted=entailment_calls > 0,
            counts=counts,
            rates=rates,
            coverage=coverage,
            details=tuple(details),
        )
