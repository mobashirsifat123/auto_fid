"""Prose-style audit: kill the "AI-generated garbage feel" (2026-10-16).

The owner requirement: generated manuscripts must read at the level of the
deeply-reviewed exemplar papers — no LLM-writing tells, no formulaic filler.

Design (deterministic, fail-closed, disclosed-as-heuristic):
- :data:`AI_TELL_PATTERNS` — a frozen catalogue of well-documented
  LLM-prose tells (cliché verbs, throat-clearing adverbials, self-laudatory
  labels, formulaic connectors). Counted with spans, severity-weighted.
- :func:`extract_prose_metrics` — style statistics (sentence-length
  distribution, lexical diversity, connector density, concrete-number and
  citation density, passive-construction density).
- :func:`exemplar_style_band` — the SAME metrics computed over the top-K
  exemplar texts, producing target bands (median ± tolerance).
- :func:`prose_audit` — one manuscript vs the exemplar band: tells + style
  deviations + concrete rewrite guidance for the revision pass. Verdict
  ``PASS`` or ``REVISE``; hard ``REVISE`` when tell density exceeds the
  frozen ceiling regardless of band.

Honesty rails: this is a heuristic pattern/statistics audit — labeled as
such everywhere it is reported; it never certifies "human-written", it only
reports measured features and detected catalogue patterns. Style gaps close
by revision (or fallback), never by ignoring the audit.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

__all__ = [
    "AI_TELL_PATTERNS",
    "ProseAuditReport",
    "ProseMetrics",
    "exemplar_style_band",
    "extract_prose_metrics",
    "prose_audit",
]

# name -> (compiled pattern, per-1000-words ceiling, severity)
AI_TELL_PATTERNS: tuple[tuple[str, re.Pattern[str], float, int], ...] = (
    ("cliche_delve", re.compile(r"\bdelv(e|es|ing)\b", re.IGNORECASE), 0.0, 3),
    ("cliche_realm", re.compile(r"\bin the realm of\b", re.IGNORECASE), 0.0, 3),
    (
        "cliche_landscape",
        re.compile(r"\bthe\s+\w+\s+landscape\b", re.IGNORECASE),
        0.0,
        2,
    ),
    (
        "cliche_crucial_role",
        re.compile(r"\bplays? a (crucial|pivotal|vital) role\b", re.IGNORECASE),
        0.0,
        3,
    ),
    (
        "self_novel",
        re.compile(
            r"\bour novel\b|\bwe (propose|present|introduce) a novel\b", re.IGNORECASE
        ),
        0.5,
        2,
    ),
    (
        "throat_clearing",
        re.compile(
            r"\b(it is worth noting|it should be noted|needless to say)\b", re.I
        ),
        0.5,
        2,
    ),
    (
        "filler_notably",
        re.compile(
            r"^\s*(Notably|Importantly|Interestingly|Remarkably),", re.MULTILINE | re.IGNORECASE
        ),
        1.0,
        2,
    ),
    (
        "formulaic_connector",
        re.compile(r"^\s*(Moreover|Furthermore|Additionally),", re.MULTILINE),
        2.0,
        1,
    ),
    ("cliche_leverage", re.compile(r"\bleverag(e|es|ing)\b", re.IGNORECASE), 1.0, 2),
    (
        "cliche_showcase",
        re.compile(r"\b(showcas\w+|underscor\w+)\b", re.IGNORECASE),
        1.0,
        2,
    ),
    (
        "hedge_seam",
        re.compile(
            r"\b(seamlessly|holistic\w+|game-changing|cutting-edge|state-of-the-art framework)\b",
            re.I,
        ),
        0.0,
        3,
    ),
    (
        "kitchen_sink",
        re.compile(
            r"\bcomprehensive (framework|approach|method|pipeline)\b", re.IGNORECASE
        ),
        0.5,
        2,
    ),
)

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'-]*")
_CITATION = re.compile(r"\[\w+[-\w]*\]|\([^)]*\d{4}[a-z]?\)|\\cite\{[^}]*\}")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?%?\b")
_PASSIVE = re.compile(r"\b(is|are|was|were|be|been)\s+\w+(ed|en)\b", re.IGNORECASE)

_TELL_DENSITY_CEILING = 3.0  # total weighted tells per 1000 words → hard REVISE


@dataclass(frozen=True)
class ProseMetrics:
    words: int
    sentences: int
    avg_sentence_length: float
    sentence_length_std: float | None
    type_token_ratio: float
    connector_density: float  # per 1000 words
    citation_density: float  # per 1000 words
    numeric_density: float  # per 1000 words
    passive_density: float  # per 1000 words
    tell_density: float  # weighted, per 1000 words

    def as_dict(self) -> dict[str, Any]:
        out = self.__dict__.copy()
        out["sentence_length_std"] = self.sentence_length_std  # may be None — honest
        return out


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text.strip()) if s.strip()]


def _tell_density(text: str, words: int) -> tuple[float, list[dict[str, Any]]]:
    per_k = 1000.0 / max(words, 1)
    total_weight = 0
    tells: list[dict[str, Any]] = []
    for name, pattern, ceiling, severity in AI_TELL_PATTERNS:
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        density = len(matches) * per_k
        total_weight += len(matches) * severity
        tells.append(
            {
                "tell": name,
                "count": len(matches),
                "density_per_1k": round(density, 2),
                "ceiling": ceiling,
                "severity": severity,
                "over_ceiling": density > ceiling,
                "first_span": (matches[0].start(), matches[0].end()),
            }
        )
    return total_weight * per_k, tells


def extract_prose_metrics(text: str) -> ProseMetrics:
    """Deterministic prose statistics. Works identically on exemplar PDFs and
    generated manuscripts (same extractor discipline as application.exemplars)."""
    words = _WORD.findall(text)
    sentences = _sentences(text)
    lengths = [len(_WORD.findall(s)) for s in sentences] or [0]
    word_count = len(words)
    per_k = 1000.0 / max(word_count, 1)
    connector = re.compile(
        r"^(Moreover|Furthermore|Additionally|However|Therefore|Thus|Hence)", re.MULTILINE
    )
    tell_density, _ = _tell_density(text, word_count)
    return ProseMetrics(
        words=word_count,
        sentences=len(sentences),
        avg_sentence_length=round(statistics.fmean(lengths), 2),
        sentence_length_std=round(statistics.stdev(lengths), 2)
        if len(lengths) > 1
        else None,
        type_token_ratio=round(
            len(set(w.lower() for w in words)) / max(word_count, 1), 3
        ),
        connector_density=round(len(connector.findall(text)) * per_k, 2),
        citation_density=round(len(_CITATION.findall(text)) * per_k, 2),
        numeric_density=round(len(_NUMBER.findall(text)) * per_k, 2),
        passive_density=round(len(_PASSIVE.findall(text)) * per_k, 2),
        tell_density=round(tell_density, 2),
    )


@dataclass(frozen=True)
class StyleBand:
    metric: str
    exemplar_median: float
    low: float
    high: float

    def within(self, value: float) -> bool:
        return self.low <= value <= self.high


def exemplar_style_band(exemplar_texts: Sequence[str]) -> dict[str, StyleBand]:
    """Target bands from exemplar prose: median ± max(25%, 1 unit) tolerance.
    Empty input → empty bands (audit reports inconclusive, never a pass)."""
    if not exemplar_texts:
        return {}
    metrics = [extract_prose_metrics(t) for t in exemplar_texts]
    bands: dict[str, StyleBand] = {}
    for field in (
        "avg_sentence_length",
        "sentence_length_std",
        "type_token_ratio",
        "connector_density",
        "citation_density",
        "numeric_density",
        "passive_density",
    ):
        values = [getattr(m, field) for m in metrics if getattr(m, field) is not None]
        if not values:
            continue
        med = statistics.median(values)
        tol = max(abs(med) * 0.25, 1.0)
        bands[field] = StyleBand(
            field, round(med, 2), round(med - tol, 2), round(med + tol, 2)
        )
    return bands


@dataclass(frozen=True)
class ProseAuditReport:
    verdict: str  # PASS | REVISE | INCONCLUSIVE
    heuristic: str  # always "deterministic_pattern_statistics_v1"
    metrics: ProseMetrics | None
    tells: tuple[Mapping[str, Any], ...]
    band_deviations: tuple[Mapping[str, Any], ...]
    rewrite_guidance: tuple[str, ...]
    notes: tuple[str, ...] = ()


def prose_audit(
    manuscript_text: str,
    exemplar_texts: Sequence[str],
) -> ProseAuditReport:
    """Audit one manuscript's prose against the exemplar band and the tell
    catalogue. Hard REVISE on tell density over the frozen ceiling; banded
    REVISE on style deviations; INCONCLUSIVE without exemplars (never PASS)."""
    metrics = extract_prose_metrics(manuscript_text)
    _, tells = _tell_density(manuscript_text, max(metrics.words, 1))
    guidance: list[str] = []

    if metrics.tell_density > _TELL_DENSITY_CEILING or any(
        t["over_ceiling"] for t in tells
    ):
        guidance.append(
            "rewrite flagged LLM-prose tells (see tells[]): replace cliché verbs/"
            "connectors with concrete statements tied to measurements"
        )

    if not exemplar_texts:
        return ProseAuditReport(
            verdict="INCONCLUSIVE",
            heuristic="deterministic_pattern_statistics_v1",
            metrics=metrics,
            tells=tuple(tells),
            band_deviations=(),
            rewrite_guidance=tuple(guidance)
            or ("no exemplar texts supplied; band comparison skipped"),
            notes=(
                "no_exemplars: style band cannot be computed; verdict is not a pass",
            ),
        )

    bands = exemplar_style_band(exemplar_texts)
    deviations: list[Mapping[str, Any]] = []
    values = metrics.as_dict()
    for name, band in bands.items():
        value = float(values[name])
        if not band.within(value):
            deviations.append(
                {
                    "metric": name,
                    "manuscript": value,
                    "exemplar_band": [band.low, band.high],
                    "exemplar_median": band.exemplar_median,
                }
            )
            if name in ("citation_density", "numeric_density") and value < band.low:
                guidance.append(
                    f"'{name}' below exemplar band — add concrete, evidence-backed "
                    "citations/numbers (only from resolved sources and measurements)"
                )
            elif name == "avg_sentence_length":
                guidance.append(
                    f"'{name}' outside exemplar band — vary sentence structure toward "
                    f"exemplar median {band.exemplar_median}"
                )
            else:
                guidance.append(
                    f"adjust '{name}' toward the exemplar band {band.low}–{band.high}"
                )

    hard_fail = metrics.tell_density > _TELL_DENSITY_CEILING or any(
        t["over_ceiling"] for t in tells
    )
    verdict = "REVISE" if (hard_fail or deviations) else "PASS"
    return ProseAuditReport(
        verdict=verdict,
        heuristic="deterministic_pattern_statistics_v1",
        metrics=metrics,
        tells=tuple(tells),
        band_deviations=tuple(deviations),
        rewrite_guidance=tuple(guidance),
    )
