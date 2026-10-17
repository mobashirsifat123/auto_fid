"""Claim-evidence support auditor (DESIGN §5.2).

Deterministic tiers over claim-bearing sentences:

- ``verbatim_verified``: the claim contains a quoted/verbatim span that is
  verified against an evidence pool via :func:`verify_span` at EXACT or
  WHITESPACE_ONLY divergence (the two classes a quotation is allowed to have;
  NORMALIZED_ONLY and PARAPHRASE_ADJACENT are recorded as details but are
  *never* counted as verification — this is the hardened replacement for the
  whitespace-normalized substring soft spot in review/extraction.py).
- ``artifact_traced``: every numeric token in the claim appears in an
  artifact-role text (exact numeric-literal match after canonical
  formatting) — i.e. the artifact could have produced this sentence.
- ``citation_backed``: the sentence carries a citation marker whose key
  resolves in the audited bibliography (weak tier).

Fail-closed: an empty evidence pool yields NOT_RUN (harness cannot assess),
never a passing rate.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ClaimsAuditor",
    "ClaimsResult",
    "SpanDivergence",
    "verify_span",
]

_METRIC = "claims"

_CLAIM_MARKERS: tuple[str, ...] = (
    "we show",
    "we demonstrate",
    "we find",
    "we observe",
    "our method",
    "our approach",
    "our model",
    "achieves",
    "achieve",
    "outperforms",
    "outperform",
    "improves",
    "improve over",
    "surpasses",
    "exceeds",
    "state-of-the-art",
    "results show",
    "experiments show",
    "experiments demonstrate",
)


class SpanDivergence:
    """Divergence classes for verbatim-span verification (DESIGN §7.1)."""

    EXACT_OFFSET = "exact_offset"
    WHITESPACE_ONLY = "whitespace_only"
    NORMALIZED_ONLY = "normalized_only"
    PARAPHRASE_ADJACENT = "paraphrase_adjacent"
    NOT_FOUND = "not_found"


def _normalize_ws(text: str) -> str:
    return " ".join(text.split())


def _normalize_full(text: str) -> str:
    """NFKD → ASCII → lowercase token string (papers.deduplication parity)."""
    decomposed = unicodedata.normalize("NFKD", text)
    ascii_text = decomposed.encode("ascii", "ignore").decode("ascii")
    return _normalize_ws(ascii_text).lower()


def _token_set(text: str) -> set[str]:
    return set(_normalize_full(text).split())


def verify_span(quote: str, source_text: str, *, min_paraphrase_tokens: int = 6) -> str:
    """Classify how ``quote`` relates to ``source_text``.

    Only EXACT_OFFSET and WHITESPACE_ONLY count as verbatim verification by
    callers; the softer classes are recorded, never trusted.
    """
    if not quote.strip():
        return SpanDivergence.NOT_FOUND
    if quote in source_text:
        return SpanDivergence.EXACT_OFFSET
    if _normalize_ws(quote) in _normalize_ws(source_text):
        return SpanDivergence.WHITESPACE_ONLY
    if _normalize_full(quote) in _normalize_full(source_text):
        return SpanDivergence.NORMALIZED_ONLY
    # Paraphrase-adjacent: high token-overlap sliding window in the source.
    quote_tokens = _normalize_full(quote).split()
    if len(quote_tokens) >= min_paraphrase_tokens:
        quote_set = set(quote_tokens)
        source_tokens = _normalize_full(source_text).split()
        window = len(quote_tokens)
        for start in range(max(0, len(source_tokens) - window) + 1):
            chunk = source_tokens[start : start + window]
            overlap = len(quote_set & set(chunk)) / len(quote_set)
            if overlap >= 0.8:
                return SpanDivergence.PARAPHRASE_ADJACENT
    return SpanDivergence.NOT_FOUND


def _split_sentences(text: str) -> list[tuple[str, int, int]]:
    """Sentence splitter returning (sentence, start, end). Deliberately
    conservative: abbreviations like ``e.g.``/``et al.``/``Fig.`` do not end a
    sentence."""
    sentences: list[tuple[str, int, int]] = []
    start = 0
    index = 0
    protected = ("e.g.", "i.e.", "et al.", "Fig.", "eq.", "Eq.", "vs.", "cf.", "Sec.")
    length = len(text)
    while index < length:
        char = text[index]
        if char in ".!?" and index + 1 < length and text[index + 1] in " \n\t":
            candidate_end = index + 1
            window = text[max(0, index - 8) : index + 1]
            if not any(window.endswith(p) for p in protected):
                sentence = text[start:candidate_end]
                if sentence.strip():
                    sentences.append((sentence, start, candidate_end))
                start = candidate_end
                while start < length and text[start] in " \n\t":
                    start += 1
                index = start
                continue
        index += 1
    if start < length and text[start:].strip():
        sentences.append((text[start:], start, length))
    return sentences


def _is_claim(sentence: str, *, require_marker: bool = True) -> bool:
    lowered = sentence.lower()
    if any(marker in lowered for marker in _CLAIM_MARKERS):
        return True
    if require_marker is False:
        return False
    # A bare percentage claim ("92.3%") without a marker is still a claim
    # when it asserts a result in an experiment-adjacent sentence.
    return "%" in sentence and any(
        word in lowered for word in ("accuracy", "f1", "auc", "performance", "result")
    )


_NUMBER_CHARS = set("0123456789")


def _numeric_literals(sentence: str) -> list[str]:
    """Canonical numeric literals in the sentence (integers and decimals)."""
    literals: list[str] = []
    current: list[str] = []
    for char in sentence:
        if char in _NUMBER_CHARS or (char == "." and current and "." not in current):
            current.append(char)
        else:
            if current:
                literals.append(_canonical_number("".join(current)))
                current = []
    if current:
        literals.append(_canonical_number("".join(current)))
    return [value for value in literals if value != ""]


def _canonical_number(literal: str) -> str:
    literal = literal.removesuffix(".")
    if "." in literal:
        literal = literal.rstrip("0").rstrip(".")
        return literal if literal else "0"
    return literal.lstrip("0") or "0"


def _percent_variants(literal: str) -> set[str]:
    """Artifact-side forms of a percent-adjacent literal: ``92.3`` ↔ ``0.923``
    (same normalization rule as the numbers auditor's percent handling)."""
    variants: set[str] = set()
    try:
        value = float(literal)
    except ValueError:
        return variants
    for candidate in (value / 100.0, value * 100.0):
        text = _canonical_number(f"{candidate:.6f}".rstrip("0"))
        if text:
            variants.add(text)
    return variants


_CITATION_PREFIXES = ("@", "\\cite{")


def _strip_citation_markers(sentence: str) -> str:
    """Remove citation markers so reference years (e.g. '@good2024' → 2024)
    are never treated as claim numbers."""
    cleaned = sentence
    for prefix in _CITATION_PREFIXES:
        scan = 0
        result = []
        while True:
            at = cleaned.find(prefix, scan)
            if at < 0:
                result.append(cleaned[scan:])
                break
            result.append(cleaned[scan:at])
            rest = cleaned[at + len(prefix) :]
            end = rest.find("}" if prefix == "\\cite{" else " ")
            consumed = len(prefix) + (end if end >= 0 else len(rest))
            scan = at + consumed
        cleaned = "".join(result)
    return cleaned


def _citation_keys(sentence: str) -> list[str]:
    keys: list[str] = []
    for prefix in _CITATION_PREFIXES:
        scan = 0
        while True:
            at = sentence.find(prefix, scan)
            if at < 0:
                break
            rest = sentence[at + len(prefix) :]
            end = rest.find("}" if prefix == "\\cite{" else " ")
            token = rest[: end if end >= 0 else len(rest)]
            token = token.strip('.,;:) ]}"\u2019\u201d')
            if token:
                keys.append(token)
            scan = at + len(prefix)
    return keys


@dataclass(frozen=True)
class ClaimsResult:
    status: str  # RUN_COMPLETE | NOT_RUN
    not_run_reason: str | None
    counts: Mapping[str, int]
    rates: Mapping[str, float | None]
    coverage: Mapping[str, Any]
    details: tuple[Mapping[str, Any], ...]

    def to_section(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "not_run_reason": self.not_run_reason,
            "llm_assisted": False,
            "counts": dict(self.counts),
            "rates": dict(self.rates),
            "coverage": dict(self.coverage),
            "details": [dict(detail) for detail in self.details],
        }


def _not_run(reason: str) -> ClaimsResult:
    return ClaimsResult(
        status="NOT_RUN",
        not_run_reason=reason,
        counts={},
        rates={"strong_support_rate": None, "citation_backed_rate": None},
        coverage={"items_examined": 0, "items_skipped": 0, "skip_reasons": {}},
        details=(),
    )


class ClaimsAuditor:
    """Tiered claim-evidence support over paper sections (§5.2)."""

    def audit(
        self,
        sections: Sequence[Any],
        *,
        evidence_texts: Sequence[tuple[str, str]] = (),
        bibliography_keys: frozenset[str] = frozenset(),
    ) -> ClaimsResult:
        """``sections``: objects with ``name``/``text`` (contract.TextSection
        or numbers.TextSection). ``evidence_texts``: (label, text) pairs from
        artifact-role files and resolved-source metadata. ``bibliography_keys``:
        parsed bibliography keys (empty ⇒ citation tier is unavailable ⇒
        NOT_RUN only if the paper actually cites something)."""
        if not any(section.text.strip() for section in sections):
            return _not_run("no_paper_text")

        claims_examined = 0
        verbatim_verified = 0
        artifact_traced = 0
        citation_backed = 0
        details: list[dict[str, Any]] = []
        skipped: dict[str, int] = {}
        evidence_label_text = tuple(evidence_texts)
        for section in sections:
            for sentence, offset, _end in _split_sentences(section.text):
                if not _is_claim(sentence):
                    continue
                claims_examined += 1
                claim_detail: dict[str, Any] = {
                    "section": section.name,
                    "offset": offset,
                    "tier": "unsupported",
                    "span_class": None,
                }

                # 1. verbatim tier: quoted span verified against evidence pool.
                tier = "unsupported"
                span_class = None
                quoted = []
                scan = 0
                while True:
                    open_at = sentence.find('"', scan)
                    if open_at < 0:
                        break
                    close_at = sentence.find('"', open_at + 1)
                    if close_at < 0:
                        break
                    quoted.append(sentence[open_at + 1 : close_at])
                    scan = close_at + 1
                for quote in quoted:
                    for label, evidence in evidence_label_text:
                        found = verify_span(quote, evidence)
                        if found in (
                            SpanDivergence.EXACT_OFFSET,
                            SpanDivergence.WHITESPACE_ONLY,
                        ):
                            tier = "verbatim_verified"
                            span_class = found
                            claim_detail["evidence_label"] = label
                            break
                        if found == SpanDivergence.NORMALIZED_ONLY:
                            # recorded, not trusted (soft-spot hardening)
                            span_class = span_class or found
                            claim_detail["normalized_only_in"] = label
                    if tier == "verbatim_verified":
                        break

                # 2. artifact-traced tier: numeric literals present in artifacts.
                if tier == "unsupported":
                    literals = _numeric_literals(_strip_citation_markers(sentence))
                    if literals:
                        artifact_blob = " ".join(
                            text
                            for label, text in evidence_label_text
                            if label.startswith("artifact:")
                        )
                        if artifact_blob:
                            percent_context = "%" in sentence
                            matched = True
                            for value in literals:
                                canonical = _canonical_number(value)
                                if canonical in artifact_blob or value in artifact_blob:
                                    continue
                                if percent_context and any(
                                    variant in artifact_blob
                                    for variant in _percent_variants(value)
                                ):
                                    continue
                                matched = False
                                break
                            if matched:
                                tier = "artifact_traced"

                # 3. citation-backed (weak) tier.
                keys = _citation_keys(sentence)
                if keys:
                    if bibliography_keys:
                        if any(key in bibliography_keys for key in keys):
                            if tier == "unsupported":
                                tier = "citation_backed"
                        else:
                            skipped["citation_key_unresolved"] = (
                                skipped.get("citation_key_unresolved", 0) + 1
                            )
                    else:
                        skipped["no_bibliography"] = (
                            skipped.get("no_bibliography", 0) + 1
                        )

                if tier == "verbatim_verified":
                    verbatim_verified += 1
                    artifact_traced += 1  # verbatim implies traceable support
                elif tier == "artifact_traced":
                    artifact_traced += 1
                elif tier == "citation_backed":
                    citation_backed += 1
                claim_detail["tier"] = tier
                claim_detail["span_class"] = span_class
                details.append(claim_detail)

        if claims_examined == 0:
            return _not_run("no_claims_found")

        return ClaimsResult(
            status="RUN_COMPLETE",
            not_run_reason=None,
            counts={
                "claims_examined": claims_examined,
                "verbatim_verified": verbatim_verified,
                "artifact_traced": artifact_traced,
                "citation_backed": citation_backed,
            },
            rates={},  # derived by MetricSection.complete in wiring
            coverage={
                "items_examined": claims_examined,
                "items_skipped": sum(skipped.values()),
                "skip_reasons": dict(skipped),
            },
            details=tuple(details),
        )
