"""Deterministic keyword extraction shared by the review, findings-memory,
and manuscript stages.

Keywords are always DERIVED from stored text (seed metadata, paper metadata,
recorded findings) — never invented by a model. The tokenizer is a simple,
inspectable heuristic: lowercase alphanumeric tokens, stopwords removed,
first-seen order, capped length. It is NOT topic modeling and makes no
semantic claims.
"""

from __future__ import annotations

import re

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "without",
    "by", "as", "is", "are", "can", "does", "do", "not", "how", "what", "which",
    "that", "this", "these", "those", "it", "its", "from", "at", "be", "using",
    "into", "than", "then", "when", "where", "who", "why", "will", "would", "there",
    "was", "were", "has", "have", "had", "their", "our", "we", "they",
}

_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z0-9_-]{2,}")


def keyword_tokens(*texts: str, limit: int = 24) -> list[str]:
    """Ordered, deduplicated keyword tokens from stored text. First-seen
    order is preserved so the manuscript keywords stay reproducible."""
    words: list[str] = []
    for text in texts:
        for token in _TOKEN_RE.findall(str(text or "").lower()):
            if token not in _STOPWORDS and token not in words:
                words.append(token)
        if len(words) >= limit:
            break
    return words[:limit]


def pick_keywords(sources: list[list[str]], *, maximum: int = 6) -> list[str]:
    """Merge ordered keyword sources (earlier sources rank first), dedupe,
    cap at ``maximum``. Returns as many as the sources honestly provide."""
    merged: list[str] = []
    for source in sources:
        for token in source:
            if token not in merged:
                merged.append(token)
    return merged[:maximum]
