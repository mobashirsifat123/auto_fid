"""Identifier-first scholarly metadata deduplication."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace

from .metadata_sources import MetadataPaper


@dataclass(frozen=True)
class SearchProvenance:
    source: str
    strategy: str
    query: str
    source_rank: int
    raw_checksum: str


@dataclass(frozen=True)
class SourcedCandidate:
    paper: MetadataPaper
    provenance: SearchProvenance


@dataclass(frozen=True)
class DeduplicatedCandidate:
    paper: MetadataPaper
    provenance: tuple[SearchProvenance, ...]


def normalized_title(value: str) -> str:
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode().lower()
    return " ".join(re.findall(r"[a-z0-9]+", value))


def deduplication_key(paper: MetadataPaper) -> str:
    if paper.doi:
        return f"doi:{paper.doi.lower()}"
    if paper.arxiv_id:
        return f"arxiv:{paper.arxiv_id.lower()}"
    author = normalized_title(paper.authors[0]) if paper.authors else "unknown"
    return f"title:{normalized_title(paper.title)}|author:{author}|year:{paper.year or 0}"


def deduplicate(candidates: list[SourcedCandidate]) -> list[DeduplicatedCandidate]:
    merged: dict[str, DeduplicatedCandidate] = {}
    for candidate in candidates:
        key = deduplication_key(candidate.paper)
        existing = merged.get(key)
        if not existing:
            merged[key] = DeduplicatedCandidate(candidate.paper, (candidate.provenance,))
            continue
        merged[key] = DeduplicatedCandidate(_merge(existing.paper, candidate.paper), (*existing.provenance, candidate.provenance))
    return list(merged.values())


def _merge(left: MetadataPaper, right: MetadataPaper) -> MetadataPaper:
    sources = sorted(set(filter(None, left.source.split(",") + right.source.split(","))))
    return MetadataPaper(
        title=left.title if len(left.title) >= len(right.title) else right.title,
        abstract=left.abstract if len(left.abstract) >= len(right.abstract) else right.abstract,
        authors=left.authors if len(left.authors) >= len(right.authors) else right.authors,
        year=left.year or right.year,
        publication_date=max(filter(None, (left.publication_date, right.publication_date)), default=None),
        doi=left.doi or right.doi, arxiv_id=left.arxiv_id or right.arxiv_id,
        url=left.url or right.url, open_access_pdf_url=left.open_access_pdf_url or right.open_access_pdf_url,
        is_open_access=left.is_open_access or right.is_open_access,
        citation_count=max(left.citation_count, right.citation_count), venue=left.venue or right.venue,
        source=",".join(sources), external_id=left.external_id or right.external_id,
    )
