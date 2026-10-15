"""Seed input and PDF parsing with checksums and page provenance."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from pypdf import PdfReader

from autoresearch.domain.models import content_checksum
from autoresearch.papers.metadata_sources import normalize_doi


@dataclass(frozen=True)
class ParsedPage:
    page: int
    text: str


@dataclass(frozen=True)
class ParsedDocument:
    input_type: str
    input_value: str
    title: str
    pages: tuple[ParsedPage, ...]
    checksum: str
    doi: str | None = None
    url: str | None = None

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages if page.text.strip())


class DocumentParser:
    def parse_seed(self, value: str | Path) -> ParsedDocument:
        text = str(value).strip()
        path = Path(text).expanduser()
        if path.exists():
            if not path.is_file() or path.suffix.lower() != ".pdf":
                raise ValueError("Seed path must be a PDF file.")
            payload = path.read_bytes()
            reader = PdfReader(path)
            pages = tuple(ParsedPage(index, page.extract_text() or "") for index, page in enumerate(reader.pages, start=1))
            metadata = reader.metadata or {}
            title = str(metadata.get("/Title") or path.stem).strip()
            if not pages:
                raise ValueError("Seed PDF has no pages.")
            return ParsedDocument("pdf", str(path.resolve()), title, pages, content_checksum(payload))
        doi = normalize_doi(text)
        if doi:
            return ParsedDocument("doi", text, doi, (), content_checksum(doi), doi=doi, url=f"https://doi.org/{doi}")
        parsed = urlparse(text)
        if parsed.scheme == "https" and parsed.netloc:
            doi_match = re.search(r"10\.\d{4,9}/[^?#\s]+", text, flags=re.I)
            normalized = normalize_doi(doi_match.group(0)) if doi_match else None
            return ParsedDocument("url", text, text, (), content_checksum(text), doi=normalized, url=text)
        if text and len(text) <= 2_000:
            return ParsedDocument("topic", text, text, (ParsedPage(1, text),), content_checksum(text))
        raise ValueError("Seed input must be a research topic, existing PDF path, DOI, or HTTPS URL.")
