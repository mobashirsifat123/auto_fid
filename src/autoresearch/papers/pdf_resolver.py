"""Legal open-access PDF resolution and bounded downloading."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import httpx

from autoresearch.domain.models import content_checksum

from .metadata_sources import MetadataPaper


@dataclass(frozen=True)
class PdfResolution:
    available: bool
    url: str | None
    source: str
    legal_basis: str | None
    license: str | None
    reason: str | None


@dataclass(frozen=True)
class DownloadedPdf:
    path: Path
    checksum: str
    size_bytes: int


class PdfResolver(Protocol):
    async def resolve(self, paper: MetadataPaper) -> PdfResolution: ...
    async def download(self, resolution: PdfResolution, target: Path) -> DownloadedPdf: ...
    async def close(self) -> None: ...


class UnpaywallClient:
    def __init__(self, *, email_env: str = "UNPAYWALL_EMAIL", transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.email_env = email_env
        self.client = httpx.AsyncClient(base_url="https://api.unpaywall.org/v2", timeout=30, follow_redirects=True, transport=transport)

    async def resolve(self, doi: str) -> PdfResolution:
        email = os.getenv(self.email_env, "")
        if not email:
            return PdfResolution(False, None, "unpaywall", None, None, f"{self.email_env} is not configured")
        response = await self.client.get(f"/{doi}", params={"email": email})
        if response.status_code == 404:
            return PdfResolution(False, None, "unpaywall", None, None, "DOI was not found by Unpaywall")
        response.raise_for_status()
        payload = response.json()
        if not payload.get("is_oa"):
            return PdfResolution(False, None, "unpaywall", None, None, "No legal open-access location is recorded")
        location = payload.get("best_oa_location") or {}
        url = location.get("url_for_pdf")
        if not _safe_https(url):
            return PdfResolution(False, None, "unpaywall", None, None, "Open-access record has no HTTPS PDF URL")
        return PdfResolution(True, str(url), "unpaywall", "Unpaywall open-access location", location.get("license"), None)

    async def close(self) -> None:
        await self.client.aclose()


class LegalPdfResolver:
    """Resolve only source-declared OA PDFs; never scrape publisher pages."""

    def __init__(self, *, unpaywall: UnpaywallClient | None = None, max_bytes: int = 50 * 1024 * 1024, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.unpaywall = unpaywall or UnpaywallClient()
        self.max_bytes = max_bytes
        self.client = httpx.AsyncClient(timeout=60, follow_redirects=True, transport=transport)

    async def resolve(self, paper: MetadataPaper) -> PdfResolution:
        sources = set(paper.source.split(","))
        if paper.open_access_pdf_url and _safe_https(paper.open_access_pdf_url):
            if "arxiv" in sources:
                return PdfResolution(True, paper.open_access_pdf_url, "arxiv", "arXiv public full-text repository", "arXiv distribution terms", None)
            if "openreview" in sources:
                return PdfResolution(True, paper.open_access_pdf_url, "openreview", "OpenReview public submission", "OpenReview-hosted public PDF", None)
            if paper.is_open_access:
                return PdfResolution(True, paper.open_access_pdf_url, paper.source, "Metadata source explicitly marks this location open access", None, None)
        if paper.arxiv_id:
            return PdfResolution(True, f"https://arxiv.org/pdf/{paper.arxiv_id}.pdf", "arxiv", "arXiv public full-text repository", "arXiv distribution terms", None)
        if paper.doi:
            return await self.unpaywall.resolve(paper.doi)
        return PdfResolution(False, None, paper.source or "metadata", None, None, "No legally verified open-access PDF location")

    async def download(self, resolution: PdfResolution, target: Path) -> DownloadedPdf:
        if not resolution.available or not resolution.url or not resolution.legal_basis:
            raise ValueError("A legal PDF resolution is required before download.")
        if not _safe_https(resolution.url):
            raise ValueError("Only HTTPS PDF downloads are allowed.")
        target.parent.mkdir(parents=True, exist_ok=True)
        chunks: list[bytes] = []
        size = 0
        async with self.client.stream("GET", resolution.url) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "").lower()
            async for chunk in response.aiter_bytes():
                size += len(chunk)
                if size > self.max_bytes:
                    raise ValueError(f"PDF exceeds the {self.max_bytes} byte download limit.")
                chunks.append(chunk)
        payload = b"".join(chunks)
        if not payload.startswith(b"%PDF") and "application/pdf" not in content_type:
            raise ValueError("Resolved resource is not a PDF.")
        target.write_bytes(payload)
        return DownloadedPdf(target, content_checksum(payload), len(payload))

    async def close(self) -> None:
        await self.client.aclose()
        await self.unpaywall.close()


def _safe_https(value: object) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return parsed.scheme == "https" and bool(parsed.netloc) and not parsed.username and not parsed.password
