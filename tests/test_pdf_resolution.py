from __future__ import annotations

import httpx
import pytest

from autoresearch.papers.metadata_sources import MetadataPaper
from autoresearch.papers.pdf_resolver import LegalPdfResolver, UnpaywallClient


@pytest.mark.asyncio
async def test_known_open_repository_is_accepted():
    resolver = LegalPdfResolver()
    paper = MetadataPaper("Paper", arxiv_id="2501.00001", source="arxiv", is_open_access=True, open_access_pdf_url="https://arxiv.org/pdf/2501.00001.pdf")
    result = await resolver.resolve(paper)
    assert result.available
    assert result.legal_basis == "arXiv public full-text repository"
    await resolver.close()


@pytest.mark.asyncio
async def test_unverified_or_insecure_pdf_is_not_downloaded(monkeypatch):
    monkeypatch.delenv("UNPAYWALL_EMAIL", raising=False)
    resolver = LegalPdfResolver()
    paper = MetadataPaper("Paper", source="publisher", open_access_pdf_url="http://publisher.example/p.pdf", is_open_access=True)
    result = await resolver.resolve(paper)
    assert not result.available
    assert "legally verified" in result.reason
    await resolver.close()


@pytest.mark.asyncio
async def test_unpaywall_resolution_and_bounded_pdf_download(tmp_path, monkeypatch):
    monkeypatch.setenv("UNPAYWALL_EMAIL", "researcher@example.com")
    unpaywall = UnpaywallClient(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={"is_oa": True, "best_oa_location": {"url_for_pdf": "https://oa.example/paper.pdf", "license": "cc-by"}})))
    resolver = LegalPdfResolver(unpaywall=unpaywall, transport=httpx.MockTransport(lambda request: httpx.Response(200, content=b"%PDF-1.4\n%%EOF", headers={"content-type": "application/pdf"})))
    resolution = await resolver.resolve(MetadataPaper("Paper", doi="10.1/test", source="crossref"))
    downloaded = await resolver.download(resolution, tmp_path / "paper.pdf")
    assert resolution.available and resolution.license == "cc-by"
    assert downloaded.path.exists()
    assert len(downloaded.checksum) == 64
    await resolver.close()
