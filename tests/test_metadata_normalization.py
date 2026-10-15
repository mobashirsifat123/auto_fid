from __future__ import annotations

import httpx
import pytest

from autoresearch.papers.metadata_sources import CrossrefSource, OpenAlexSource, SemanticScholarSource, normalize_arxiv_id, normalize_doi


def test_identifier_normalization():
    assert normalize_doi("https://doi.org/10.1000/ABC.1") == "10.1000/abc.1"
    assert normalize_doi("not-a-doi") is None
    assert normalize_arxiv_id("https://arxiv.org/abs/2401.12345v2") == "2401.12345"


@pytest.mark.asyncio
async def test_semantic_scholar_normalization():
    payload = {"data": [{"paperId": "s1", "title": " Paper Title ", "abstract": "Abstract", "authors": [{"name": "A. Author"}], "year": 2025, "publicationDate": "2025-02-03", "externalIds": {"DOI": "10.1/TEST", "ArXiv": "2501.00001v1"}, "url": "https://example/p", "openAccessPdf": {"url": "https://example/p.pdf"}, "isOpenAccess": True, "citationCount": 7, "venue": "Venue"}]}
    source = SemanticScholarSource(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=payload)))
    result = await source.search("query", limit=20, since_year=2024)
    paper = result.papers[0]
    assert paper.doi == "10.1/test"
    assert paper.arxiv_id == "2501.00001"
    assert paper.authors == ("A. Author",)
    assert paper.is_open_access
    await source.close()


@pytest.mark.asyncio
async def test_openalex_and_crossref_normalization():
    openalex_payload = {"results": [{"id": "https://openalex.org/W1", "display_name": "OpenAlex paper", "abstract_inverted_index": {"useful": [1], "A": [0]}, "authorships": [{"author": {"display_name": "Researcher"}}], "publication_year": 2024, "publication_date": "2024-05-01", "doi": "https://doi.org/10.2/OA", "ids": {}, "primary_location": {"landing_page_url": "https://example/oa", "pdf_url": "https://example/oa.pdf", "source": {"display_name": "Journal"}}, "open_access": {"is_oa": True}, "cited_by_count": 3}]}
    source = OpenAlexSource(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=openalex_payload)))
    paper = (await source.search("query", limit=10)).papers[0]
    assert paper.abstract == "A useful"
    assert paper.doi == "10.2/oa"
    await source.close()

    crossref_payload = {"message": {"items": [{"DOI": "10.3/CR", "title": ["Crossref paper"], "abstract": "<jats:p>Useful abstract</jats:p>", "author": [{"given": "Ada", "family": "Lovelace"}], "published": {"date-parts": [[2023, 7]]}, "URL": "https://doi.org/10.3/CR", "link": [{"URL": "https://example/cr.pdf", "content-type": "application/pdf"}], "license": [{"URL": "https://license"}], "is-referenced-by-count": 5, "container-title": ["Proceedings"]}]}}
    source2 = CrossrefSource(transport=httpx.MockTransport(lambda request: httpx.Response(200, json=crossref_payload)))
    paper2 = (await source2.search("query", limit=10)).papers[0]
    assert paper2.authors == ("Ada Lovelace",)
    assert paper2.abstract == "Useful abstract"
    assert paper2.is_open_access
    await source2.close()
