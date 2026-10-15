"""Async scholarly metadata-source adapters with normalized records."""

from __future__ import annotations

import asyncio
import html
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from typing import Any, Protocol
from urllib.parse import quote

import httpx

from autoresearch.domain.models import content_checksum


@dataclass(frozen=True)
class MetadataPaper:
    title: str
    abstract: str = ""
    authors: tuple[str, ...] = ()
    year: int | None = None
    publication_date: str | None = None
    doi: str | None = None
    arxiv_id: str | None = None
    url: str | None = None
    open_access_pdf_url: str | None = None
    is_open_access: bool = False
    citation_count: int = 0
    venue: str = ""
    source: str = ""
    external_id: str = ""


@dataclass(frozen=True)
class SourceSearchResult:
    source: str
    query: str
    papers: tuple[MetadataPaper, ...]
    raw_checksum: str


class MetadataSource(Protocol):
    name: str
    async def search(self, query: str, *, limit: int, since_year: int | None = None) -> SourceSearchResult: ...
    async def close(self) -> None: ...


class HttpMetadataSource:
    name = "metadata"

    def __init__(self, base_url: str, *, requests_per_second: float = 2.0, timeout: float = 30.0, transport: httpx.AsyncBaseTransport | None = None) -> None:
        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout, follow_redirects=True, transport=transport)
        self._interval = 1.0 / requests_per_second
        self._lock = asyncio.Lock()
        self._next = 0.0

    async def _get(self, path: str, *, params: dict[str, Any], headers: dict[str, str] | None = None) -> httpx.Response:
        async with self._lock:
            now = time.monotonic()
            if self._next > now:
                await asyncio.sleep(self._next - now)
            self._next = max(now, self._next) + self._interval
        response = await self.client.get(path, params=params, headers=headers)
        response.raise_for_status()
        return response

    async def close(self) -> None:
        await self.client.aclose()


class SemanticScholarSource(HttpMetadataSource):
    name = "semantic_scholar"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("https://api.semanticscholar.org/graph/v1", **kwargs)

    async def search(self, query: str, *, limit: int, since_year: int | None = None) -> SourceSearchResult:
        fields = "paperId,title,abstract,authors,year,publicationDate,externalIds,url,openAccessPdf,isOpenAccess,citationCount,venue"
        params: dict[str, Any] = {"query": query, "limit": min(limit, 100), "fields": fields}
        if since_year:
            params["year"] = f"{since_year}-"
        key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
        response = await self._get("/paper/search", params=params, headers={"x-api-key": key} if key else None)
        payload = response.json()
        papers = []
        for row in payload.get("data") or []:
            external = row.get("externalIds") or {}
            oa = row.get("openAccessPdf") or {}
            papers.append(MetadataPaper(
                title=_text(row.get("title")), abstract=_text(row.get("abstract")),
                authors=tuple(_text(item.get("name")) for item in row.get("authors") or [] if _text(item.get("name"))),
                year=_year(row.get("year")), publication_date=_date(row.get("publicationDate")),
                doi=normalize_doi(external.get("DOI")), arxiv_id=normalize_arxiv_id(external.get("ArXiv")),
                url=_url(row.get("url")), open_access_pdf_url=_url(oa.get("url")),
                is_open_access=bool(row.get("isOpenAccess")), citation_count=_count(row.get("citationCount")),
                venue=_text(row.get("venue")), source=self.name, external_id=_text(row.get("paperId")),
            ))
        return SourceSearchResult(self.name, query, tuple(papers), content_checksum(payload))


class OpenAlexSource(HttpMetadataSource):
    name = "openalex"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("https://api.openalex.org", **kwargs)

    async def search(self, query: str, *, limit: int, since_year: int | None = None) -> SourceSearchResult:
        params: dict[str, Any] = {"search": query, "per-page": min(limit, 200)}
        if since_year:
            params["filter"] = f"from_publication_date:{since_year}-01-01"
        mailto = os.getenv("OPENALEX_MAILTO", "")
        if mailto:
            params["mailto"] = mailto
        payload = (await self._get("/works", params=params)).json()
        papers = []
        for row in payload.get("results") or []:
            ids = row.get("ids") or {}
            location = row.get("primary_location") or {}
            source = location.get("source") or {}
            oa = row.get("open_access") or {}
            papers.append(MetadataPaper(
                title=_text(row.get("display_name") or row.get("title")), abstract=_openalex_abstract(row.get("abstract_inverted_index")),
                authors=tuple(_text((item.get("author") or {}).get("display_name")) for item in row.get("authorships") or [] if _text((item.get("author") or {}).get("display_name"))),
                year=_year(row.get("publication_year")), publication_date=_date(row.get("publication_date")),
                doi=normalize_doi(row.get("doi") or ids.get("doi")), arxiv_id=normalize_arxiv_id(ids.get("arxiv")),
                url=_url(location.get("landing_page_url") or row.get("id")), open_access_pdf_url=_url(location.get("pdf_url") or oa.get("oa_url")),
                is_open_access=bool(oa.get("is_oa")), citation_count=_count(row.get("cited_by_count")), venue=_text(source.get("display_name")),
                source=self.name, external_id=_text(row.get("id")),
            ))
        return SourceSearchResult(self.name, query, tuple(papers), content_checksum(payload))


class CrossrefSource(HttpMetadataSource):
    name = "crossref"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("https://api.crossref.org", **kwargs)

    async def search(self, query: str, *, limit: int, since_year: int | None = None) -> SourceSearchResult:
        params: dict[str, Any] = {"query.bibliographic": query, "rows": min(limit, 1000), "select": "DOI,title,abstract,author,published,URL,link,is-referenced-by-count,container-title,license"}
        mailto = os.getenv("CROSSREF_MAILTO", "")
        if mailto:
            params["mailto"] = mailto
        if since_year:
            params["filter"] = f"from-pub-date:{since_year}-01-01"
        payload = (await self._get("/works", params=params)).json()
        papers = []
        for row in (payload.get("message") or {}).get("items") or []:
            date_parts = ((row.get("published") or {}).get("date-parts") or [[None]])[0]
            publication_date = "-".join(f"{int(value):02d}" if index else str(value) for index, value in enumerate(date_parts) if value is not None) or None
            links = row.get("link") or []
            pdf = next((_url(item.get("URL")) for item in links if "pdf" in _text(item.get("content-type")).lower()), None)
            licenses = row.get("license") or []
            papers.append(MetadataPaper(
                title=_text((row.get("title") or [""])[0]), abstract=_strip_markup(_text(row.get("abstract"))),
                authors=tuple(" ".join(filter(None, (_text(item.get("given")), _text(item.get("family"))))).strip() for item in row.get("author") or []),
                year=_year(date_parts[0] if date_parts else None), publication_date=_date(publication_date), doi=normalize_doi(row.get("DOI")),
                url=_url(row.get("URL")), open_access_pdf_url=pdf, is_open_access=bool(pdf and licenses), citation_count=_count(row.get("is-referenced-by-count")),
                venue=_text((row.get("container-title") or [""])[0]), source=self.name, external_id=_text(row.get("DOI")),
            ))
        return SourceSearchResult(self.name, query, tuple(papers), content_checksum(payload))


class ArxivSource(HttpMetadataSource):
    name = "arxiv"
    _ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("https://export.arxiv.org", requests_per_second=kwargs.pop("requests_per_second", 0.34), **kwargs)

    async def search(self, query: str, *, limit: int, since_year: int | None = None) -> SourceSearchResult:
        response = await self._get("/api/query", params={"search_query": f"all:{query}", "start": 0, "max_results": min(limit, 100), "sortBy": "submittedDate", "sortOrder": "descending"})
        root = ET.fromstring(response.content)
        papers = []
        for entry in root.findall("a:entry", self._ATOM):
            published = _node_text(entry, "a:published", self._ATOM)
            year = _year(published[:4])
            if since_year and year and year < since_year:
                continue
            identifier = _node_text(entry, "a:id", self._ATOM).rstrip("/").split("/")[-1]
            identifier = identifier.removeprefix("abs/")
            pdf = next((item.attrib.get("href") for item in entry.findall("a:link", self._ATOM) if item.attrib.get("type") == "application/pdf" or item.attrib.get("title") == "pdf"), None)
            authors = tuple(_node_text(item, "a:name", self._ATOM) for item in entry.findall("a:author", self._ATOM))
            papers.append(MetadataPaper(title=_node_text(entry, "a:title", self._ATOM), abstract=_node_text(entry, "a:summary", self._ATOM), authors=authors, year=year, publication_date=_date(published), doi=normalize_doi(_node_text(entry, "arxiv:doi", self._ATOM)), arxiv_id=normalize_arxiv_id(identifier), url=_node_text(entry, "a:id", self._ATOM), open_access_pdf_url=_url(pdf), is_open_access=True, venue="arXiv", source=self.name, external_id=identifier))
        return SourceSearchResult(self.name, query, tuple(papers), content_checksum(response.content))


class OpenReviewSource(HttpMetadataSource):
    name = "openreview"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__("https://api2.openreview.net", **kwargs)

    async def search(self, query: str, *, limit: int, since_year: int | None = None) -> SourceSearchResult:
        payload = (await self._get("/notes/search", params={"term": query, "source": "forum", "limit": min(limit, 1000), "offset": 0})).json()
        papers = []
        for row in payload.get("notes") or []:
            # Search may return a review/rebuttal note and attach the actual
            # submission fields under forumContent. Prefer the forum payload so
            # only paper metadata enters discovery.
            content = row.get("forumContent") or row.get("content") or {}
            value = lambda key, default="": (content.get(key) or {}).get("value", default) if isinstance(content.get(key), dict) else content.get(key, default)
            date_ms = row.get("pdate") or row.get("cdate")
            year = int(time.gmtime(int(date_ms) / 1000).tm_year) if date_ms else None
            if since_year and year and year < since_year:
                continue
            identifier = _text(row.get("forum") or row.get("id"))
            pdf_path = _text(value("pdf"))
            pdf = f"https://openreview.net{pdf_path}" if pdf_path.startswith("/") else _url(pdf_path)
            authors = value("authors", [])
            papers.append(MetadataPaper(title=_text(value("title")), abstract=_text(value("abstract")), authors=tuple(map(str, authors if isinstance(authors, list) else [authors])), year=year, publication_date=None, doi=normalize_doi(value("doi")), url=f"https://openreview.net/forum?id={quote(identifier)}", open_access_pdf_url=pdf, is_open_access=bool(pdf), citation_count=0, venue=_text(value("venue")), source=self.name, external_id=identifier))
        return SourceSearchResult(self.name, query, tuple(papers), content_checksum(payload))


def normalize_doi(value: Any) -> str | None:
    text = _text(value).lower().strip()
    text = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", text)
    text = re.sub(r"^doi:\s*", "", text)
    return text.rstrip(" .") if text.startswith("10.") and "/" in text else None


def normalize_arxiv_id(value: Any) -> str | None:
    text = _text(value).strip()
    text = re.sub(r"^https?://arxiv\.org/(?:abs|pdf)/", "", text, flags=re.I)
    text = re.sub(r"(?i)^arxiv:", "", text).removesuffix(".pdf")
    text = re.sub(r"v\d+$", "", text)
    return text or None


def metadata_payload(paper: MetadataPaper) -> dict[str, Any]:
    return asdict(paper)


def _text(value: Any) -> str:
    return " ".join(str(value or "").split())


def _url(value: Any) -> str | None:
    text = _text(value)
    return text if text.startswith("https://") or text.startswith("http://") else None


def _date(value: Any) -> str | None:
    text = _text(value)
    match = re.match(r"^(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?", text)
    if not match:
        return None
    year, month, day = match.groups()
    return f"{year}-{int(month or 1):02d}-{int(day or 1):02d}"


def _year(value: Any) -> int | None:
    try:
        year = int(value)
        return year if 1000 <= year <= 3000 else None
    except (TypeError, ValueError):
        return None


def _count(value: Any) -> int:
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def _strip_markup(value: str) -> str:
    return _text(html.unescape(re.sub(r"<[^>]+>", " ", value)))


def _openalex_abstract(index: Any) -> str:
    if not isinstance(index, dict):
        return ""
    positions = [(int(position), str(token)) for token, values in index.items() for position in values]
    return " ".join(token for _, token in sorted(positions))


def _node_text(node: ET.Element, path: str, namespaces: dict[str, str]) -> str:
    found = node.find(path, namespaces)
    return _text(found.text if found is not None else "")
