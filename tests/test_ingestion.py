from pathlib import Path

from pypdf import PdfWriter

from autoresearch.ingestion.chunking import chunk_document
from autoresearch.ingestion.document_parser import DocumentParser, ParsedDocument, ParsedPage


def test_pdf_parser_preserves_checksum_and_page_count(tmp_path):
    path = tmp_path / "seed.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_metadata({"/Title": "Seed title"})
    with path.open("wb") as handle:
        writer.write(handle)
    parsed = DocumentParser().parse_seed(path)
    assert parsed.title == "Seed title"
    assert len(parsed.pages) == 1
    assert len(parsed.checksum) == 64


def test_seed_parser_accepts_doi_and_https_url():
    parser = DocumentParser()
    assert parser.parse_seed("10.1000/example").doi == "10.1000/example"
    assert parser.parse_seed("https://example.org/paper").input_type == "url"


def test_chunking_is_page_aware_and_deterministic():
    document = ParsedDocument("pdf", "x.pdf", "Title", (ParsedPage(1, "METHODS\n" + "Sentence. " * 200),), "checksum")
    first = chunk_document(document, target_characters=500, overlap_characters=50)
    second = chunk_document(document, target_characters=500, overlap_characters=50)
    assert first == second
    assert len(first) > 1
    assert all(chunk.page_start == 1 for chunk in first)
