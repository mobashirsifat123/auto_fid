from autoresearch.papers.deduplication import SearchProvenance, SourcedCandidate, deduplicate, deduplication_key, normalized_title
from autoresearch.papers.metadata_sources import MetadataPaper
from autoresearch.papers.search_pipeline import StudySearchPipeline


def _source(name, rank=1):
    return SearchProvenance(name, "topic", "query", rank, f"checksum-{name}")


def test_deduplicates_doi_and_merges_open_access_metadata():
    first = MetadataPaper("A Study", "short", ("A",), 2024, doi="10.1/x", source="crossref")
    second = MetadataPaper("A Study", "a much longer abstract", ("A",), 2024, doi="10.1/x", open_access_pdf_url="https://example/x.pdf", is_open_access=True, source="openalex")
    result = deduplicate([SourcedCandidate(first, _source("crossref")), SourcedCandidate(second, _source("openalex"))])
    assert len(result) == 1
    assert result[0].paper.abstract == "a much longer abstract"
    assert result[0].paper.is_open_access
    assert len(result[0].provenance) == 2


def test_title_author_year_fallback_is_stable():
    left = MetadataPaper("Attention: A Study!", authors=("Jane Doe",), year=2025)
    right = MetadataPaper("attention a study", authors=("Jane Doe",), year=2025)
    assert normalized_title(left.title) == normalized_title(right.title)
    assert deduplication_key(left) == deduplication_key(right)


def test_paper_storage_ids_are_deterministic_and_project_scoped():
    paper = MetadataPaper("Shared paper", authors=("Researcher",), year=2026)
    first = StudySearchPipeline._paper_record("project-a", "run-a", paper)
    repeated = StudySearchPipeline._paper_record("project-a", "run-b", paper)
    independent = StudySearchPipeline._paper_record("project-b", "run-c", paper)
    assert first.id == repeated.id
    assert first.id != independent.id
