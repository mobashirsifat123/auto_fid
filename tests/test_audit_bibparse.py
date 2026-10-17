"""Unit tests for the strict BibTeX / thebibliography parser (audit/bibparse.py).

Hermetic: no network, no filesystem. Fail-closed contract under test
(DESIGN.md §4.3): malformed entries become per-entry error records counted
against the audited artifact — the parser must never crash and never guess
a field value.
"""

from __future__ import annotations

from autoresearch.audit.bibparse import (
    BibAuthor,
    parse_bibtex,
    parse_thebibliography,
)

WELL_FORMED = """
@article{smith2020,
  author = {Smith, John and {de Bruijn}, N. G. and others},
  title = "A {Study} of \\emph{Fusion}",
  journal = {IEEE \\& Transactions},
  year = 2020,
  doi = {https://doi.org/10.1234/abc.5678},
}
@inproceedings(Jones2021,
  author = "Jones, B. and Doe, Alice",
  TITLE = {Masked {Pretraining}},
  booktitle = {NeurIPS},
  year = {2021},
)
@misc{arnold2022,
  author = {Arnold, K.},
  title = {Notes on Things},
  year = {about 2022},
  eprint = {2401.12345v2},
  archivePrefix = {arXiv},
}
"""


# ----------------------------------------------------------------------
# Well-formed input
# ----------------------------------------------------------------------


def test_parses_well_formed_entries() -> None:
    result = parse_bibtex(WELL_FORMED)
    assert [entry.key for entry in result.entries] == ["smith2020", "Jones2021", "arnold2022"]
    assert result.entry_count == 3
    assert result.error_count == 0


def test_entry_types_and_field_names_are_case_insensitive() -> None:
    result = parse_bibtex(WELL_FORMED)
    assert result.entries[0].entry_type == "article"
    assert result.entries[1].entry_type == "inproceedings"
    # "TITLE =" is read as the title field.
    assert result.entries[1].title == "Masked Pretraining"


def test_author_name_variants() -> None:
    result = parse_bibtex(WELL_FORMED)
    assert result.entries[0].authors == (
        BibAuthor(family="Smith", given="John"),
        BibAuthor(family="de Bruijn", given="N. G."),
        BibAuthor(family="others", given=""),
    )
    # Quoted author lists, "Given Family" order, paren-delimited entries.
    assert result.entries[1].authors == (
        BibAuthor(family="Jones", given="B."),
        BibAuthor(family="Doe", given="Alice"),
    )
    assert result.entries[2].authors == (BibAuthor(family="Arnold", given="K."),)


def test_latex_normalization_of_titles_and_venues() -> None:
    result = parse_bibtex(WELL_FORMED)
    assert result.entries[0].title == "A Study of Fusion"
    assert result.entries[0].venue == "IEEE & Transactions"


def test_venue_prefers_booktitle_for_proceedings() -> None:
    result = parse_bibtex(WELL_FORMED)
    assert result.entries[1].venue == "NeurIPS"


def test_year_extraction_from_numbers_strings_and_noise() -> None:
    result = parse_bibtex(WELL_FORMED)
    assert result.entries[0].year == 2020
    assert result.entries[1].year == 2021
    # "about 2022" still yields the four-digit year; garbage yields None.
    assert result.entries[2].year == 2022
    noisy = parse_bibtex("@article{a, title={T}, year={20x0}}")
    assert noisy.entries[0].year is None


def test_doi_normalization() -> None:
    result = parse_bibtex(WELL_FORMED)
    assert result.entries[0].doi == "10.1234/abc.5678"


def test_arxiv_id_from_eprint_and_url() -> None:
    result = parse_bibtex(WELL_FORMED)
    # eprint + archivePrefix, version suffix stripped.
    assert result.entries[2].arxiv_id == "2401.12345"
    from_url = parse_bibtex(
        "@misc{b, title={T}, url={https://arxiv.org/abs/1909.11942v3.pdf}}"
    )
    assert from_url.entries[0].arxiv_id == "1909.11942"


def test_string_macros_months_and_concatenation() -> None:
    source = """
    @string{jav = "Journal of Approximate Veracity"}
    @article{m1, journal = jav # { and } # "Studies", month = jan, year = 2019}
    """
    result = parse_bibtex(source)
    assert result.error_count == 0
    assert result.entries[0].venue == "Journal of Approximate Veracity and Studies"


def test_comment_and_preamble_entries_are_skipped() -> None:
    source = """
    @comment{not an entry}
    @preamble{"\\usepackage{foo}"}
    @article{real, title={Real}, year=2001}
    """
    result = parse_bibtex(source)
    assert [entry.key for entry in result.entries] == ["real"]
    assert result.error_count == 0


def test_garbage_between_entries_is_ignored() -> None:
    source = "stray text with an @ sign\n@article{ok, title={T}, year=2001}\nmore stray"
    result = parse_bibtex(source)
    assert [entry.key for entry in result.entries] == ["ok"]
    assert result.error_count == 0


# ----------------------------------------------------------------------
# Malformed input: errors are data, never crashes
# ----------------------------------------------------------------------


def test_truncated_entry_is_recorded_and_does_not_crash() -> None:
    # An unterminated entry at end of file: strict parser, no guessing.
    result = parse_bibtex("@article{broken, title={Unclosed and the file just ends")
    assert result.entries == ()
    assert result.errors[0].reason == "truncated_entry"
    assert result.errors[0].key is None


def test_missing_key_is_an_error_and_parsing_continues() -> None:
    source = """
    @article{, title={No Key}, year=2001}
    @article{good, title={Good}, year=2002}
    """
    result = parse_bibtex(source)
    assert [entry.key for entry in result.entries] == ["good"]
    assert [error.reason for error in result.errors] == ["missing_key"]


def test_duplicate_key_records_error_and_keeps_first_entry() -> None:
    source = """
    @article{dup, title={First}, year=2001}
    @article{dup, title={Second}, year=2002}
    """
    result = parse_bibtex(source)
    assert len(result.entries) == 1
    assert result.entries[0].title == "First"
    assert result.errors[0].reason == "duplicate_key"
    assert result.errors[0].key == "dup"


def test_unknown_macro_is_error_data_with_empty_field() -> None:
    source = "@article{m, journal = mystery_conf, title={T}, year=2001}"
    result = parse_bibtex(source)
    assert result.entries[0].venue == ""
    assert result.errors[0].reason == "unknown_macro"
    assert "mystery_conf" in result.errors[0].detail


def test_missing_field_value_is_error_data() -> None:
    result = parse_bibtex("@article{m, title = , year=2001}")
    assert result.entries[0].title == ""
    assert result.errors[0].reason == "missing_value"


def test_unterminated_quoted_value_is_error_data() -> None:
    result = parse_bibtex('@article{m, title = "Open, year = 2001}')
    assert result.errors[0].reason == "unterminated_value"
    assert result.errors[0].key == "m"
    assert result.entries[0].title == ""


def test_invalid_field_syntax_is_error_data() -> None:
    result = parse_bibtex("@article{m, !nonsense, title={T}, year=2001}")
    reasons = [error.reason for error in result.errors]
    assert "invalid_field" in reasons
    assert result.entries[0].title == "T"


def test_crossref_is_not_inline_expanded() -> None:
    source = """
    @inproceedings{child, title={Child}, crossref = {parent}}
    """
    result = parse_bibtex(source)
    assert len(result.entries) == 1
    assert result.entries[0].title == "Child"
    assert result.errors[0].reason == "crossref_not_expanded"
    assert result.errors[0].key == "child"


# ----------------------------------------------------------------------
# thebibliography extraction
# ----------------------------------------------------------------------


THEBIB = r"""
\begin{thebibliography}{10}
\bibitem{smith2020} Smith, J. and Jones, B. ``Adaptive fusion for robust pipelines.'' In \emph{NeurIPS}, 2020. arXiv:2001.01234
\bibitem{doe2021} Doe, A. "Masked Pretraining," \emph{JMLR}, vol. 22, 2021. https://doi.org/10.5555/111.222
\bibitem{plain} No delimiters here at all, 1999.
\end{thebibliography}
"""


def test_thebibliography_items_are_extracted() -> None:
    result = parse_thebibliography(THEBIB)
    assert [entry.key for entry in result.entries] == ["smith2020", "doe2021", "plain"]
    first, second = result.entries[0], result.entries[1]
    assert first.entry_type == "thebibliography_item"
    assert first.title == "Adaptive fusion for robust pipelines."
    assert first.authors == (BibAuthor(family="Smith", given="J."), BibAuthor(family="Jones", given="B."))
    assert first.year == 2020
    assert first.arxiv_id == "2001.01234"
    assert second.title == "Masked Pretraining"
    assert second.year == 2021
    assert second.doi == "10.5555/111.222"


def test_thebibliography_item_without_delimited_title_refuses_to_guess() -> None:
    result = parse_thebibliography(THEBIB)
    plain = result.entries[2]
    assert plain.title == ""
    assert plain.authors == ()
    reasons = [error.reason for error in result.errors]
    assert "title_not_delimited" in reasons


def test_thebibliography_missing_key_is_error_data() -> None:
    result = parse_thebibliography(
        "\\begin{thebibliography}{1}\n\\bibitem{} Anonymous, 2000.\n\\end{thebibliography}"
    )
    assert result.entries == ()
    assert result.errors[0].reason == "missing_key"


def test_thebibliography_duplicate_keys_across_environments() -> None:
    text = THEBIB + "\n" + THEBIB
    result = parse_thebibliography(text)
    assert len(result.entries) == 3
    assert sum(1 for e in result.errors if e.reason == "duplicate_key") == 3


def test_thebibliography_environment_without_items() -> None:
    result = parse_thebibliography("\\begin{thebibliography}{9}\n\\end{thebibliography}")
    assert result.entries == ()
    assert result.errors == ()


def test_text_without_thebibliography_is_empty() -> None:
    result = parse_thebibliography("Just an abstract, no bibliography.")
    assert result.entries == ()
    assert result.errors == ()
