"""Unit tests for the audit input contract and adapters (audit/contract.py).

Hermetic: every bundle is built under ``tmp_path``; PDFs are generated with
pypdf; no network. Fail-closed rules under test (DESIGN.md §4.1/§4.3): every
read file is manifested with role+sha256+size, unclassified files are excluded
from metric inputs, and missing ground truth yields ``None`` (NOT_RUN), never
a pass.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from pypdf import PdfWriter

from autoresearch.audit.contract import (
    AutoresearchAdapter,
    FileRole,
    GenericDirectoryAdapter,
)
from autoresearch.audit.report import bundle_to_report_input

PAPER_MD = """# Abstract
We report a 92.3% improvement over the baseline.

# Introduction
Prior work cites 5 studies.

# Results

```python
# this fenced line is not a heading
value = 1
```
We observe gains of 0.12.
"""

REFERENCES_BIB = """
@article{smith2020,
  author = {Smith, John},
  title = "Adaptive Fusion",
  journal = {Journal of Testing},
  year = 2020,
  doi = {10.1234/abc.5678},
}
@inproceedings{jones2021,
  author = {Jones, B.},
  title = {Masked {Pretraining}},
  booktitle = {NeurIPS},
  year = 2021,
}
@misc{broken
"""

PAPER_TEX = r"""
\documentclass{article}
\begin{abstract}
We study fusion.
\end{abstract}
\section{Introduction}
Prior work \cite{smith2020} is limited.
% a comment line
\section{Results}
Gains of 12 points.
\begin{thebibliography}{9}
\bibitem{tex2020} Smith, J. ``Fusion via attention.'' NeurIPS, 2020.
\end{thebibliography}
"""

README_MD = """# Reproduction

```bash
python train.py --seed 1
pip install -e .
echo not a run command
```
"""

EVIDENCE_BUNDLE = {
    "lineage": {
        "gap_id": "gap-1", "portfolio_id": "port-1", "candidate_id": "cand-1",
        "spec_id": "spec-1", "analysis_id": "ana-1", "report_id": "rep-1",
    },
    "analysis": {
        "seed_level_results": [
            {"seed": 1, "accuracy": 0.90},
            {"seed": 2, "accuracy": 0.91},
            {"seed": 3, "accuracy": 0.89},
        ],
    },
}


@pytest.fixture
def generic_root(tmp_path: Path) -> Path:
    root = tmp_path / "generic-paper"
    root.mkdir()
    (root / "paper.md").write_text(PAPER_MD, encoding="utf-8")
    (root / "references.bib").write_text(REFERENCES_BIB, encoding="utf-8")
    (root / "notes.txt").write_text("unclassified scratch", encoding="utf-8")
    (root / "README.md").write_text(README_MD, encoding="utf-8")
    (root / "logs").mkdir()
    (root / "logs" / "run.log").write_text("seed=1 ok\n", encoding="utf-8")
    (root / "artifacts").mkdir()
    (root / "artifacts" / "metrics.json").write_text('{"accuracy": 0.9}', encoding="utf-8")
    (root / ".cache").mkdir()
    (root / ".cache" / "tmp.bin").write_bytes(b"\x00")
    return root


# ----------------------------------------------------------------------
# GenericDirectoryAdapter: roles and manifest
# ----------------------------------------------------------------------


def test_generic_roles_follow_section_4_1(generic_root: Path) -> None:
    bundle = GenericDirectoryAdapter().collect(generic_root)
    roles = {record.path: record.role for record in bundle.files}
    assert roles["paper.md"] == FileRole.PAPER_SOURCE
    assert roles["references.bib"] == FileRole.BIBLIOGRAPHY
    assert roles["logs/run.log"] == FileRole.RUN_LOG
    assert roles["artifacts/metrics.json"] == FileRole.ARTIFACT
    assert roles["notes.txt"] == FileRole.OTHER
    assert roles["README.md"] == FileRole.OTHER


def test_generic_manifest_records_sha256_and_size_for_every_read_file(generic_root: Path) -> None:
    root = generic_root
    bundle = GenericDirectoryAdapter().collect(root)
    by_path = {record.path: record for record in bundle.files}
    manifest_paths = set(by_path) | set(bundle.excluded_files)
    # Every non-hidden file under the root is manifested exactly once.
    walked = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and not any(part.startswith(".") for part in path.relative_to(root).parts)
    }
    assert manifest_paths == walked
    record = by_path["paper.md"]
    payload = (root / "paper.md").read_bytes()
    assert record.sha256 == hashlib.sha256(payload).hexdigest()
    assert record.bytes == len(payload)


def test_generic_unclassified_files_are_excluded_from_metric_inputs(generic_root: Path) -> None:
    bundle = GenericDirectoryAdapter().collect(generic_root)
    assert set(bundle.excluded_files) == {"notes.txt", "README.md"}
    source_files = {section.source_file for section in bundle.paper_text}
    assert source_files == {"paper.md"}


def test_generic_paper_text_markdown_sections_fence_aware(generic_root: Path) -> None:
    bundle = GenericDirectoryAdapter().collect(generic_root)
    names = [section.section for section in bundle.paper_text]
    assert names == ["Abstract", "Introduction", "Results"]
    results = bundle.paper_text[2]
    assert "# this fenced line is not a heading" in results.text
    assert "0.12" in results.text


def test_generic_bibliography_parsed_with_errors_as_data(generic_root: Path) -> None:
    bundle = GenericDirectoryAdapter().collect(generic_root)
    assert [entry.key for entry in bundle.bibliography] == ["smith2020", "jones2021"]
    assert bundle.bibliography[0].doi == "10.1234/abc.5678"
    # The truncated third entry is artifact-side data, never a crash (§4.3).
    assert [error.reason for error in bundle.bibliography_errors] == ["truncated_entry"]


def test_generic_run_log_summary_is_none_no_ground_truth(generic_root: Path) -> None:
    bundle = GenericDirectoryAdapter().collect(generic_root)
    assert GenericDirectoryAdapter().run_log_summary(bundle) is None


def test_generic_reproduction_commands_from_readme_fences(generic_root: Path) -> None:
    bundle = GenericDirectoryAdapter().collect(generic_root)
    commands = GenericDirectoryAdapter().reproduction_commands(bundle)
    assert [tuple(command.command) for command in commands] == [
        ("python", "train.py", "--seed", "1"),
        ("pip", "install", "-e", "."),
    ]
    assert all(command.source_file == "README.md" for command in commands)


def test_generic_hidden_files_are_never_manifested(generic_root: Path) -> None:
    bundle = GenericDirectoryAdapter().collect(generic_root)
    assert all(not record.path.startswith(".cache") for record in bundle.files)


def test_generic_nested_markdown_is_not_paper_source(generic_root: Path) -> None:
    root = generic_root
    (root / "docs").mkdir()
    (root / "docs" / "notes.md").write_text("# Notes\nscratch", encoding="utf-8")
    bundle = GenericDirectoryAdapter().collect(root)
    by_path = {record.path: record for record in bundle.files}
    assert by_path["docs/notes.md"].role == FileRole.OTHER
    assert "docs/notes.md" in bundle.excluded_files


# ----------------------------------------------------------------------
# PDF-only bundle (fallback path, §4.1)
# ----------------------------------------------------------------------


def test_pdf_only_bundle_uses_pdf_text_mode(generic_root: Path) -> None:
    root = generic_root.parent / "pdf-only"
    root.mkdir()
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with (root / "paper.pdf").open("wb") as handle:
        writer.write(handle)
    bundle = GenericDirectoryAdapter().collect(root)
    assert bundle.files[0].role == FileRole.PAPER_PDF
    assert [section.section for section in bundle.paper_text] == ["page-1"]
    report_input = bundle_to_report_input(bundle)
    assert report_input.bundle.text_extraction.mode == "pdf_text"


def test_corrupt_pdf_yields_empty_text_not_crash(generic_root: Path) -> None:
    root = generic_root.parent / "corrupt-pdf"
    root.mkdir()
    (root / "paper.pdf").write_bytes(b"not a pdf at all")
    bundle = GenericDirectoryAdapter().collect(root)
    assert bundle.paper_text == ()
    assert bundle.files[0].role == FileRole.PAPER_PDF


# ----------------------------------------------------------------------
# LaTeX source with embedded thebibliography
# ----------------------------------------------------------------------


def test_latex_source_sections_and_embedded_bibliography(generic_root: Path) -> None:
    root = generic_root.parent / "latex-paper"
    root.mkdir()
    (root / "paper.tex").write_text(PAPER_TEX, encoding="utf-8")
    bundle = GenericDirectoryAdapter().collect(root)
    names = [section.section for section in bundle.paper_text]
    assert names == ["preamble", "abstract", "Introduction", "Results"]
    assert "We study fusion." in bundle.paper_text[1].text
    assert [entry.key for entry in bundle.bibliography] == ["tex2020"]
    assert bundle.bibliography[0].year == 2020


# ----------------------------------------------------------------------
# AutoresearchAdapter (our manuscript output directory)
# ----------------------------------------------------------------------


@pytest.fixture
def autoresearch_root(tmp_path: Path) -> Path:
    root = tmp_path / "manuscripts" / "manuscript-1"
    root.mkdir(parents=True)
    (root / "manuscript.md").write_text(PAPER_MD, encoding="utf-8")
    (root / "manuscript.tex").write_text(PAPER_TEX, encoding="utf-8")
    (root / "references.bib").write_text(REFERENCES_BIB, encoding="utf-8")
    (root / "evidence_bundle.json").write_text(json.dumps(EVIDENCE_BUNDLE), encoding="utf-8")
    (root / "manifest.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    (root / "tables").mkdir()
    (root / "tables" / "results.md").write_text("| metric | value |\n", encoding="utf-8")
    (root / "logs").mkdir()
    (root / "logs" / "execution.log").write_text("run ok\n", encoding="utf-8")
    (root / "stray.dat").write_bytes(b"\x01\x02")
    return root


def test_autoresearch_roles_for_known_layout(autoresearch_root: Path) -> None:
    bundle = AutoresearchAdapter().collect(autoresearch_root)
    roles = {record.path: record.role for record in bundle.files}
    assert roles["manuscript.md"] == FileRole.PAPER_SOURCE
    assert roles["manuscript.tex"] == FileRole.PAPER_SOURCE
    assert roles["references.bib"] == FileRole.BIBLIOGRAPHY
    assert roles["evidence_bundle.json"] == FileRole.ARTIFACT
    assert roles["manifest.json"] == FileRole.ARTIFACT
    assert roles["tables/results.md"] == FileRole.ARTIFACT
    assert roles["logs/execution.log"] == FileRole.RUN_LOG
    assert roles["stray.dat"] == FileRole.OTHER
    assert bundle.excluded_files == ("stray.dat",)


def test_autoresearch_ground_truth_tier_is_structured(autoresearch_root: Path) -> None:
    bundle = AutoresearchAdapter().collect(autoresearch_root)
    assert bundle.ground_truth_tier == "structured"


def test_autoresearch_paper_text_prefers_markdown_manuscript(autoresearch_root: Path) -> None:
    bundle = AutoresearchAdapter().collect(autoresearch_root)
    sources = {section.source_file for section in bundle.paper_text}
    # Markdown manuscript is preferred; the .tex is still a manifested
    # paper_source but its thebibliography feeds the bibliography.
    assert "manuscript.md" in sources
    assert "manuscript.tex" not in sources
    assert {entry.key for entry in bundle.bibliography} == {"smith2020", "jones2021", "tex2020"}


def test_autoresearch_run_log_summary_from_evidence_bundle(autoresearch_root: Path) -> None:
    bundle = AutoresearchAdapter().collect(autoresearch_root)
    summary = AutoresearchAdapter().run_log_summary(bundle)
    assert summary is not None
    assert summary.source_files == ("evidence_bundle.json",)
    assert summary.runs_total == 3
    assert summary.seeds_executed == (1, 2, 3)
    assert summary.stages_executed == ("review", "novelty", "experiment", "analysis", "report")


def test_autoresearch_run_log_summary_none_without_evidence_bundle(autoresearch_root: Path) -> None:
    root = autoresearch_root
    (root / "evidence_bundle.json").unlink()
    bundle = AutoresearchAdapter().collect(root)
    assert AutoresearchAdapter().run_log_summary(bundle) is None


def test_autoresearch_run_log_summary_none_for_corrupt_evidence_bundle(autoresearch_root: Path) -> None:
    root = autoresearch_root
    (root / "evidence_bundle.json").write_text("{not json", encoding="utf-8")
    bundle = AutoresearchAdapter().collect(root)
    assert AutoresearchAdapter().run_log_summary(bundle) is None


def test_autoresearch_reproduction_commands_empty_in_v1(autoresearch_root: Path) -> None:
    bundle = AutoresearchAdapter().collect(autoresearch_root)
    assert AutoresearchAdapter().reproduction_commands(bundle) == ()


# ----------------------------------------------------------------------
# Bundle → report input conversion (report.py integration helper)
# ----------------------------------------------------------------------


def test_bundle_to_report_input_serializes_manifest(autoresearch_root: Path) -> None:
    bundle = AutoresearchAdapter().collect(autoresearch_root)
    report_input = bundle_to_report_input(bundle)
    payload = report_input.bundle
    assert payload.system_id == "autoresearch"
    assert payload.adapter_name == "autoresearch"
    assert payload.text_extraction.mode == "markdown_source"
    assert payload.ground_truth_tier == "structured"
    roles = {f.path: f.role for f in payload.files}
    assert roles["manuscript.md"] == "paper_source"
    assert roles["tables/results.md"] == "artifact"
    assert payload.excluded_files == ("stray.dat",)
    assert payload.text_extraction.quality["sections"] == len(bundle.paper_text)
    assert payload.text_extraction.quality["source_files"] == 2
    assert all(len(f.sha256) == 64 for f in payload.files)
