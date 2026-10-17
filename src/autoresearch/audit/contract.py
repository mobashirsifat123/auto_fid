"""Input contract for the trust-audit harness (DESIGN.md §4, order 2).

Defines the bundle dataclasses (§4.2), the ``BaselineAdapter`` protocol, and
the two shipped adapters:

* :class:`GenericDirectoryAdapter` — heuristic role assignment for unknown
  systems per §4.1 (paper source / PDF / bibliography / run log / artifact /
  other). Every file the harness reads appears in the bundle manifest with
  a role, SHA-256 checksum, and byte size; files the adapter cannot classify
  are excluded from metric inputs and listed under ``excluded_files``
  (fail-closed: no unclassified byte is ever used as evidence, §4.1).
* :class:`AutoresearchAdapter` — first-class mapping of our manuscript
  output directory (``manuscript.md``/``.tex``, ``references.bib``,
  ``tables/``, ``evidence_bundle.json``, ``manifest.json``, run logs).

Fail-closed notes (§4.3 asymmetry):

* The Generic adapter has no run-log ground truth, so
  :meth:`GenericDirectoryAdapter.run_log_summary` returns ``None`` — the
  disclosure metric reports ``NO_GROUND_TRUTH`` (NOT_RUN), never a pass.
* Ground truth the adapter cannot read (e.g. a corrupt
  ``evidence_bundle.json``) yields ``None`` rather than a partially
  guessed summary.
* Unreadable text (non-UTF-8) is decoded with replacement characters; a PDF
  whose text cannot be extracted yields empty sections — artifact-side
  failures recorded, never crashes.

Deviations from DESIGN.md §4.2 (recorded rationale):

* ``PaperArtifactBundle`` carries two extra optional fields:
  ``ground_truth_tier`` (the §8.1 report schema requires it in the bundle
  block, and only the adapter knows the tier) and ``bibliography_errors``
  (§5.1 needs per-entry ``unparseable`` subreasons, which must survive from
  parser to metric without re-parsing). Both default to the §4.2 shape.
* ``BundleFile`` carries ``role_confidence`` (§4.2 adapter note: heuristic
  assignment quality is recorded per file). It is bundle-side only and is
  not serialized into the §8.1 report files array.
* ``BibEntry``/``BibAuthor``/``BibParseError`` are re-exported from
  ``bibparse`` (see that module's docstring for the acyclicity rationale).
* ``AutoresearchAdapter.reproduction_commands`` returns an empty tuple in
  v1: the recorded worker commands live in the experiment-report database
  rows, not in the file bundle; the harness is deliberately
  store-independent (§3 layering). The reproducibility metric will report
  this honestly as ``NOT_RUNNABLE``/``no_commands`` until a file-side
  command record exists.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from autoresearch.audit.bibparse import (
    BibAuthor,
    BibEntry,
    BibParseError,
    _latex_text,
    has_thebibliography,
    parse_bibtex,
    parse_thebibliography,
)
from autoresearch.domain.models import content_checksum
from autoresearch.ingestion.document_parser import DocumentParser

__all__ = [
    "AutoresearchAdapter",
    "BaselineAdapter",
    "BibAuthor",
    "BibEntry",
    "BibParseError",
    "BundleFile",
    "FileRole",
    "GenericDirectoryAdapter",
    "PaperArtifactBundle",
    "ReproductionCommand",
    "RunLogSummary",
    "TextSection",
]

ADAPTER_VERSION = "1.0.0"

# Known manuscript file names (Autoresearch output, §4.2 adapter 1).
_AUTORESEARCH_PAPER = {
    "manuscript.md", "manuscript_deterministic.md", "manuscript.tex",
}
_AUTORESEARCH_ARTIFACTS = {
    "evidence_bundle.json", "manifest.json", "writing_audit.json",
}
_AUTORESEARCH_DIRECTORIES = {"tables", "artifacts", "experiments", "figures"}

# Preferred paper source names, in priority order.
_PREFERRED_PAPER = (
    "manuscript.md", "paper.md", "manuscript.tex", "paper.tex",
    "manuscript_deterministic.md",
)

# Reproduction-command verbs accepted from README fenced code blocks
# (Generic adapter, §5.6: "any executable + README run instructions the
# adapter can parse, else empty").
_RUN_VERBS = {"python", "python3", "pip", "pip3", "bash", "sh", "make", "docker", "uv"}

_MARKDOWN_HEADING = re.compile(r"#{1,6}\s+(.+?)\s*#*\s*$")
_LATEX_SECTION = re.compile(r"\\section\*?\s*\{")
_LATEX_ABSTRACT = re.compile(r"\\begin\{abstract\}")
_README_FENCE = re.compile(r"```[^\n]*\n(.*?)```", re.DOTALL)


class FileRole(StrEnum):
    PAPER_SOURCE = "paper_source"
    PAPER_PDF = "paper_pdf"
    BIBLIOGRAPHY = "bibliography"
    RUN_LOG = "run_log"
    ARTIFACT = "artifact"
    OTHER = "other"


@dataclass(frozen=True)
class BundleFile:
    """One manifest entry: role, path, sha256, size (§4.1)."""

    path: str
    role: FileRole
    sha256: str
    bytes: int
    role_confidence: float = 1.0


@dataclass(frozen=True)
class TextSection:
    """One paper text section: name + text + source file reference."""

    section: str
    text: str
    source_file: str


@dataclass(frozen=True)
class PaperArtifactBundle:
    """The audited input bundle (§4.2)."""

    system_id: str
    adapter_name: str
    adapter_version: str
    root: str
    files: tuple[BundleFile, ...] = ()
    paper_text: tuple[TextSection, ...] = ()
    bibliography: tuple[BibEntry, ...] = ()
    excluded_files: tuple[str, ...] = ()
    ground_truth_tier: str = "none"
    bibliography_errors: tuple[BibParseError, ...] = ()


@dataclass(frozen=True)
class RunLogSummary:
    """Run-log ground truth summary (§4.2). ``None`` fields are unknown."""

    source_files: tuple[str, ...] = ()
    runs_total: int | None = None
    runs_failed: int | None = None
    seeds_executed: tuple[int, ...] | None = None
    stages_executed: tuple[str, ...] = ()
    stages_failed_or_skipped: tuple[str, ...] = ()
    error_events: tuple[str, ...] = ()


@dataclass(frozen=True)
class ReproductionCommand:
    """One adapter-declared re-run command, pinned to its source file."""

    command: tuple[str, ...]
    source_file: str
    description: str = ""


class BaselineAdapter(Protocol):
    """Adapter contract (§4.2). A missing mapping yields ``None``/empty —
    never a weakened metric (fail-closed)."""

    system_id: str
    adapter_version: str

    def collect(self, root: Path) -> PaperArtifactBundle: ...
    def run_log_summary(self, bundle: PaperArtifactBundle) -> RunLogSummary | None: ...
    def reproduction_commands(self, bundle: PaperArtifactBundle) -> tuple[ReproductionCommand, ...]: ...


# ----------------------------------------------------------------------
# Shared file-walking, role assignment, bundle staging
# ----------------------------------------------------------------------


def _iter_bundle_files(root: Path) -> Iterator[Path]:
    """Deterministic walk; hidden entries are skipped (cache/VCS noise)."""
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_file():
            yield path


def _file_record(root: Path, path: Path, role: FileRole, confidence: float) -> BundleFile:
    payload = path.read_bytes()
    return BundleFile(
        path=str(path.relative_to(root)),
        role=role,
        sha256=content_checksum(payload),
        bytes=len(payload),
        role_confidence=confidence,
    )


def _read_text(path: Path) -> str:
    """Read UTF-8 text; decode errors become replacement characters."""
    payload = path.read_bytes()
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError:
        return payload.decode("utf-8", errors="replace")


def _stage_bundle(
    root: Path,
    system_id: str,
    adapter_name: str,
    assign: Callable[[Path], tuple[FileRole, float]],
    ground_truth_tier: str = "none",
) -> PaperArtifactBundle:
    """Walk ``root``, assign roles, extract paper text + bibliography."""
    files: list[BundleFile] = []
    for path in _iter_bundle_files(root):
        role, confidence = assign(path.relative_to(root))
        files.append(_file_record(root, path, role, confidence))
    excluded = tuple(record.path for record in files if record.role == FileRole.OTHER)
    paper_text = _paper_text(root, files)
    bibliography, bibliography_errors = _bibliography(root, files)
    return PaperArtifactBundle(
        system_id=system_id,
        adapter_name=adapter_name,
        adapter_version=ADAPTER_VERSION,
        root=str(root),
        files=tuple(files),
        paper_text=tuple(paper_text),
        bibliography=tuple(bibliography),
        excluded_files=excluded,
        ground_truth_tier=ground_truth_tier,
        bibliography_errors=tuple(bibliography_errors),
    )


# ----------------------------------------------------------------------
# Paper text extraction
# ----------------------------------------------------------------------


def _paper_text(root: Path, files: list[BundleFile]) -> list[TextSection]:
    sources = [record for record in files if record.role == FileRole.PAPER_SOURCE]
    if sources:
        # Exactly one primary source file provides paper text: bundles that
        # carry the same manuscript in two formats (our markdown + latex
        # twins) must not double-count tokens in the text metrics, and a
        # missing secondary file yields missing text (fail-closed), never a
        # fabricated count. Non-primary sources stay manifested with their
        # role; their thebibliography is still harvested (see _bibliography).
        primary = _prefer_paper_files(sources)[0]
        text = _read_text(root / primary.path)
        if primary.path.lower().endswith(".tex"):
            return _latex_sections(text, primary.path)
        return _markdown_sections(text, primary.path)
    pdfs = [record for record in files if record.role == FileRole.PAPER_PDF]
    if pdfs:
        return _pdf_sections(root, min(pdfs, key=lambda record: record.path).path)
    return []


def _prefer_paper_files(sources: list[BundleFile]) -> list[BundleFile]:
    ranked: list[BundleFile] = []
    for name in _PREFERRED_PAPER:
        ranked.extend(record for record in sources if record.path == name)
    remaining = sorted(
        (record for record in sources if record not in ranked),
        key=lambda record: (0 if record.path.lower().endswith(".md") else 1, record.path),
    )
    return [*ranked, *remaining]


def _markdown_sections(text: str, source_file: str) -> list[TextSection]:
    """Split markdown into heading-delimited sections (code-fence aware)."""
    sections: list[TextSection] = []
    current_name = "document"
    current: list[str] = []
    fenced = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fenced = not fenced
            current.append(line)
            continue
        heading = None if fenced else _markdown_heading(stripped)
        if heading is not None:
            if any(part.strip() for part in current):
                sections.append(TextSection(current_name, "\n".join(current).strip(), source_file))
            current_name = heading
            current = []
            continue
        current.append(line)
    if any(part.strip() for part in current):
        sections.append(TextSection(current_name, "\n".join(current).strip(), source_file))
    return sections


def _markdown_heading(line: str) -> str | None:
    if not line.startswith("#"):
        return None
    match = _MARKDOWN_HEADING.match(line)
    return match.group(1).strip() if match else None


def _latex_sections(text: str, source_file: str) -> list[TextSection]:
    """Split LaTeX into \\section / abstract sections; comment lines dropped."""
    cleaned = "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("%")
    )
    markers: list[tuple[int, str]] = []
    for match in _LATEX_SECTION.finditer(cleaned):
        markers.append((match.start(), _latex_group(cleaned, match.end())))
    abstract = _LATEX_ABSTRACT.search(cleaned)
    if abstract:
        markers.append((abstract.start(), "abstract"))
    markers.sort()
    if not markers:
        return [TextSection("document", cleaned.strip(), source_file)] if cleaned.strip() else []
    sections: list[TextSection] = []
    head = cleaned[:markers[0][0]].strip()
    if head:
        sections.append(TextSection("preamble", head, source_file))
    for index, (start, name) in enumerate(markers):
        end = markers[index + 1][0] if index + 1 < len(markers) else len(cleaned)
        sections.append(TextSection(name, cleaned[start:end].strip(), source_file))
    return sections


def _latex_group(text: str, start: int) -> str:
    """Read the brace group content starting just before ``start``."""
    depth = 0
    for index in range(start - 1, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return _latex_text(text[start:index])
    return "section"


def _pdf_sections(root: Path, relative: str) -> list[TextSection]:
    """Extract PDF text page-by-page; failure yields empty sections (§4.3)."""
    try:
        parsed = DocumentParser().parse_seed(str((root / relative).resolve()))
        return [TextSection(f"page-{page.page}", page.text, relative) for page in parsed.pages]
    except Exception:  # noqa: BLE001 — artifact-side extraction failure is recorded, not crashed
        return []


# ----------------------------------------------------------------------
# Bibliography collection
# ----------------------------------------------------------------------


def _bibliography(
    root: Path, files: list[BundleFile],
) -> tuple[list[BibEntry], list[BibParseError]]:
    entries: list[BibEntry] = []
    errors: list[BibParseError] = []
    for record in files:
        if record.role == FileRole.BIBLIOGRAPHY:
            result = parse_bibtex(_read_text(root / record.path))
            entries.extend(result.entries)
            errors.extend(result.errors)
        elif record.role == FileRole.PAPER_SOURCE and record.path.lower().endswith(".tex"):
            text = _read_text(root / record.path)
            if has_thebibliography(text):
                result = parse_thebibliography(text)
                entries.extend(result.entries)
                errors.extend(result.errors)
    return entries, errors


# ----------------------------------------------------------------------
# GenericDirectoryAdapter (§4.1 heuristic roles)
# ----------------------------------------------------------------------


class GenericDirectoryAdapter:
    """Heuristic role assignment for unknown systems (§4.2 adapter 2)."""

    system_id = "generic"
    adapter_version = ADAPTER_VERSION

    def collect(self, root: Path) -> PaperArtifactBundle:
        return _stage_bundle(root, self.system_id, "generic", self._role)

    def run_log_summary(self, bundle: PaperArtifactBundle) -> RunLogSummary | None:
        # No run-log ground truth: disclosure must be NOT_RUN, never pass (§4.3).
        return None

    def reproduction_commands(self, bundle: PaperArtifactBundle) -> tuple[ReproductionCommand, ...]:
        commands: list[ReproductionCommand] = []
        for record in bundle.files:
            if Path(record.path).name.lower().startswith("readme") and record.path.lower().endswith((".md", ".txt")):
                commands.extend(_readme_commands(Path(bundle.root), record.path))
        return tuple(commands)

    @staticmethod
    def _role(relative: Path) -> tuple[FileRole, float]:
        parts = relative.parts
        first = parts[0].lower() if parts else ""
        name = relative.name.lower()
        if first in {"logs", "log"}:
            return FileRole.RUN_LOG, 0.7
        if first in {"artifacts", "artifact"}:
            return FileRole.ARTIFACT, 0.7
        if relative.suffix.lower() == ".bib":
            return FileRole.BIBLIOGRAPHY, 1.0
        if relative.suffix.lower() == ".pdf":
            confidence = 1.0 if name in {"paper.pdf", "manuscript.pdf"} else 0.9
            return FileRole.PAPER_PDF, confidence
        if name.startswith("readme"):
            # Run instructions: used only for reproduction hints, never as
            # metric input text (therefore role OTHER / excluded).
            return FileRole.OTHER, 1.0
        if relative.suffix.lower() in {".tex", ".md"} and len(parts) == 1:
            # §4.1: paper sources live at the bundle root.
            confidence = 1.0 if name in {"paper.md", "paper.tex", "manuscript.md", "manuscript.tex"} else 0.9
            return FileRole.PAPER_SOURCE, confidence
        if relative.suffix.lower() in {".tex", ".md"}:
            return FileRole.OTHER, 0.4
        return FileRole.OTHER, 0.5


def _readme_commands(root: Path, relative: str) -> list[ReproductionCommand]:
    """Parse fenced code blocks whose lines start with a run verb."""
    commands: list[ReproductionCommand] = []
    for block in _README_FENCE.findall(_read_text(root / relative)):
        for line in block.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                tokens = shlex.split(stripped)
            except ValueError:
                continue
            if tokens and tokens[0] in _RUN_VERBS:
                commands.append(ReproductionCommand(tuple(tokens), relative))
    return commands


# ----------------------------------------------------------------------
# AutoresearchAdapter (our manuscript output directory)
# ----------------------------------------------------------------------


class AutoresearchAdapter:
    """First-class adapter for AutoResearch manuscript output (§4.2 adapter 1)."""

    system_id = "autoresearch"
    adapter_version = ADAPTER_VERSION

    def collect(self, root: Path) -> PaperArtifactBundle:
        return _stage_bundle(root, self.system_id, "autoresearch", self._role, ground_truth_tier="structured")

    def run_log_summary(self, bundle: PaperArtifactBundle) -> RunLogSummary | None:
        record = next((f for f in bundle.files if f.path == "evidence_bundle.json"), None)
        if record is None or record.role != FileRole.ARTIFACT:
            return None
        try:
            data = json.loads((Path(bundle.root) / "evidence_bundle.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # Unreadable ground truth: no summary, never a guessed one (§4.3).
            return None
        if not isinstance(data, dict):
            return None
        analysis = data.get("analysis")
        rows = analysis.get("seed_level_results") if isinstance(analysis, dict) else None
        seeds: tuple[int, ...] | None = None
        runs_total: int | None = None
        if isinstance(rows, list):
            collected: set[int] = set()
            for row in rows:
                if isinstance(row, dict):
                    try:
                        collected.add(int(row["seed"]))
                    except (KeyError, TypeError, ValueError):
                        continue
            seeds = tuple(sorted(collected))
            runs_total = len(rows)
        lineage = data.get("lineage") if isinstance(data.get("lineage"), dict) else {}
        stages = []
        if lineage.get("gap_id"):
            stages.append("review")
        if lineage.get("portfolio_id") or lineage.get("candidate_id"):
            stages.append("novelty")
        if lineage.get("spec_id"):
            stages.append("experiment")
        if lineage.get("analysis_id"):
            stages.append("analysis")
        if lineage.get("report_id"):
            stages.append("report")
        return RunLogSummary(
            source_files=("evidence_bundle.json",),
            runs_total=runs_total,
            runs_failed=None,
            seeds_executed=seeds,
            stages_executed=tuple(stages),
            stages_failed_or_skipped=(),
            error_events=(),
        )

    def reproduction_commands(self, bundle: PaperArtifactBundle) -> tuple[ReproductionCommand, ...]:
        # v1: worker commands live in the experiment-report database rows,
        # not in the file bundle (store-independent harness, §3 layering).
        return ()

    @staticmethod
    def _role(relative: Path) -> tuple[FileRole, float]:
        parts = relative.parts
        first = parts[0].lower() if parts else ""
        name = relative.name.lower()
        if first in _AUTORESEARCH_DIRECTORIES:
            return FileRole.ARTIFACT, 1.0
        if first in {"logs", "log"} or relative.suffix.lower() == ".log":
            return FileRole.RUN_LOG, 1.0
        if name in _AUTORESEARCH_PAPER:
            return FileRole.PAPER_SOURCE, 1.0
        if name == "manuscript.pdf":
            return FileRole.PAPER_PDF, 1.0
        if name == "references.bib":
            return FileRole.BIBLIOGRAPHY, 1.0
        if name in _AUTORESEARCH_ARTIFACTS:
            return FileRole.ARTIFACT, 1.0
        if len(parts) == 1 and relative.suffix.lower() in {".tex", ".md"}:
            return FileRole.PAPER_SOURCE, 0.7
        return FileRole.OTHER, 0.5
