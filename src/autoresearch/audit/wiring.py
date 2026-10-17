"""Wiring layer: contract bundle → six auditors → metric sections.

Integration rule (2026-10-16, supervisor session): every metric section is
constructed through :meth:`report.MetricSection.complete` / ``not_run`` so
rate derivation stays centralized in ``report.RATE_SPECS``. Auditor results
carry counts + details; wiring maps them. Any count-key drift against
``RATE_SPECS`` raises ``AuditRefusal`` here (fail-closed) instead of silently
dropping a rate.

Seam for the runner (order 11): :func:`audit_bundle` takes a
:class:`contract.PaperArtifactBundle` plus injected dependencies and returns
:class:`AuditOutcome`. Dependencies are optional; absence yields an honest
NOT_RUN for the affected metric — never a passing score.

Section identity: the auditors each define their own minimal ``TextSection``
(``name``/``text``); contract sections carry ``section``/``text``/``source_file``
and are mapped by value (no shared mutable state).
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.audit import citations as citations_mod
from autoresearch.audit import claims as claims_mod
from autoresearch.audit import contract as contract_mod
from autoresearch.audit import disclosure as disclosure_mod
from autoresearch.audit import fetchcache as fetchcache_mod
from autoresearch.audit import numbers as numbers_mod
from autoresearch.audit import overclaim as overclaim_mod
from autoresearch.audit import report as report_mod
from autoresearch.audit import reproducibility as reproducibility_mod

__all__ = ["AuditOutcome", "MetricRunnerDeps", "audit_bundle", "map_text_sections"]


@dataclass(frozen=True)
class MetricRunnerDeps:
    """Injected runtime dependencies (network, judge, ground truth).

    - ``citation_sources``/``citation_cache``: resolved-source adapters + the
      checksummed fetch cache. Without both, the citation metric is NOT_RUN
      (harness-side missing dependency, excluded from pooling).
    - ``entailment_judge``: optional advisory LLM pass for overclaim;
      ``llm_assisted`` auto-flags when present.
    - ``recomputers``: metric recomputation registry (defaults to the standard
      registry inside the numeric auditor).
    - ``run_log_summary``: adapter-derived ground truth (usually
      ``adapter.run_log_summary(bundle)``); ``None`` ⇒ disclosure NOT_RUN /
      NO_GROUND_TRUTH.
    - ``has_statistical_evidence``: overclaim scope flag (patterns only fire
      as violations when statistical evidence exists to overclaim against).
    """

    citation_sources: Sequence[citations_mod.CitationMetadataSource] = ()
    citation_cache: fetchcache_mod.FetchCache | None = None
    entailment_judge: overclaim_mod.EntailmentJudge | None = None
    stated_results: tuple[float, ...] = ()
    recomputers: numbers_mod.MetricRecomputers | None = None
    run_log_summary: contract_mod.RunLogSummary | None = None
    has_statistical_evidence: bool = False


@dataclass(frozen=True)
class AuditOutcome:
    metric_sections: tuple[report_mod.MetricSection, ...]
    claims_result: claims_mod.ClaimsResult | None = None
    citation_result: citations_mod.CitationAuditResult | None = None
    numeric_result: numbers_mod.NumericProvenanceResult | None = None
    overclaim_result: overclaim_mod.OverclaimResult | None = None
    disclosure_result: disclosure_mod.DisclosureResult | None = None
    fetch_cache_manifest: report_mod.FetchCacheManifest | None = None


def map_text_sections(
    sections: Sequence[contract_mod.TextSection],
) -> list[numbers_mod.TextSection]:
    """contract.TextSection(section, text, source_file) → auditor
    TextSection(name, text). Value mapping; the source_file stays in the
    bundle manifest for provenance."""
    return [numbers_mod.TextSection(name=ts.section, text=ts.text) for ts in sections]


def _artifact_files(
    bundle: contract_mod.PaperArtifactBundle,
) -> list[numbers_mod.ArtifactFile]:
    root = Path(bundle.root)
    files: list[numbers_mod.ArtifactFile] = []
    for bf in bundle.files:
        if bf.role != contract_mod.FileRole.ARTIFACT:
            continue
        path = root / bf.path
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise report_mod.AuditRefusal(
                "ARTIFACT_UNREADABLE",
                f"artifact '{bf.path}' listed in manifest but unreadable: {exc}",
            ) from exc
        files.append(
            numbers_mod.ArtifactFile(path=bf.path, sha256=bf.sha256, text=text)
        )
    return files


def _log_files(
    bundle: contract_mod.PaperArtifactBundle,
) -> list[disclosure_mod.LogFile]:
    root = Path(bundle.root)
    files: list[disclosure_mod.LogFile] = []
    for bf in bundle.files:
        if bf.role != contract_mod.FileRole.RUN_LOG:
            continue
        path = root / bf.path
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            raise report_mod.AuditRefusal(
                "RUN_LOG_UNREADABLE",
                f"run log '{bf.path}' listed in manifest but unreadable: {exc}",
            ) from exc
        files.append(disclosure_mod.LogFile(path=bf.path, sha256=bf.sha256, text=text))
    return files


def _bib_entry_records(
    bundle: contract_mod.PaperArtifactBundle,
) -> tuple[citations_mod.BibEntryRecord, ...]:
    records: list[citations_mod.BibEntryRecord] = []
    for entry in bundle.bibliography:
        records.append(
            citations_mod.BibEntryRecord(
                key=entry.key,
                title=entry.title,
                authors=tuple(f"{a.given} {a.family}".strip() for a in entry.authors),
                year=entry.year,
                venue=entry.venue,
                doi=entry.doi,
                arxiv_id=entry.arxiv_id,
                entry_type=entry.entry_type,
                unparseable=False,
            )
        )
    # Parse errors are artifact-side data: one unparseable record per error.
    for error in bundle.bibliography_errors:
        records.append(
            citations_mod.BibEntryRecord(
                key=error.key or "",
                unparseable=True,
            )
        )
    return tuple(records)


def _ground_truth_from(
    summary: contract_mod.RunLogSummary | None,
) -> disclosure_mod.RunLogGroundTruth | None:
    if summary is None:
        return None
    return disclosure_mod.RunLogGroundTruth(
        source_files=tuple(summary.source_files),
        runs_total=summary.runs_total,
        runs_failed=summary.runs_failed,
        seeds_executed=summary.seeds_executed,
        stages_executed=tuple(summary.stages_executed)
        if summary.stages_executed
        else None,
        stages_failed_or_skipped=(
            tuple(summary.stages_failed_or_skipped)
            if summary.stages_failed_or_skipped
            else None
        ),
        error_events=tuple(summary.error_events),
    )


def _source_values_from_citations(
    result: citations_mod.CitationAuditResult | None,
) -> list[numbers_mod.SourceValue]:
    """TRACED_SOURCE allowed set: bibliographic numbers from RESOLVED_EXACT
    entries (reference years; DESIGN §5.3). Misattributed/unresolvable
    entries contribute nothing."""
    values: list[numbers_mod.SourceValue] = []
    if result is None:
        return values
    for entry in result.entries:
        if (
            entry.verdict == citations_mod.CitationVerdict.RESOLVED_EXACT
            and entry.record.year is not None
        ):
            values.append(
                numbers_mod.SourceValue(
                    value=float(entry.record.year), label=f"{entry.record.key}.year"
                )
            )
    return values


def _section_from_result(
    metric: str,
    *,
    status: str,
    not_run_reason: str | None,
    counts: Any,
    coverage: Any,
    details: Any,
    llm_assisted: bool = False,
) -> report_mod.MetricSection:
    if status != "RUN_COMPLETE":
        return report_mod.MetricSection.not_run(
            metric, not_run_reason or "not_run", llm_assisted=llm_assisted
        )
    return report_mod.MetricSection.complete(
        metric,
        counts=dict(counts),
        coverage=report_mod.CoverageInfo(
            items_examined=int(coverage.get("items_examined", 0)),
            items_skipped=int(coverage.get("items_skipped", 0)),
            skip_reasons=dict(coverage.get("skip_reasons", {})),
        ),
        details=tuple(details),
        llm_assisted=llm_assisted,
    )


async def _run_citations(
    bundle: contract_mod.PaperArtifactBundle, deps: MetricRunnerDeps
) -> citations_mod.CitationAuditResult | None:
    if not deps.citation_sources or deps.citation_cache is None:
        return None
    handle = citations_mod.BundleHandle(
        system_id=bundle.system_id,
        bibliography=_bib_entry_records(bundle),
    )
    auditor = citations_mod.CitationAuditor(
        sources=deps.citation_sources, cache=deps.citation_cache
    )
    return await auditor.audit(handle)


def audit_bundle(
    bundle: contract_mod.PaperArtifactBundle,
    deps: MetricRunnerDeps | None = None,
    *,
    adapter: contract_mod.BaselineAdapter | None = None,
) -> AuditOutcome:
    """Run every implementable metric against one audited bundle.

    ``adapter`` (optional) supplies the run-log summary and reproduction
    commands when the caller did not pass them via ``deps``.
    """
    deps = deps or MetricRunnerDeps()
    if adapter is not None and deps.run_log_summary is None:
        deps = MetricRunnerDeps(
            citation_sources=deps.citation_sources,
            citation_cache=deps.citation_cache,
            entailment_judge=deps.entailment_judge,
            stated_results=deps.stated_results,
            recomputers=deps.recomputers,
            run_log_summary=adapter.run_log_summary(bundle),
            has_statistical_evidence=deps.has_statistical_evidence,
        )

    sections = map_text_sections(bundle.paper_text)
    artifact_files = _artifact_files(bundle)

    # --- citations (async; network via injected sources + fetch cache) ---
    citation_result = asyncio.run(_run_citations(bundle, deps))
    if (
        citation_result is not None
        and citation_result.status == citations_mod.CitationStatus.RUN_COMPLETE
    ):
        citations_section = _section_from_result(
            "citations",
            status="RUN_COMPLETE",
            not_run_reason=None,
            counts=citation_result.counts,
            coverage=citation_result.coverage(),
            details=citation_result.detail_records(),
        )
    elif citation_result is not None:
        citations_section = report_mod.MetricSection.not_run(
            "citations", citation_result.not_run_reason or "not_run"
        )
    else:
        citations_section = report_mod.MetricSection.not_run(
            "citations", "no_citation_sources"
        )

    # --- claims (deterministic; evidence pool = artifacts + run logs) ---
    evidence: list[tuple[str, str]] = [
        (f"artifact:{af.path}", af.text) for af in artifact_files
    ]
    evidence.extend((f"run_log:{lf.path}", lf.text) for lf in _log_files(bundle))
    bib_keys = frozenset(entry.key for entry in bundle.bibliography if entry.key)
    claims_result = claims_mod.ClaimsAuditor().audit(
        sections, evidence_texts=evidence, bibliography_keys=bib_keys
    )
    claims_section = _section_from_result(
        "claims",
        status=claims_result.status,
        not_run_reason=claims_result.not_run_reason,
        counts=claims_result.counts,
        coverage=claims_result.coverage,
        details=claims_result.details,
    )

    # --- numbers (deterministic; recomputation registry) ---
    numeric_result = numbers_mod.NumericProvenanceAuditor(
        recomputers=deps.recomputers
    ).audit(
        sections,
        artifact_files=artifact_files,
        source_values=_source_values_from_citations(citation_result),
    )
    # Count-key translation (integration 2026-10-16): the numeric auditor emits
    # 'metrics_recomputable'/'metrics_reportable'; RATE_SPECS names the
    # denominator 'recomputable'. Mapping-layer fix; auditor untouched.
    numbers_counts = dict(numeric_result.counts)
    if "recomputable" not in numbers_counts:
        numbers_counts["recomputable"] = int(
            numbers_counts.get("metrics_recomputable", 0)
        )
    numbers_section = _section_from_result(
        "numbers",
        status=numeric_result.status,
        not_run_reason=numeric_result.not_run_reason,
        counts=numbers_counts,
        coverage=numeric_result.coverage,
        details=numeric_result.details,
    )

    # --- overclaim (deterministic patterns; optional advisory judge) ---
    overclaim_result = overclaim_mod.OverclaimAuditor(
        has_statistical_evidence=deps.has_statistical_evidence
    ).audit(
        sections,
        judge=deps.entailment_judge,
        stated_results=deps.stated_results,
    )
    overclaim_section = _section_from_result(
        "overclaim",
        status=overclaim_result.status,
        not_run_reason=overclaim_result.not_run_reason,
        counts=overclaim_result.counts,
        coverage=overclaim_result.coverage,
        details=overclaim_result.details,
        llm_assisted=deps.entailment_judge is not None,
    )

    # --- disclosure (deterministic; requires ground truth) ---
    truth = _ground_truth_from(deps.run_log_summary)
    if truth is None:
        disclosure_section = report_mod.MetricSection.not_run(
            "disclosure", "NO_GROUND_TRUTH"
        )
        disclosure_result = None
    else:
        disclosure_result = disclosure_mod.FailureDisclosureAuditor().audit(
            sections, ground_truth=truth
        )
        disclosure_section = _section_from_result(
            "disclosure",
            status=disclosure_result.status,
            not_run_reason=disclosure_result.not_run_reason,
            counts=disclosure_result.counts,
            coverage=disclosure_result.coverage,
            details=disclosure_result.details,
        )

    # --- reproducibility (re-run released commands in a pristine artifact copy) ---
    if adapter is not None:
        reproduction_commands = adapter.reproduction_commands(bundle)
    else:
        reproduction_commands = ()
    reproduction_result = reproducibility_mod.ReproducibilityAuditor().audit(
        reproduction_commands, Path(bundle.root)
    )
    reproduction_section = _section_from_result(
        "reproducibility",
        status=reproduction_result.status,
        not_run_reason=reproduction_result.not_run_reason,
        counts=reproduction_result.counts,
        coverage=reproduction_result.coverage,
        details=reproduction_result.details,
    )

    manifest = (
        deps.citation_cache.manifest() if deps.citation_cache is not None else None
    )

    return AuditOutcome(
        metric_sections=(
            citations_section,
            claims_section,
            numbers_section,
            overclaim_section,
            disclosure_section,
            reproduction_section,
        ),
        claims_result=claims_result,
        citation_result=citation_result,
        numeric_result=numeric_result,
        overclaim_result=overclaim_result,
        disclosure_result=disclosure_result,
        fetch_cache_manifest=manifest,
    )
