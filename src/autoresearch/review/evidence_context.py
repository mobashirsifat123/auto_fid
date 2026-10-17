"""Purpose: Builds the bounded, validated literature context that grounds
novelty candidate generation and experiment planning in the project's
evidence corpus.

Adapted (mechanism only, no code copied; upstream repository carries no
license file) from HKUDS/AI-Researcher @ f9a6f8480860c193afff600eeffe3defcee8a978,
``research_agent/inno/agents/inno_agent/idea_agent.py``: the Idea Generation
Agent conditions proposal writing on a structured summary of *existing
methods* extracted from the actual papers, and the Survey Agent traces every
atomic definition back to reference papers.  Here the same grounding is
served from the verified SQLite corpus instead of an agent browsing files,
and every identifier passed downstream already passed excerpt validation at
review time.
"""

from __future__ import annotations

from typing import Any, Iterable

from autoresearch.storage.repository import ResearchRepository

# Bounds keep the prompt deterministic in size regardless of corpus scale.
MAX_PAPERS = 8
MAX_CLAIMS = 12
MAX_EXCERPT_CHARS = 700


def build_evidence_context(
    repository: ResearchRepository,
    *,
    project_id: str,
    supporting_paper_ids: Iterable[str],
    supporting_claim_ids: Iterable[str],
    counter_claim_ids: Iterable[str] = (),
    candidate_id: str | None = None,
    gap_row: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Collect papers, review methods, and validated claims for prompt grounding.

    Missing identifiers are skipped silently: the context is an enrichment of
    prompts, not a validation gate, so a sparse corpus degrades to the previous
    behaviour instead of failing the run. Paper and claim lookups are
    project-scoped so a cross-project ID collision can never leak another
    project's literature into the prompt.
    """
    paper_ids = list(dict.fromkeys(str(p) for p in supporting_paper_ids))
    claim_ids = list(dict.fromkeys(str(c) for c in supporting_claim_ids))
    counter_ids = [str(c) for c in counter_claim_ids if str(c) not in set(claim_ids)]

    papers: list[dict[str, Any]] = []
    seen_papers: set[str] = set()
    for paper_id in paper_ids[:MAX_PAPERS]:
        entry = _paper_entry(repository, project_id, paper_id)
        if entry is None:
            continue
        papers.append(entry)
        seen_papers.add(paper_id)

    def _claim_rows(ids: list[str]) -> list[dict[str, Any]]:
        if not ids:
            return []
        placeholders = ",".join("?" for _ in ids)
        rows = {
            str(row["id"]): row
            for row in repository.rows(
                f"SELECT id,paper_id,excerpt,section FROM evidence_claims "
                f"WHERE project_id=? AND id IN ({placeholders})",
                (project_id, *ids),
            )
        }
        ordered: list[dict[str, Any]] = []
        for claim_id in ids:  # preserve the gap's prioritized claim order
            row = rows.get(claim_id)
            if row is None:
                continue
            paper_id = str(row["paper_id"])
            # Claims may reference papers beyond the MAX_PAPERS cut; render
            # those papers as title-only entries so every cited paper is
            # present in the prompt (bounded by MAX_CLAIMS).
            if paper_id not in seen_papers and len(papers) < 2 * MAX_PAPERS:
                entry = _paper_entry(repository, project_id, paper_id)
                if entry is not None:
                    papers.append(entry)
                    seen_papers.add(paper_id)
            ordered.append({
                "claim_id": claim_id,
                "paper_id": paper_id,
                "excerpt": " ".join(str(row["excerpt"] or "").split())[:MAX_EXCERPT_CHARS],
                "section": str(row.get("section") or ""),
            })
        return ordered[:MAX_CLAIMS]

    claims = _claim_rows(claim_ids)
    counter_claims = _claim_rows(counter_ids)

    context: dict[str, Any] = {
        "papers": papers,
        "claims": claims,
        "counter_evidence_claims": counter_claims,
    }

    if gap_row is not None:
        context["gap"] = {
            "statement": str(gap_row.get("statement") or ""),
            "falsification_test": str(gap_row.get("falsification_test") or ""),
            "missing_experiment": str(gap_row.get("missing_experiment") or ""),
        }

    if candidate_id is not None:
        assessment = repository.get_prior_art_assessment(candidate_id)
        if assessment is not None:
            context["closest_prior_methods"] = {
                "paper_ids": list(assessment.closest_paper_ids),
                "overlap_status": assessment.overlap_status.value,
                "mechanism_comparison": assessment.mechanism_comparison[:600],
            }

    return context


def _paper_entry(repository: ResearchRepository, project_id: str, paper_id: str) -> dict[str, Any] | None:
    row = repository.row("SELECT id,title,year FROM papers WHERE id=? AND project_id=?", (paper_id, project_id))
    if row is None:
        return None
    review_row = repository.row(
        "SELECT method,limitations_json FROM paper_reviews WHERE project_id=? AND paper_id=? ORDER BY rowid DESC LIMIT 1",
        (project_id, paper_id),
    )
    method = ""
    limitations: list[str] = []
    if review_row is not None:
        method = str(review_row["method"] or "")[:400]
        try:
            import json as _json
            limitations = [str(x) for x in _json.loads(review_row["limitations_json"] or "[]")][:4]
        except (TypeError, ValueError):
            limitations = []
    return {
        "paper_id": str(row["id"]),
        "title": str(row["title"]),
        "year": row["year"],
        "method": method,
        "limitations": limitations,
    }


def format_evidence_context(context: dict[str, Any] | None) -> str:
    """Render the context as a compact prompt block; empty string when absent."""
    if not context:
        return ""
    parts: list[str] = []
    papers = context.get("papers") or []
    claims = context.get("claims") or []
    counter = context.get("counter_evidence_claims") or []
    if papers:
        lines = [
            f"- [{p['paper_id']}] {p['title']} ({p.get('year') or 'n.d.'})"
            + (f" method: {p['method']}" if p.get("method") else "")
            + (f" | limitations: {'; '.join(p['limitations'])}" if p.get("limitations") else "")
            for p in papers
        ]
        parts.append("Reviewed papers and their methods:\n" + "\n".join(lines))
    if claims:
        lines = [
            f"- [{c['claim_id']}] (paper {c['paper_id']}"
            + (f", {c['section']}" if c.get("section") else "")
            + f"): {c['excerpt']}"
            for c in claims
        ]
        parts.append("Verified evidence claims (verbatim excerpts):\n" + "\n".join(lines))
    if counter:
        lines = [f"- [{c['claim_id']}] (paper {c['paper_id']}): {c['excerpt']}" for c in counter]
        parts.append("Counter-evidence claims that may weaken the gap:\n" + "\n".join(lines))
    gap = context.get("gap")
    if gap:
        parts.append(
            "Gap record: "
            f"falsification_test='{gap.get('falsification_test') or ''}' "
            f"missing_experiment='{gap.get('missing_experiment') or ''}'"
        )
    prior = context.get("closest_prior_methods")
    if prior:
        parts.append(
            "Closest prior methods identified by prior-art check: "
            f"paper_ids={prior.get('paper_ids')} overlap_status={prior.get('overlap_status')} "
            f"comparison: {prior.get('mechanism_comparison')}"
        )
    findings = context.get("findings_memory") or []
    if findings:
        lines = [
            f"- [{f['finding_id']}] status={f['status']} hypothesis='{f['hypothesis']}' "
            f"outcome={f.get('observed_outcome') or {}} applicability='{f['applicability']}' "
            f"lesson='{f['lesson']}' next_action='{f['next_action']}'"
            for f in findings
        ]
        parts.append(
            "Findings Memory — prior experimental outcomes for this question, INCLUDING failures and refutations. "
            "Do NOT repropose a mechanism that a finding refuted under matching applicability conditions "
            "without an explicit mechanism change:\n" + "\n".join(lines)
        )
    methodology = context.get("methodology")
    if methodology:
        predicted = methodology.get("predicted_observations")
        procedure = methodology.get("procedure")
        falsification = methodology.get("falsification_conditions")
        parts.append(
            f"Frozen methodology v{methodology.get('version')} (freeze {methodology.get('freeze_id')}) "
            f"— the design must implement THIS methodology: "
            f"hypothesis='{methodology.get('hypothesis_statement') or ''}' "
            f"algorithm_outline='{str(methodology.get('algorithm_outline') or '')[:600]}' "
            f"procedure={procedure if procedure else ''} "
            f"predicted_observations={predicted if predicted else ''} "
            f"falsification_conditions={falsification if falsification else ''} "
            f"expected_resource_demands='{methodology.get('expected_resource_demands') or ''}'"
        )
    return "\n\n".join(parts)
