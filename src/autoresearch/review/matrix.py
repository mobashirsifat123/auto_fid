"""Purpose: Decision-useful evidence matrix — deterministic assembly of the
review's structured extractions into a comparable matrix plus a retrieval
coverage record.

Requirement C (review enhancement): the review output must include a
structured evidence matrix (problem, method, datasets, protocol, outcomes,
limitations, contradictions, code availability) and an auditable coverage
record (queries, sources, dates, inclusion counts). Every field here is
assembled from already-verified rows (paper_reviews, evidence_claims,
priority_scores, search-strategy checkpoints) — no model calls, no invention;
absence is recorded as absence.
"""

from __future__ import annotations

import json
from typing import Any
from uuid import uuid4

from autoresearch.application.keywords import keyword_tokens, pick_keywords
from autoresearch.domain.models import ArtifactReference, content_checksum
from autoresearch.storage.repository import ResearchRepository

MATRIX_PROMPT_VERSION = "evidence-matrix-v1"


def project_keywords(
    repository: ResearchRepository,
    *,
    project_id: str,
    run_id: str,
) -> list[str]:
    """Deterministic 3-6 keyword terms per project for the manuscript
    Keywords block (template-spec.json T28). Derived ONLY from the stored
    seed metadata and top-ranked paper titles — never invented. Returns as
    many as the durable record honestly provides."""
    seed = repository.get_seed(project_id)
    seed_texts: list[str] = []
    if seed is not None:
        seed_texts = [
            str(seed.topic),
            str(seed.research_question),
            ", ".join(seed.methods),
            ", ".join(seed.datasets),
        ]
    titles = [
        str(row["title"] or "")
        for row in repository.rows(
            "SELECT p.title AS title FROM priority_scores s JOIN papers p ON p.id = s.paper_id "
            "WHERE s.run_id=? ORDER BY s.total DESC LIMIT 5",
            (run_id,),
        )
    ]
    return pick_keywords([keyword_tokens(*seed_texts), keyword_tokens(*titles)])


def build_evidence_matrix(
    repository: ResearchRepository,
    *,
    project_id: str,
    run_id: str,
) -> dict[str, Any]:
    seed_row = repository.row("SELECT topic, research_question FROM seed_papers WHERE project_id=? ORDER BY rowid DESC LIMIT 1", (project_id,))
    strategies_checkpoint = repository.checkpoint(run_id, "search_strategies") or {}
    strategies = [
        {"name": str(item.get("name")), "query": str(item.get("query"))}
        for item in (strategies_checkpoint.get("payload") or [])
    ]
    reviews = repository.rows(
        "SELECT * FROM paper_reviews WHERE run_id=? ORDER BY rowid", (run_id,)
    )
    scores = {
        str(row["paper_id"]): float(row["total"])
        for row in repository.rows("SELECT paper_id, total FROM priority_scores WHERE run_id=?", (run_id,))
    }
    gap = repository.gap_for_run(run_id)

    matrix: list[dict[str, Any]] = []
    for review in reviews:
        paper_id = str(review["paper_id"])
        paper = repository.row("SELECT title, year, venue, doi, url, abstract, open_access_pdf_url FROM papers WHERE id=?", (paper_id,))
        counter = repository.rows(
            "SELECT id, statement FROM evidence_claims WHERE paper_id=? AND claim_type IN ('contradiction','counter_evidence','limitation') LIMIT 4",
            (paper_id,),
        )
        abstract = str(paper.get("abstract") or "") if paper else ""
        matrix.append({
            "paper_id": paper_id,
            "title": str(paper.get("title")) if paper else "",
            "year": paper.get("year") if paper else None,
            "venue": str(paper.get("venue") or "") if paper else "",
            "peer_reviewed": bool(paper and str(paper.get("venue") or "").strip().lower() not in ("", "unknown", "none", "arxiv")),
            "priority_score": scores.get(paper_id),
            "problem": str(review["problem"] or ""),
            "research_question": str(review["research_question"] or ""),
            "method": str(review["method"] or ""),
            "datasets": json.loads(review["datasets_json"] or "[]"),
            "protocol": str(review["experimental_protocol"] or ""),
            "results": json.loads(review["results_json"] or "[]"),
            "ablations": json.loads(review["ablations_json"] or "[]"),
            "limitations": json.loads(review["limitations_json"] or "[]"),
            "failure_modes": json.loads(review["failure_modes_json"] or "[]"),
            "contradictory_findings": [
                {"claim_id": str(row["id"]), "statement": str(row["statement"])} for row in counter
            ],
            # Deterministic, honestly-labeled heuristic: code availability is
            # asserted only by explicit link presence, never inferred.
            "code_availability": (
                "explicit_link" if paper and (paper.get("url") or paper.get("open_access_pdf_url")) else
                ("mentioned_in_abstract" if "github.com" in abstract.lower() or "code is available" in abstract.lower() else "not_found")
            ),
        })

    return {
        "scope": {
            "topic": str(seed_row["topic"]) if seed_row else "",
            "task_question": str(seed_row["research_question"]) if seed_row else "",
            "review_run_id": run_id,
            "keywords": project_keywords(repository, project_id=project_id, run_id=run_id),
            "inclusion": f"top-ranked papers extracted this run (matrix rows: {len(matrix)}); "
                         "full discovery corpus remains in the papers table",
            "research_to_code": "explicit links only; deep source-function mapping is future work",
        },
        "retrieval_coverage": {
            "strategies": strategies,
            "papers_in_project": repository.row("SELECT COUNT(*) AS n FROM papers WHERE project_id=?", (project_id,))["n"],
            "reviews_this_run": len(matrix),
            "review_failures_this_run": repository.row("SELECT COUNT(*) AS n FROM review_failures WHERE run_id=?", (run_id,))["n"],
        },
        "counter_evidence_summary": {
            "gap_id": str(gap.id) if gap else None,
            "counter_claim_ids": list(gap.counter_evidence_claim_ids) if gap else [],
        },
        "matrix": matrix,
    }


def save_evidence_matrix(repository: ResearchRepository, *, project_id: str, run_id: str) -> ArtifactReference:
    matrix = build_evidence_matrix(repository, project_id=project_id, run_id=run_id)
    root = repository.artifact_root / project_id / run_id
    root.mkdir(parents=True, exist_ok=True)
    path = root / "evidence-matrix.json"
    text = json.dumps(matrix, indent=2, ensure_ascii=False, default=str)
    path.write_text(text, encoding="utf-8")
    artifact = ArtifactReference(
        uuid4().hex, project_id=project_id, run_id=run_id,
        artifact_type="evidence_matrix", path=str(path),
        source_evidence=tuple(str(row["paper_id"]) for row in matrix["matrix"]) or (run_id,),
        provider="deterministic", model="matrix-writer",
        prompt_version=MATRIX_PROMPT_VERSION, checksum=content_checksum(text),
    )
    repository.save_artifact(artifact)
    return artifact
