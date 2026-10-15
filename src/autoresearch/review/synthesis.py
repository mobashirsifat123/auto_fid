"""Purpose: Compiles all verified evidence claims, paper reviews, and the validated gap into structured JSON and human-readable Markdown reports."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

from autoresearch.domain.models import ArtifactReference, EvidenceClaim, PaperRecord, PaperReview, PriorityScore, ResearchGap, content_checksum
from autoresearch.storage.repository import ResearchRepository


class ReviewSynthesisWriter:
    def __init__(self, repository: ResearchRepository) -> None:
        self.repository = repository

    def write(self, *, project_id: str, run_id: str, papers: list[PaperRecord], reviews: list[PaperReview], scores: list[PriorityScore], claims: list[EvidenceClaim], gap: ResearchGap) -> tuple[ArtifactReference, ArtifactReference]:
        root = self.repository.artifact_root / project_id / run_id
        root.mkdir(parents=True, exist_ok=True)
        paper_map = {paper.id: paper for paper in papers}
        review_map = {review.paper_id: review for review in reviews}
        claim_map = {claim.id: claim for claim in claims}
        downloads = self.repository.rows("SELECT paper_id,status,path FROM download_records WHERE project_id=? ORDER BY created_at", (project_id,))
        legal_full_text = {str(row["paper_id"]) for row in downloads if row["status"] == "downloaded" and row.get("path")}
        reviewed_ids = set(review_map)
        evidence_papers = {claim.paper_id for claim in claims}
        verified_evidence = {claim.paper_id for claim in claims if claim.excerpt.strip()}
        coverage = {
            "papers_ingested": len(papers), "papers_parsed": len(reviewed_ids),
            "full_text_papers": len(legal_full_text), "abstract_only_papers": len(set(paper_map) - legal_full_text),
            "papers_with_extracted_evidence": len(evidence_papers),
            "papers_with_verified_evidence": len(verified_evidence),
            "failures": len(self.repository.rows("SELECT id FROM review_failures WHERE run_id=?", (run_id,))),
        }
        payload = {
            "project_id": project_id, "run_id": run_id,
            "coverage": coverage,
            "provenance": {"provider": gap.provenance.provider, "model": gap.provenance.model, "prompt_version": gap.provenance.prompt_version, "timestamp": gap.provenance.created_at, "checksum": gap.provenance.checksum, "source_evidence": gap.provenance.source_evidence},
            "ranked_papers": [{"rank": index, "paper": _paper_payload(paper_map[score.paper_id]), "review": _review_payload(review_map[score.paper_id]), "score": {"total": score.total, **dict(score.explanation)}} for index, score in enumerate(scores, start=1)],
            "gap": {"statement": gap.statement, "classification": gap.classification.value, "supporting_paper_ids": gap.supporting_paper_ids, "supporting_claims": [_claim_payload(claim_map[item]) for item in gap.supporting_claim_ids], "why_existing_work_does_not_solve_it": gap.why_unsolved, "missing_experiment": gap.missing_experiment, "falsification_test": gap.falsification_test, "confidence": gap.confidence, "counter_evidence": [_claim_payload(claim_map[item]) for item in gap.counter_evidence_claim_ids if item in claim_map]},
        }
        json_path = root / "gap-report.json"
        markdown_path = root / "gap-report.md"
        json_text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        json_path.write_text(json_text, encoding="utf-8")
        markdown = _markdown(payload)
        markdown_path.write_text(markdown, encoding="utf-8")
        evidence = tuple(dict.fromkeys((*gap.supporting_claim_ids, *gap.counter_evidence_claim_ids)))
        common = {"project_id": project_id, "run_id": run_id, "source_evidence": evidence, "provider": gap.provenance.provider, "model": gap.provenance.model, "prompt_version": gap.provenance.prompt_version}
        json_artifact = ArtifactReference(content_checksum(f"gap_report_json|{run_id}")[:32], artifact_type="gap_report_json", path=str(json_path), checksum=content_checksum(json_text), **common)
        markdown_artifact = ArtifactReference(content_checksum(f"gap_report_markdown|{run_id}")[:32], artifact_type="gap_report_markdown", path=str(markdown_path), checksum=content_checksum(markdown), **common)
        self.repository.save_artifact(json_artifact)
        self.repository.save_artifact(markdown_artifact)
        return json_artifact, markdown_artifact


def _paper_payload(paper: PaperRecord) -> dict:
    return {"id": paper.id, "title": paper.title, "authors": paper.authors, "year": paper.year, "doi": paper.doi, "url": paper.url}


def _review_payload(review: PaperReview) -> dict:
    return {"problem": review.problem, "research_question": review.research_question, "method": review.method, "datasets": review.datasets, "experimental_protocol": review.experimental_protocol, "results": review.results, "ablations": review.ablations, "limitations": review.limitations, "failure_modes": review.failure_modes, "llm_evidence_support_score": review.confidence, "evidence_claim_ids": review.evidence_claim_ids}


def _claim_payload(claim: EvidenceClaim) -> dict:
    return {"id": claim.id, "paper_id": claim.paper_id, "statement": claim.statement, "excerpt": claim.excerpt, "page": claim.page, "section": claim.section, "confidence": claim.confidence}


def _markdown(payload: dict) -> str:
    gap = payload["gap"]
    provenance = payload["provenance"]
    lines = [
        "# Evidence-cited Research Gap",
        "",
        f"**Classification:** {gap['classification']}",
        f"**LLM evidence-support score (not statistical confidence):** {gap['confidence']:.2f}",
        "",
        "## Provenance",
        "",
        f"- Project ID: `{payload['project_id']}`",
        f"- Run ID: `{payload['run_id']}`",
        f"- Provider/model: `{provenance['provider']}` / `{provenance['model']}`",
        f"- Prompt version: `{provenance['prompt_version']}`",
        f"- Timestamp: `{provenance['timestamp']}`",
        f"- Source evidence: `{', '.join(provenance['source_evidence'])}`",
        f"- Content checksum: `{provenance['checksum']}`",
        "",
        "## Gap statement",
        "",
        gap["statement"],
        "",
        "## Why existing work does not solve it",
        "",
        gap["why_existing_work_does_not_solve_it"],
        "",
        "## Supporting evidence",
        "",
    ]
    coverage = payload.get("coverage", {})
    lines.extend(["## Review coverage", "", *[f"- {key.replace('_', ' ')}: {value}" for key, value in coverage.items()], ""])
    for claim in gap["supporting_claims"]:
        lines.extend([f"- `{claim['paper_id']}` — {claim['statement']}", f"  > {claim['excerpt']}"])
    lines.extend(["", "## Missing experiment", "", gap["missing_experiment"], "", "## Falsification test", "", gap["falsification_test"], "", "## Counter-evidence", ""])
    if gap["counter_evidence"]:
        for claim in gap["counter_evidence"]:
            lines.extend([f"- `{claim['paper_id']}` — {claim['statement']}", f"  > {claim['excerpt']}"])
    else:
        lines.append("No explicit counter-evidence claim was identified in the reviewed corpus.")
    lines.extend(["", "## Ranked papers", ""])
    for item in payload["ranked_papers"]:
        lines.append(f"{item['rank']}. {item['paper']['title']} — score {item['score']['total']:.4f}")
    lines.append("")
    return "\n".join(lines)
