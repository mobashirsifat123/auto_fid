"""Auditable post-run validation for live and scale workflows."""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from autoresearch.domain.models import DownloadStatus, PipelineKind, RunStatus, content_checksum
from autoresearch.providers.redaction import redact_text
from autoresearch.storage.repository import ResearchRepository


def audit_project(repository: ResearchRepository, project_id: str, *, run_id: str | None = None) -> dict[str, Any]:
    project = repository.get_project(project_id)
    if not project:
        raise ValueError(f"Project '{project_id}' was not found.")
    runs = [repository.get_run(row["id"]) for row in repository.rows("SELECT id FROM pipeline_runs WHERE project_id=? ORDER BY started_at", (project_id,))]
    runs = [run for run in runs if run]
    discovery = repository.get_run(run_id) if run_id else repository.latest_run(project_id, PipelineKind.DISCOVERY)
    review = repository.latest_run(project_id, PipelineKind.REVIEW)
    papers = repository.list_papers(project_id)
    downloads = repository.downloads(project_id)
    claims = repository.claims(project_id)
    usage = repository.rows("SELECT * FROM provider_usage WHERE project_id=? ORDER BY created_at", (project_id,))
    source_events = repository.search_events(discovery.id) if discovery else []
    source_counts: dict[str, dict[str, int]] = {}
    for event in source_events:
        count = source_counts.setdefault(event["source"], {"requests": 0, "successes": 0, "failures": 0, "records": 0})
        count["requests"] += 1
        count["records"] += int(event["result_count"])
        count["failures" if event.get("error") else "successes"] += 1

    evidence_rows = []
    for claim in claims:
        paper = repository.get_paper(claim.paper_id)
        source_text = paper.abstract if paper else ""
        download = repository.download_for_paper(claim.paper_id)
        if download and download.status is DownloadStatus.DOWNLOADED and download.path and Path(download.path).is_file():
            try:
                source_text = "\n\n".join(page.extract_text() or "" for page in PdfReader(download.path).pages)
            except Exception as exc:
                evidence_rows.append({"claim_id": claim.id, "paper_id": claim.paper_id, "verified": False, "error": redact_text(f"{type(exc).__name__}: {exc}")})
                continue
        evidence_rows.append({"claim_id": claim.id, "paper_id": claim.paper_id, "verified": _normalize(claim.excerpt) in _normalize(source_text), "source": "pdf" if download and download.status is DownloadStatus.DOWNLOADED else "abstract"})

    discovery_payload = _discovery_payload(repository, discovery)
    secret_hits = _secret_hits(repository, project_id, runs)
    costs = [row["estimated_cost"] for row in usage if row.get("estimated_cost") is not None]
    latencies = [float(row["latency_ms"]) for row in usage if row.get("latency_ms") is not None]
    status_counts = Counter(item.status.value for item in downloads)
    checkpoint_rows = []
    for run in runs:
        checkpoint_rows.extend(repository.checkpoints(run.id))
    return {
        "project_id": project_id,
        "project_title": project.title,
        "audited_at": repository.row("SELECT datetime('now') AS now")["now"],
        "database_integrity": repository.integrity_check(),
        "runs": [{"id": run.id, "kind": run.kind.value, "status": run.status.value, "started_at": run.started_at, "completed_at": run.completed_at, "error": redact_text(run.error) if run.error else None} for run in runs],
        "discovery_run_id": discovery.id if discovery else None,
        "review_run_id": review.id if review else None,
        "paper_count": len(papers),
        "source_counts": source_counts,
        "deduplication": discovery_payload.get("deduplication", {}),
        "pdf_download_status": dict(status_counts),
        "pdf_legal_basis": Counter(item.legal_basis or "none" for item in downloads),
        "evidence_validation": {"claims": len(evidence_rows), "verified": sum(item["verified"] for item in evidence_rows), "failed": sum(not item["verified"] for item in evidence_rows), "rows": evidence_rows},
        "provider_usage": {"records": len(usage), "prompt_tokens": sum(int(row["prompt_tokens"] or 0) for row in usage), "completion_tokens": sum(int(row["completion_tokens"] or 0) for row in usage), "estimated_cost": round(sum(costs), 8), "currency": sorted({row["currency"] for row in usage if row.get("currency")}), "latency_ms_total": round(sum(latencies), 3), "latency_ms_avg": round(sum(latencies) / len(latencies), 3) if latencies else None, "by_provider_model": Counter(f"{row['provider']}/{row['model']}" for row in usage)},
        "checkpoint_recovery": {"records": len(checkpoint_rows), "completed": sum(row["status"] == "completed" for row in checkpoint_rows), "failed": sum(row["status"] == "failed" for row in checkpoint_rows), "retried": sum(int(row["attempts"]) > 1 for row in checkpoint_rows), "max_attempts": max((int(row["attempts"]) for row in checkpoint_rows), default=0)},
        "secret_scan": {"hits": secret_hits, "passed": not secret_hits},
    }


def write_audit(repository: ResearchRepository, project_id: str, *, run_id: str | None = None) -> Path:
    payload = audit_project(repository, project_id, run_id=run_id)
    effective_run = run_id or payload.get("review_run_id") or payload.get("discovery_run_id") or "audit"
    root = repository.artifact_root / project_id / effective_run
    root.mkdir(parents=True, exist_ok=True)
    path = root / "validation.json"
    text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
    path.write_text(text, encoding="utf-8")
    return path


def _discovery_payload(repository: ResearchRepository, run) -> dict[str, Any]:
    if not run:
        return {}
    artifacts = [item for item in repository.artifacts_for_run(run.id) if item.artifact_type == "discovery_summary"]
    if not artifacts or not Path(artifacts[-1].path).is_file():
        return {}
    try:
        return json.loads(Path(artifacts[-1].path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _secret_hits(repository: ResearchRepository, project_id: str, runs) -> list[str]:
    roots = [repository.artifact_root / project_id, repository.path]
    values = [value for name, value in os.environ.items() if value and len(value) >= 12 and any(marker in name.upper() for marker in ("KEY", "TOKEN", "SECRET", "PASSWORD"))]
    hits: list[str] = []
    for root in roots:
        paths = [root] if root.is_file() else list(root.rglob("*")) if root.exists() else []
        for path in paths:
            if not path.is_file():
                continue
            try:
                raw = path.read_bytes()
            except OSError:
                continue
            if any(value.encode() in raw for value in values):
                hits.append(str(path))
            if re.search(rb"(?i)(bearer|api[_ -]?key|authorization)\s*[:=]\s*[^\s,;]{12,}", raw):
                hits.append(str(path))
    return sorted(set(hits))


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"\w+", value.lower()))
