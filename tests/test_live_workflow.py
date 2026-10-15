from __future__ import annotations

import os
from pathlib import Path

import pytest

from autoresearch.application.audit import audit_project
from autoresearch.application.config import Settings
from autoresearch.application.factory import ApplicationContainer


@pytest.mark.skipif(os.getenv("RUN_LIVE_AUTORESEARCH_TESTS") != "1", reason="set RUN_LIVE_AUTORESEARCH_TESTS=1 for the opt-in paid/network workflow")
@pytest.mark.asyncio
async def test_live_twenty_five_workflow_with_auditable_integrations(tmp_path: Path):
    """One opt-in real-network test; normal tests never use paid credentials or scholarly APIs."""
    provider_config = Path(os.getenv("AUTORESEARCH_PROVIDER_CONFIG", Path(__file__).parents[1] / "docs" / "live_config.example.toml")).resolve()
    settings = Settings(
        data_root=tmp_path,
        database_path=tmp_path / "autoresearch.sqlite3",
        artifact_root=tmp_path / "artifacts",
        download_root=tmp_path / "downloads",
        provider_config=provider_config,
        search_sources=("semantic_scholar", "openalex", "crossref", "arxiv", "openreview"),
    )
    maximum_cost = float(os.getenv("AUTORESEARCH_LIVE_MAX_COST_USD", "2.00"))
    maximum_runtime = float(os.getenv("AUTORESEARCH_LIVE_MAX_RUNTIME_SECONDS", "1800"))
    seed = os.getenv("AUTORESEARCH_LIVE_SEED", "10.48550/arXiv.2005.11401")
    container = ApplicationContainer(settings)
    try:
        discovery = await container.discovery.discover(seed, limit=20, max_cost_usd=maximum_cost, max_runtime_seconds=maximum_runtime)
        review = await container.review.review(discovery.project_id, top_k=5, hierarchical=True, max_cost_usd=maximum_cost, max_runtime_seconds=maximum_runtime)
        audit = audit_project(container.repository, discovery.project_id, run_id=discovery.run_id)
    finally:
        await container.close()

    assert discovery.retrieved == 20
    assert len(review.top_paper_ids) == 5
    assert Path(review.markdown_report).is_file()
    assert {"semantic_scholar", "openalex", "crossref", "arxiv", "openreview"} <= set(audit["source_counts"])
    assert all(audit["source_counts"][source]["successes"] > 0 for source in ("semantic_scholar", "openalex", "crossref", "arxiv", "openreview"))
    assert audit["provider_usage"]["records"] > 0
    assert audit["provider_usage"]["estimated_cost"] <= maximum_cost
    assert audit["evidence_validation"]["failed"] == 0
    assert audit["secret_scan"]["passed"] is True
