"""Durable reinvention-cycle controller for iterative multi-cycle research."""

from __future__ import annotations

import logging
from uuid import uuid4

from autoresearch.domain.models import NoveltyCandidate, ReinventionAnalysis
from autoresearch.reinvention.pipeline import ReinventionPipeline
from autoresearch.storage.repository import ResearchRepository

logger = logging.getLogger(__name__)


class DurableReinventionCycleController:
    def __init__(self, repository: ResearchRepository, pipeline: ReinventionPipeline) -> None:
        self.repository = repository
        self.pipeline = pipeline

    async def advance_cycle(self, candidate_id: str) -> tuple[ReinventionAnalysis, NoveltyCandidate | None]:
        """Advance the research lifecycle from Cycle N to Cycle N+1 for a given candidate."""
        run_id = f"reinvention-{uuid4().hex[:8]}"
        logger.info("Advancing research cycle for candidate %s with run_id %s", candidate_id, run_id)
        return await self.pipeline.analyze_and_reinvent(candidate_id, run_id=run_id)
