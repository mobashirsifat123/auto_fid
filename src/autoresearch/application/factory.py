"""Production composition root for the five research skills."""

from __future__ import annotations

from autoresearch.papers.metadata_sources import ArxivSource, CrossrefSource, MetadataSource, OpenAlexSource, OpenReviewSource, SemanticScholarSource
from autoresearch.papers.pdf_resolver import LegalPdfResolver
from autoresearch.papers.search_pipeline import StudySearchPipeline
from autoresearch.providers.budget import ProviderBudgetGuard
from autoresearch.providers.factory import build_router
from autoresearch.storage.repository import ResearchRepository
from autoresearch.workers.local_typed import LocalTypedWorker

from .services import DiscoveryService, LiteratureReviewService, NoveltyService, ExperimentService
from .manuscript import ManuscriptService
from .workflow import WorkflowService


class ApplicationContainer:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.repository = ResearchRepository(settings.database_path, settings.artifact_root)
        self.router = build_router(settings.provider_config, usage_sink=self.repository.record_provider_usage, budget_guard=ProviderBudgetGuard(self.repository, default_token_ceiling=settings.default_max_tokens, default_hard_stop=settings.default_max_cost_usd), auto_provider=settings.auto_provider)
        self.sources = _sources(settings.search_sources)
        self.pdf_resolver = LegalPdfResolver()
        self.search_pipeline = StudySearchPipeline(
            self.repository, self.sources, self.pdf_resolver, settings.download_root,
            citation_expansion=settings.discovery_citation_expansion,
            citation_anchors=settings.discovery_citation_anchors,
            citation_neighbors=settings.discovery_citation_neighbors,
        )
        self.discovery = DiscoveryService(
            self.repository, self.router, self.search_pipeline,
            strategy_perspectives=settings.discovery_query_perspectives,
        )
        self.review = LiteratureReviewService(
            self.repository, self.router, ensemble=settings.review_ensemble
        )
        self.novelty = NoveltyService(self.repository, self.router, self.search_pipeline)
        self.worker = LocalTypedWorker(self.repository, settings.data_root / "workspace")
        self.experiment = ExperimentService(self.repository, self.router, settings.data_root / "workspace", worker=self.worker)
        self.manuscript = ManuscriptService(self.repository, self.router, writer_mode=settings.manuscript_writer)
        self.workflow = WorkflowService(
            self.repository, self.router,
            discovery=self.discovery, review=self.review, novelty=self.novelty,
            experiment=self.experiment,
            manuscript=self.manuscript,
        )

    async def close(self) -> None:
        await self.search_pipeline.close()
        await self.router.close()


def _sources(names: tuple[str, ...]) -> list[MetadataSource]:
    factories = {
        "semantic_scholar": SemanticScholarSource,
        "openalex": OpenAlexSource,
        "crossref": CrossrefSource,
        "arxiv": ArxivSource,
        "openreview": OpenReviewSource,
    }
    unknown = sorted(set(names) - set(factories))
    if unknown:
        raise ValueError(f"Unknown scholarly metadata sources: {', '.join(unknown)}")
    return [factories[name]() for name in names]
