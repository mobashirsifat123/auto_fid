"""Durable orchestration for the first AutoResearch milestone."""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, ArtifactReference, CandidateStatus, DownloadStatus, GapClassification, NoveltyCandidate, PaperRecord, PipelineKind, PriorArtStatus, ResearchGap, RunStatus, SeedPaper, ExperimentSpec, content_checksum, utc_now
from autoresearch.ingestion.chunking import DocumentChunk, chunk_document
from autoresearch.ingestion.document_parser import DocumentParser, ParsedDocument, ParsedPage
from autoresearch.ingestion.extraction import SEED_PROMPT_VERSION, SeedAnalysis, SeedAnalyzer
from autoresearch.papers.search_pipeline import DiscoveryResult, SearchStrategy, StudySearchPipeline
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.errors import ProviderError
from autoresearch.providers.redaction import redact_text
from autoresearch.providers.router import ModelRouter


def require_model_role(router: ModelRouter, role: ModelRole) -> None:
    """Actionable early diagnostic: fail fast with the remedy when a stage's
    required model role is unconfigured, instead of a downstream symptom."""
    try:
        router.identity(role)
    except ProviderError as exc:
        raise ProviderError(
            f"provider_unconfigured: {exc.args[0]} "
            "Copy .autoresearch/config.example.toml to .autoresearch/config.toml, "
            "fill the provider placeholders, and export the referenced key "
            "environment variables (see README 'Provider configuration'). "
            "Offline commands (discover --from-catalog, skill status, audit) "
            "work without a provider.",
            getattr(exc, "kind", None),
        ) from exc
from autoresearch.review.extraction import PaperReviewExtractor, ReviewBundle
from autoresearch.review.gap_analysis import GapAnalyzer
from autoresearch.review.similarity import ReviewRanker
from autoresearch.review.synthesis import ReviewSynthesisWriter
from autoresearch.storage.repository import ResearchRepository
from autoresearch.novelty.pipeline import NoveltyPipeline
from autoresearch.novelty.candidate_generator import LLMCandidateGenerator
from autoresearch.novelty.prior_art_checker import EvidenceBackedPriorArtChecker
from autoresearch.novelty.novelty_critic import ScientificCritic
from autoresearch.novelty.selection import PortfolioSelector
from autoresearch.experiments.pipeline import ExperimentPipeline
from autoresearch.experiments.planner import ExperimentSpecGenerator
from autoresearch.experiments.environment import IsolatedEnvironmentBuilder
from autoresearch.experiments.executor import DurableExperimentExecutor
from autoresearch.experiments.result_analysis import StatisticalAnalyzer
from autoresearch.experiments.report_writer import ScientificReportWriter
from autoresearch.experiments.policy import ActionType, ExecutionAction, specification_checksum
from autoresearch.reinvention.pipeline import ReinventionPipeline
from autoresearch.workers.local_typed import LocalTypedWorker

SEARCH_PROMPT_VERSION = "search-strategies-v1"
SEARCH_STRATEGIES_SCHEMA = {
    "type": "object",
    "properties": {
        "strategies": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}, "query": {"type": "string"}},
                "required": ["name", "query"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["strategies"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class DiscoveryOutcome:
    project_id: str
    run_id: str
    requested: int
    retrieved: int
    downloaded: int
    unavailable: int
    rejected: int
    failed: int
    artifact_path: str


@dataclass(frozen=True)
class ReviewOutcome:
    project_id: str
    run_id: str
    reviewed: int
    failed: int
    top_paper_ids: tuple[str, ...]
    gap: ResearchGap
    json_report: str
    markdown_report: str
    reused: int = 0


class SearchStrategyGenerator:
    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    async def generate(self, analysis: SeedAnalysis, *, project_id: str, run_id: str) -> tuple[SearchStrategy, ...]:
        payload = await self.router.structured_output(
            ModelRole.RESEARCH_REASONING,
            messages=[
                {"role": "system", "content": "Generate complementary scholarly-search queries. Queries must be concise and grounded in the supplied seed analysis."},
                {"role": "user", "content": f"Prompt version: {SEARCH_PROMPT_VERSION}\nTopic: {analysis.topic}\nQuestion: {analysis.research_question}\nMethods: {list(analysis.methods)}\nDatasets: {list(analysis.datasets)}\nKeywords: {list(analysis.keywords)}\nReturn strategies[] with unique name and query. Include topic, method, research-question, and limitation strategies."},
            ],
            schema={"name": "search_strategies", "schema": SEARCH_STRATEGIES_SCHEMA},
            project_id=project_id,
            run_id=run_id,
            operation="search_strategy_generation",
        )
        strategies: list[SearchStrategy] = []
        seen: set[str] = set()
        for raw in payload.get("strategies") or []:
            if not isinstance(raw, dict):
                continue
            name = " ".join(str(raw.get("name") or "").split())
            query = " ".join(str(raw.get("query") or "").split())
            if name and query and query.lower() not in seen:
                strategies.append(SearchStrategy(name, query))
                seen.add(query.lower())
        if len(strategies) < 3:
            raise ValueError("Provider must return at least three distinct search strategies.")
        return tuple(strategies[:8])


class DiscoveryService:
    def __init__(self, repository: ResearchRepository, router: ModelRouter, search_pipeline: StudySearchPipeline, *, parser: DocumentParser | None = None) -> None:
        self.repository = repository
        self.router = router
        self.search_pipeline = search_pipeline
        self.parser = parser or DocumentParser()
        self.seed_analyzer = SeedAnalyzer(router)
        self.strategy_generator = SearchStrategyGenerator(router)

    async def import_catalog(
        self,
        catalog_path: Path,
        *,
        project_title: str | None = None,
    ) -> DiscoveryOutcome:
        """Import papers from a local JSON catalog without external API calls."""
        catalog_path = catalog_path.expanduser().resolve()
        if not catalog_path.is_file():
            raise ValueError(f"Catalog file not found: {catalog_path}")
        with open(catalog_path, encoding="utf-8") as handle:
            catalog = json.load(handle)
        if not isinstance(catalog, list):
            raise ValueError("Catalog must be a JSON array of paper entries.")
        title = project_title or f"Local catalog: {catalog_path.name}"
        project = self.repository.create_project(title)
        run = self.repository.create_run(project.id, PipelineKind.DISCOVERY, {
            "seed_input": str(catalog_path), "limit": len(catalog), "since_year": None,
            "max_cost_usd": None, "max_runtime_seconds": None,
            "source": "local_catalog",
        })
        papers: list[PaperRecord] = []
        skipped = 0
        for entry in catalog:
            if not isinstance(entry, dict):
                skipped += 1
                continue
            raw_pid = str(entry.get("id") or uuid4().hex)
            paper_id = f"{project.id[:8]}_{raw_pid}"
            title_text = " ".join(str(entry.get("title") or "").split())
            if not title_text:
                skipped += 1
                continue
            authors_raw = entry.get("authors")
            if isinstance(authors_raw, str) and authors_raw.strip():
                authors = tuple(a.strip() for a in authors_raw.split(",") if a.strip())
            elif isinstance(authors_raw, list):
                authors = tuple(str(a) for a in authors_raw if str(a).strip())
            else:
                authors = ()
            year = None
            raw_year = entry.get("year")
            if raw_year is not None:
                try:
                    year = int(raw_year)
                except (TypeError, ValueError):
                    pass
            doi = str(entry.get("doi") or "").strip() or None
            venue = str(entry.get("venue") or "").strip()
            abstract = str(entry.get("abstract") or entry.get("raw_text") or "").strip()
            url = str(entry.get("pdf_url") or entry.get("url") or "").strip() or None
            source_names = ("local_catalog",)
            checksum = content_checksum(f"{paper_id}|{title_text}|{abstract[:200]}")
            paper = PaperRecord(
                id=paper_id, project_id=project.id, run_id=run.id,
                title=title_text, abstract=abstract, authors=authors,
                year=year, publication_date=None, doi=doi, arxiv_id=None,
                url=url, source_names=source_names,
                open_access_pdf_url=url, is_open_access=bool(url),
                citation_count=0, venue=venue, checksum=checksum,
            )
            papers.append(paper)
        if not papers:
            raise ValueError("No valid paper entries found in catalog.")
        imported = self.repository.import_papers_batch(papers)
        if papers:
            p0 = papers[0]
            # Provenance label only — catalog import makes no model call. With
            # no provider configured, record the deterministic local origin
            # instead of failing an otherwise fully offline import.
            try:
                extraction_provider, extraction_model = self.router.identity(ModelRole.LITERATURE_EXTRACTION)
            except ProviderError:
                extraction_provider, extraction_model = "local", "catalog_import"
            seed_prov = ArtifactProvenance.create(
                project_id=project.id, run_id=run.id,
                source_evidence=(f"catalog-checksum:{p0.checksum}",),
                provider=extraction_provider, model=extraction_model,
                prompt_version="catalog-v1", content={"title": p0.title}
            )
            seed = SeedPaper(
                id=uuid4().hex, project_id=project.id, run_id=run.id,
                input_type="catalog", input_value=str(catalog_path),
                title=p0.title, text=p0.abstract, topic="Catalog Literature",
                research_question=f"Literature analysis for {p0.title}",
                checksum=p0.checksum, provenance=seed_prov,
                doi=p0.doi, url=p0.url
            )
            self.repository.save_seed(seed)
        self.repository.finish_run(run.id)
        artifact_path = str(self.repository.artifact_root / project.id / run.id / "catalog_import.json")
        Path(artifact_path).parent.mkdir(parents=True, exist_ok=True)
        Path(artifact_path).write_text(json.dumps({
            "project_id": project.id, "run_id": run.id,
            "catalog_path": str(catalog_path), "entries": len(catalog),
            "imported": imported, "skipped": skipped,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        return DiscoveryOutcome(
            project_id=project.id, run_id=run.id,
            requested=len(catalog), retrieved=imported,
            downloaded=0, unavailable=0, rejected=skipped, failed=0,
            artifact_path=artifact_path,
        )

    async def discover(
        self,
        seed_input: str | Path,
        *,
        limit: int = 20,
        since_year: int | None = None,
        max_cost_usd: float | None = None,
        max_runtime_seconds: float | None = None,
    ) -> DiscoveryOutcome:
        parsed = self.parser.parse_seed(seed_input)
        project = self.repository.create_project(parsed.title)
        run = self.repository.create_run(project.id, PipelineKind.DISCOVERY, {
            "seed_input": str(seed_input), "limit": limit, "since_year": since_year,
            "max_cost_usd": max_cost_usd, "max_runtime_seconds": max_runtime_seconds,
        })
        if max_cost_usd is not None:
            self.repository.set_project_budget(project.id, hard_stop=max_cost_usd)
        return await self._execute(project.id, run.id, parsed, limit=limit, since_year=since_year, max_runtime_seconds=max_runtime_seconds)

    async def resume(self, run_id: str) -> DiscoveryOutcome:
        run = self.repository.get_run(run_id)
        if not run or run.kind is not PipelineKind.DISCOVERY:
            raise ValueError(f"Discovery run '{run_id}' was not found.")
        failed = [item for item in self.repository.checkpoints(run_id) if item["status"] == "failed"]
        if run.status is RunStatus.COMPLETED and not failed:
            return self._completed_outcome(run.project_id, run.id)
        data = dict(run.input_data)
        if data.get("max_cost_usd") is not None:
            self.repository.set_project_budget(run.project_id, hard_stop=float(data["max_cost_usd"]))
        self.repository.resume_run(run.id)
        parsed = self.parser.parse_seed(str(data["seed_input"]))
        return await self._execute(
            run.project_id,
            run.id,
            parsed,
            limit=int(data.get("limit", 20)),
            since_year=int(data["since_year"]) if data.get("since_year") is not None else None,
            max_runtime_seconds=float(data["max_runtime_seconds"]) if data.get("max_runtime_seconds") is not None else None,
        )

    async def _execute(self, project_id: str, run_id: str, parsed: ParsedDocument, *, limit: int, since_year: int | None, max_runtime_seconds: float | None) -> DiscoveryOutcome:
        guard = _ExecutionGuard(max_runtime_seconds)
        try:
            hydrated_checkpoint = self.repository.checkpoint(run_id, "seed_hydration")
            if hydrated_checkpoint and hydrated_checkpoint["status"] == "completed" and hydrated_checkpoint.get("payload"):
                parsed = _parsed_document(hydrated_checkpoint["payload"])
            else:
                self.repository.begin_checkpoint(run_id, "seed_hydration")
                parsed = await self._hydrate_identifier_seed(parsed)
                self.repository.complete_checkpoint(run_id, "seed_hydration", payload=_parsed_payload(parsed))
            guard.require_allowed("seed analysis")
            analysis_checkpoint = self.repository.checkpoint(run_id, "seed_analysis")
            if analysis_checkpoint and analysis_checkpoint["status"] == "completed" and analysis_checkpoint.get("payload"):
                analysis = _seed_analysis(analysis_checkpoint["payload"])
            else:
                self.repository.begin_checkpoint(run_id, "seed_analysis")
                analysis = await self.seed_analyzer.analyze(parsed, project_id=project_id, run_id=run_id)
                self.repository.complete_checkpoint(run_id, "seed_analysis", payload=asdict(analysis))
            extraction_provider, extraction_model = self.router.identity(ModelRole.LITERATURE_EXTRACTION)
            analysis_content = {"title": analysis.title, "topic": analysis.topic, "research_question": analysis.research_question, "methods": analysis.methods, "datasets": analysis.datasets, "citations": analysis.citations, "limitations": analysis.limitations, "keywords": analysis.keywords}
            seed = self.repository.get_seed(project_id)
            if not seed:
                seed_provenance = ArtifactProvenance.create(project_id=project_id, run_id=run_id, source_evidence=(f"seed-checksum:{parsed.checksum}",), provider=extraction_provider, model=extraction_model, prompt_version=SEED_PROMPT_VERSION, content=analysis_content)
                seed = SeedPaper(id=uuid4().hex, project_id=project_id, run_id=run_id, input_type=parsed.input_type, input_value=parsed.input_value, title=analysis.title, text=parsed.text, topic=analysis.topic, research_question=analysis.research_question, checksum=parsed.checksum, provenance=seed_provenance, doi=parsed.doi, url=parsed.url, methods=analysis.methods, datasets=analysis.datasets, citations=analysis.citations, limitations=analysis.limitations)
                self.repository.save_seed(seed)
            guard.require_allowed("search strategy generation")
            strategy_checkpoint = self.repository.checkpoint(run_id, "search_strategies")
            if strategy_checkpoint and strategy_checkpoint["status"] == "completed" and strategy_checkpoint.get("payload"):
                strategies = tuple(SearchStrategy(str(item["name"]), str(item["query"])) for item in strategy_checkpoint["payload"])
            else:
                self.repository.begin_checkpoint(run_id, "search_strategies")
                strategies = await self.strategy_generator.generate(analysis, project_id=project_id, run_id=run_id)
                self.repository.complete_checkpoint(run_id, "search_strategies", payload=[asdict(item) for item in strategies])
            guard.require_allowed("scholarly discovery")
            result = await self.search_pipeline.discover(project_id=project_id, run_id=run_id, seed=seed, strategies=strategies, limit=limit, since_year=since_year)
            artifact = self._write_discovery_artifact(seed, analysis, strategies, result)
            self.repository.complete_checkpoint(run_id, "discovery_artifact", payload={"path": artifact.path, "checksum": artifact.checksum})
            self.repository.finish_run(run_id)
            return DiscoveryOutcome(
                project_id, run_id, limit, len(result.papers), result.downloaded_count,
                sum(item.status is DownloadStatus.UNAVAILABLE for item in result.downloads),
                sum(item.status is DownloadStatus.REJECTED for item in result.downloads),
                sum(item.status is DownloadStatus.FAILED for item in result.downloads), artifact.path,
            )
        except BaseException as exc:
            self.repository.finish_run(run_id, error=redact_text(f"{type(exc).__name__}: {exc}"))
            raise

    def _completed_outcome(self, project_id: str, run_id: str) -> DiscoveryOutcome:
        artifacts = [item for item in self.repository.artifacts_for_run(run_id) if item.artifact_type == "discovery_summary"]
        if not artifacts:
            raise RuntimeError(f"Completed discovery run '{run_id}' has no discovery artifact.")
        payload = json.loads(Path(artifacts[-1].path).read_text(encoding="utf-8"))
        statuses = [item.get("status") for item in payload.get("downloads") or []]
        return DiscoveryOutcome(project_id, run_id, int(payload["requested"]), int(payload["retrieved"]), statuses.count("downloaded"), statuses.count("unavailable"), statuses.count("rejected"), statuses.count("failed"), artifacts[-1].path)

    async def _hydrate_identifier_seed(self, parsed: ParsedDocument) -> ParsedDocument:
        if parsed.text:
            return parsed
        query = parsed.doi or parsed.url or parsed.input_value
        results = await asyncio.gather(*(source.search(query, limit=3) for source in self.search_pipeline.sources), return_exceptions=True)
        candidates = [paper for result in results if not isinstance(result, Exception) for paper in result.papers]
        if parsed.doi:
            exact = [paper for paper in candidates if paper.doi == parsed.doi]
            candidates = exact or candidates
        if not candidates:
            raise ValueError("Seed DOI/URL could not be resolved to scholarly metadata.")
        paper = max(candidates, key=lambda item: len(item.abstract))
        return ParsedDocument(parsed.input_type, parsed.input_value, paper.title or parsed.title, (ParsedPage(1, paper.abstract),), parsed.checksum, parsed.doi or paper.doi, parsed.url or paper.url)

    def _write_discovery_artifact(self, seed: SeedPaper, analysis: SeedAnalysis, strategies: Sequence[SearchStrategy], result: DiscoveryResult) -> ArtifactReference:
        provider, model = self.router.identity(ModelRole.RESEARCH_REASONING)
        source_summary: dict[str, dict[str, int]] = {}
        for event in self.repository.search_events(result.run_id):
            summary = source_summary.setdefault(event["source"], {"requests": 0, "successes": 0, "failures": 0, "records": 0})
            summary["requests"] += 1
            summary["records"] += int(event["result_count"])
            summary["failures" if event.get("error") else "successes"] += 1
        payload = {
            "project_id": seed.project_id, "run_id": seed.run_id, "seed_id": seed.id,
            "provider": provider, "model": model, "prompt_version": SEARCH_PROMPT_VERSION,
            "timestamp": utc_now(),
            "source_evidence": [seed.id, *(paper.id for paper in result.papers)],
            "seed_analysis": analysis.__dict__, "strategies": [item.__dict__ for item in strategies],
            "requested": result.requested, "retrieved": len(result.papers), "downloaded": result.downloaded_count,
            "downloads": [{"paper_id": item.paper_id, "status": item.status.value, "source": item.source, "legal_basis": item.legal_basis, "reason": item.reason, "checksum": item.checksum} for item in result.downloads],
            "source_errors": result.source_errors,
            "source_summary": source_summary,
            "deduplication": {
                "raw_candidates": result.raw_candidate_count,
                "unique_candidates": result.unique_candidate_count,
                "duplicates_merged": result.duplicate_count,
                "seed_records_excluded": result.seed_excluded_count,
                "selected": len(result.papers),
            },
        }
        root = self.repository.artifact_root / seed.project_id / seed.run_id
        root.mkdir(parents=True, exist_ok=True)
        path = root / "discovery.json"
        text = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        path.write_text(text, encoding="utf-8")
        artifact = ArtifactReference(
            content_checksum(f"discovery_summary|{seed.run_id}")[:32],
            seed.project_id,
            seed.run_id,
            "discovery_summary",
            str(path),
            tuple(payload["source_evidence"]),
            provider,
            model,
            SEARCH_PROMPT_VERSION,
            content_checksum(text),
        )
        self.repository.save_artifact(artifact)
        return artifact


class LiteratureReviewService:
    def __init__(self, repository: ResearchRepository, router: ModelRouter, *, parser: DocumentParser | None = None) -> None:
        self.repository = repository
        self.router = router
        self.parser = parser or DocumentParser()
        self.extractor = PaperReviewExtractor(router)
        self.ranker = ReviewRanker(router)
        self.gap_analyzer = GapAnalyzer(router)
        self.writer = ReviewSynthesisWriter(repository)
        self.review_semaphore = asyncio.Semaphore(4)

    async def review(
        self,
        project_id: str,
        *,
        top_k: int = 5,
        hierarchical: bool = False,
        max_cost_usd: float | None = None,
        max_runtime_seconds: float | None = None,
        fresh: bool = False,
    ) -> ReviewOutcome:
        require_model_role(self.router, ModelRole.LITERATURE_EXTRACTION)
        if not 1 <= top_k <= 15:
            raise ValueError("top_k must be between 1 and 15.")
        project = self.repository.get_project(project_id)
        seed = self.repository.get_seed(project_id)
        if not project or not seed:
            raise ValueError("Project or seed paper was not found.")
        papers = self.repository.list_papers(project_id)
        if not papers:
            raise ValueError("Run discovery before literature review.")
        run = self.repository.create_run(project_id, PipelineKind.REVIEW, {
            "top_k": top_k, "paper_count": len(papers), "hierarchical": hierarchical,
            "max_cost_usd": max_cost_usd, "max_runtime_seconds": max_runtime_seconds,
        })
        if max_cost_usd is not None:
            self.repository.set_project_budget(project_id, hard_stop=max_cost_usd)
        return await self._execute(run.id, project_id, top_k=top_k, hierarchical=hierarchical, max_runtime_seconds=max_runtime_seconds, fresh=fresh)

    async def resume(self, run_id: str) -> ReviewOutcome:
        run = self.repository.get_run(run_id)
        if not run or run.kind is not PipelineKind.REVIEW:
            raise ValueError(f"Review run '{run_id}' was not found.")
        failed = [item for item in self.repository.checkpoints(run_id) if item["status"] == "failed"]
        if run.status is RunStatus.COMPLETED and not failed:
            return self._completed_outcome(run.project_id, run.id)
        data = dict(run.input_data)
        if data.get("max_cost_usd") is not None:
            self.repository.set_project_budget(run.project_id, hard_stop=float(data["max_cost_usd"]))
        self.repository.resume_run(run.id)
        return await self._execute(
            run.id,
            run.project_id,
            top_k=int(data.get("top_k", 5)),
            hierarchical=bool(data.get("hierarchical", False)),
            max_runtime_seconds=float(data["max_runtime_seconds"]) if data.get("max_runtime_seconds") is not None else None,
        )

    async def _execute(self, run_id: str, project_id: str, *, top_k: int, hierarchical: bool, max_runtime_seconds: float | None, fresh: bool = False) -> ReviewOutcome:
        guard = _ExecutionGuard(max_runtime_seconds)
        project = self.repository.get_project(project_id)
        seed = self.repository.get_seed(project_id)
        papers = self.repository.list_papers(project_id)
        if not project or not seed or not papers:
            raise ValueError("Project discovery state is incomplete.")
        try:
            guard.require_allowed("paper review")
            results = await asyncio.gather(*(self._review_one(paper, run_id, hierarchical=hierarchical, allow_reuse=not fresh) for paper in papers), return_exceptions=True)
            bundles: list[ReviewBundle] = []
            for paper, result in zip(papers, results):
                if isinstance(result, Exception):
                    reason = redact_text(f"{type(result).__name__}: {result}")
                    self.repository.record_review_failure(project_id=project_id, run_id=run_id, paper_id=paper.id, reason=reason)
                else:
                    bundles.append(result)
            if len(bundles) < top_k:
                raise RuntimeError(f"Only {len(bundles)} papers produced evidence-backed reviews; {top_k} are required.")
            reviews = [bundle.review for bundle in bundles]
            claims = [claim for bundle in bundles for claim in bundle.claims]
            reviewed_papers = [paper for paper in papers if paper.id in {review.paper_id for review in reviews}]
            guard.require_allowed("ranking embeddings")
            embedding_checkpoint = self.repository.checkpoint(run_id, "ranking_embeddings")
            cached_vectors = None
            if embedding_checkpoint and embedding_checkpoint["status"] == "completed" and embedding_checkpoint.get("payload"):
                cached_vectors = embedding_checkpoint["payload"].get("vectors")
            else:
                self.repository.begin_checkpoint(run_id, "ranking_embeddings")
            try:
                all_scores = await self.ranker.rank(
                    seed=seed,
                    papers=reviewed_papers,
                    reviews=reviews,
                    run_id=run_id,
                    embedded_vectors=cached_vectors,
                    embedding_sink=lambda vectors: self.repository.complete_checkpoint(run_id, "ranking_embeddings", payload={"vectors": vectors}),
                )
            except Exception as exc:
                self.repository.fail_checkpoint(run_id, "ranking_embeddings", error=redact_text(f"{type(exc).__name__}: {exc}"))
                raise
            for score in all_scores:
                self.repository.save_score(score)
            self.repository.complete_checkpoint(run_id, "ranking", payload={"paper_ids": [score.paper_id for score in all_scores]})
            top_scores = all_scores[:top_k]
            top_ids = {score.paper_id for score in top_scores}
            top_papers = [paper for paper in reviewed_papers if paper.id in top_ids]
            top_reviews = [review for review in reviews if review.paper_id in top_ids]
            top_claims = [claim for claim in claims if claim.paper_id in top_ids]
            guard.require_allowed("gap analysis")
            gap = self.repository.gap_for_run(run_id)
            if not gap or not (self.repository.checkpoint(run_id, "gap_analysis") or {}).get("status") == "completed":
                self.repository.begin_checkpoint(run_id, "gap_analysis")
                try:
                    gap = await self.gap_analyzer.analyze(project_id=project_id, run_id=run_id, papers=top_papers, reviews=top_reviews, claims=top_claims, scores=top_scores)
                    self.repository.save_gap(gap)
                    self.repository.complete_checkpoint(run_id, "gap_analysis", payload={"gap_id": gap.id, "checksum": gap.provenance.checksum})
                except Exception as exc:
                    self.repository.fail_checkpoint(run_id, "gap_analysis", error=redact_text(f"{type(exc).__name__}: {exc}"))
                    raise
            json_artifact, markdown_artifact = self.writer.write(project_id=project_id, run_id=run_id, papers=top_papers, reviews=top_reviews, scores=top_scores, claims=top_claims, gap=gap)
            self.repository.complete_checkpoint(run_id, "review_artifacts", payload={"json": json_artifact.path, "markdown": markdown_artifact.path})
            self.repository.finish_run(run_id)
            reused = sum(1 for review in reviews if review.run_id != run_id)
            return ReviewOutcome(project_id, run_id, len(reviews), len(results) - len(reviews), tuple(score.paper_id for score in top_scores), gap, json_artifact.path, markdown_artifact.path, reused)
        except BaseException as exc:
            self.repository.finish_run(run_id, error=redact_text(f"{type(exc).__name__}: {exc}"))
            raise

    async def _review_one(self, paper, run_id: str, *, hierarchical: bool, allow_reuse: bool = True) -> ReviewBundle:
        checkpoint = self.repository.checkpoint(run_id, "paper_review", paper.id)
        existing = self.repository.review_for_paper(run_id, paper.id)
        if checkpoint and checkpoint["status"] == "completed" and existing:
            return ReviewBundle(existing, tuple(self.repository.claims(paper.project_id, ids=existing.evidence_claim_ids)))
        if allow_reuse:
            # Cross-run reuse: a verbatim-verified review of the SAME paper
            # under the SAME review prompt version is immutable evidence —
            # re-running review must not pay for it again. Re-running with
            # --fresh forces full re-extraction.
            cached = self.repository.latest_reusable_review(paper, hierarchical=hierarchical)
            if cached is not None:
                self.repository.begin_checkpoint(run_id, "paper_review", paper.id)
                self.repository.complete_checkpoint(run_id, "paper_review", paper.id, payload={"review_id": cached.id, "claim_ids": list(cached.evidence_claim_ids), "reused_from_run": cached.run_id})
                return ReviewBundle(cached, tuple(self.repository.claims(paper.project_id, ids=cached.evidence_claim_ids)))
        self.repository.begin_checkpoint(run_id, "paper_review", paper.id)
        try:
            async with self.review_semaphore:
                chunks: tuple[DocumentChunk, ...] = ()
                download = self.repository.download_for_paper(paper.id)
                if download and download.status is DownloadStatus.DOWNLOADED and download.path:
                    try:
                        chunks = chunk_document(self.parser.parse_seed(download.path))
                    except Exception:
                        chunks = ()
                bundle = await self.extractor.extract(paper, run_id=run_id, chunks=chunks, fallback_text=paper.abstract, hierarchical=hierarchical)
            for claim in bundle.claims:
                self.repository.save_claim(claim)
            self.repository.save_review(bundle.review)
            self.repository.clear_review_failures(run_id, paper.id)
            self.repository.complete_checkpoint(run_id, "paper_review", paper.id, payload={"review_id": bundle.review.id, "claim_ids": bundle.review.evidence_claim_ids, "hierarchical": bundle.review.provenance.prompt_version == "paper-review-hierarchical-v1"})
            return bundle
        except Exception as exc:
            self.repository.fail_checkpoint(run_id, "paper_review", paper.id, error=redact_text(f"{type(exc).__name__}: {exc}"))
            raise

    def _completed_outcome(self, project_id: str, run_id: str) -> ReviewOutcome:
        gap = self.repository.gap_for_run(run_id)
        artifacts = self.repository.artifacts_for_run(run_id)
        json_artifacts = [item for item in artifacts if item.artifact_type == "gap_report_json"]
        markdown_artifacts = [item for item in artifacts if item.artifact_type == "gap_report_markdown"]
        if not gap or not json_artifacts or not markdown_artifacts:
            raise RuntimeError(f"Completed review run '{run_id}' has incomplete artifacts.")
        scores = self.repository.scores(project_id, run_id)
        reviews = self.repository.reviews(project_id, run_id)
        failures = self.repository.review_failures(run_id)
        run = self.repository.get_run(run_id)
        top_k = int(run.input_data.get("top_k", 5)) if run else 5
        return ReviewOutcome(project_id, run_id, len(reviews), len(failures), tuple(score.paper_id for score in scores[:top_k]), gap, json_artifacts[-1].path, markdown_artifacts[-1].path)


class _ExecutionGuard:
    def __init__(self, max_runtime_seconds: float | None) -> None:
        if max_runtime_seconds is not None and max_runtime_seconds <= 0:
            raise ValueError("max_runtime_seconds must be positive.")
        self.max_runtime_seconds = max_runtime_seconds
        self.started = time.monotonic()

    def require_allowed(self, operation: str) -> None:
        if self.max_runtime_seconds is None:
            return
        elapsed = time.monotonic() - self.started
        if elapsed >= self.max_runtime_seconds:
            raise TimeoutError(f"Runtime limit of {self.max_runtime_seconds:g}s reached before {operation}.")


def _parsed_payload(document: ParsedDocument) -> dict:
    return {
        "input_type": document.input_type,
        "input_value": document.input_value,
        "title": document.title,
        "pages": [asdict(page) for page in document.pages],
        "checksum": document.checksum,
        "doi": document.doi,
        "url": document.url,
    }


def _parsed_document(payload: dict) -> ParsedDocument:
    return ParsedDocument(
        str(payload["input_type"]),
        str(payload["input_value"]),
        str(payload["title"]),
        tuple(ParsedPage(int(page["page"]), str(page["text"])) for page in payload.get("pages") or []),
        str(payload["checksum"]),
        str(payload["doi"]) if payload.get("doi") else None,
        str(payload["url"]) if payload.get("url") else None,
    )


def _seed_analysis(payload: dict) -> SeedAnalysis:
    return SeedAnalysis(
        str(payload["title"]),
        str(payload["topic"]),
        str(payload["research_question"]),
        tuple(str(item) for item in payload.get("methods") or []),
        tuple(str(item) for item in payload.get("datasets") or []),
        tuple(str(item) for item in payload.get("citations") or []),
        tuple(str(item) for item in payload.get("limitations") or []),
        tuple(str(item) for item in payload.get("keywords") or []),
    )


class NoveltyService:
    def __init__(self, repository: ResearchRepository, router: ModelRouter, search_pipeline: StudySearchPipeline) -> None:
        self.repository = repository
        self.router = router
        self.pipeline = NoveltyPipeline(
            repository=repository,
            generator=LLMCandidateGenerator(router),
            prior_art_checker=EvidenceBackedPriorArtChecker(router, search_pipeline),
            critic=ScientificCritic(router),
            selector=PortfolioSelector(repository),
        )

    async def import_gap(self, project_id: str, gap_data: dict[str, Any]) -> ResearchGap:
        """Import a research gap from external data (e.g., from a prior literature review).

        The gap row references pipeline_runs, so the import gets its own
        completed run instead of a dangling run_id that would violate the
        foreign key (research_gaps.run_id REFERENCES pipeline_runs(id)).
        """
        statement = " ".join(str(gap_data.get("statement") or gap_data.get("gap_statement") or "").split())
        if not statement:
            raise ValueError("Gap data must include a 'statement' or 'gap_statement' field.")
        why_unsolved = " ".join(str(gap_data.get("why_unsolved") or gap_data.get("why_existing_work_does_not_solve_it") or "").split())
        missing_experiment = " ".join(str(gap_data.get("missing_experiment") or "").split())
        falsification_test = " ".join(str(gap_data.get("falsification_test") or "").split())
        confidence = float(gap_data.get("confidence", 0.5))
        classification = GapClassification.EVIDENCE_SUPPORTED if gap_data.get("supporting_paper_ids") else GapClassification.HYPOTHESIS
        run = self.repository.create_run(project_id, PipelineKind.REVIEW, {"kind": "gap_import"})
        self.repository.finish_run(run.id)
        provenance = ArtifactProvenance.create(
            project_id=project_id, run_id=run.id,
            source_evidence=(), provider="manual_import", model="user",
            prompt_version="imported-gap-v1",
            content={"statement": statement, "why_unsolved": why_unsolved,
                     "missing_experiment": missing_experiment,
                     "falsification_test": falsification_test},
        )
        gap = ResearchGap(
            id=uuid4().hex, project_id=project_id, run_id=run.id,
            statement=statement,
            supporting_paper_ids=tuple(str(p) for p in (gap_data.get("supporting_paper_ids") or [])),
            supporting_claim_ids=tuple(str(c) for c in (gap_data.get("supporting_claim_ids") or [])),
            why_unsolved=why_unsolved, missing_experiment=missing_experiment,
            falsification_test=falsification_test, confidence=confidence,
            counter_evidence_claim_ids=(), classification=classification,
            provenance=provenance,
        )
        self.repository.save_gap(gap)
        return gap

    async def generate_portfolio(self, project_id: str, *, gap_id: str | None = None, run_id: str | None = None) -> str:
        require_model_role(self.router, ModelRole.RESEARCH_REASONING)
        gap = self.repository.row("SELECT * FROM research_gaps WHERE project_id=? AND (? IS NULL OR id=?) ORDER BY confidence DESC LIMIT 1", (project_id, gap_id, gap_id))
        if not gap:
            raise ValueError(f"No research gap found for project {project_id}.")
        
        # We need the full object, since gap is a dict right now (returned from row).
        # We can construct it, but easier to use a repository method if one existed, 
        # or simply adapt it:
        gap_obj = ResearchGap(gap["id"], gap["project_id"], gap["run_id"], gap["statement"], tuple(json.loads(gap["supporting_paper_ids_json"])), tuple(json.loads(gap["supporting_claim_ids_json"])), gap["why_unsolved"], gap["missing_experiment"], gap["falsification_test"], float(gap["confidence"]), tuple(json.loads(gap["counter_evidence_claim_ids_json"])), GapClassification(gap["classification"]), ArtifactProvenance(**json.loads(gap["provenance_json"])))
        
        run = self.repository.get_run(run_id) if run_id else self.repository.create_run(project_id, PipelineKind.NOVELTY, {"gap_id": gap_obj.id})
        if not run:
            raise ValueError(f"Novelty run {run_id} not found.")
        owns_run = run_id is None
        try:
            # We mock the run_id inside gap so provenance links to this run.
            run_gap = ResearchGap(gap_obj.id, project_id, run.id, gap_obj.statement, gap_obj.supporting_paper_ids, gap_obj.supporting_claim_ids, gap_obj.why_unsolved, gap_obj.missing_experiment, gap_obj.falsification_test, gap_obj.confidence, gap_obj.counter_evidence_claim_ids, gap_obj.classification, gap_obj.provenance)
            portfolio = await self.pipeline.generate_portfolio(run_gap)
            if owns_run:
                self.repository.finish_run(run.id)
            return portfolio.id
        except Exception as exc:
            if owns_run:
                self.repository.finish_run(run.id, error=f"{type(exc).__name__}: {exc}")
            raise


class ExperimentService:
    def __init__(self, repository: ResearchRepository, router: ModelRouter, workspace_root: Path, worker: LocalTypedWorker | None = None) -> None:
        self.repository = repository
        self.router = router
        self.pipeline = ExperimentPipeline(
            repository=repository,
            planner=ExperimentSpecGenerator(router),
            environment_builder=IsolatedEnvironmentBuilder(workspace_root),
            executor=DurableExperimentExecutor(repository, None),
            analyzer=StatisticalAnalyzer(router),
            report_writer=ScientificReportWriter(router),
        )
        self.worker = worker

    async def import_candidate(self, candidate_data: dict[str, Any]) -> NoveltyCandidate:
        """Import a novelty candidate from external data (e.g., from a prior analysis)."""
        method = " ".join(str(candidate_data.get("method") or candidate_data.get("title") or "").split())
        if not method:
            raise ValueError("Candidate data must include a 'method' or 'title' field.")
        project_id = str(candidate_data.get("project_id") or "")
        if not project_id:
            raise ValueError("Candidate data must include a 'project_id' field.")
        # The candidate row references pipeline_runs; a dangling run_id would
        # violate the foreign key, so imports get their own completed run.
        # A supplied run must belong to the same project and be a novelty run,
        # otherwise the candidate would silently attach to a foreign scope.
        supplied_run_id = str(candidate_data.get("run_id") or "")
        supplied_run = self.repository.get_run(supplied_run_id) if supplied_run_id else None
        if supplied_run is not None and (supplied_run.project_id != project_id or supplied_run.kind is not PipelineKind.NOVELTY):
            raise ValueError(
                "run_mismatch: supplied run_id must reference a novelty run of the same project."
            )
        owns_run = supplied_run is None
        run_id = supplied_run_id if supplied_run is not None else self.repository.create_run(
            project_id, PipelineKind.NOVELTY, {"kind": "candidate_import"},
        ).id
        try:
            return self._save_imported_candidate(candidate_data, project_id, run_id)
        finally:
            if owns_run:
                self.repository.finish_run(run_id)

    def _save_imported_candidate(self, candidate_data: dict[str, Any], project_id: str, run_id: str) -> NoveltyCandidate:
        method = " ".join(str(candidate_data.get("method") or candidate_data.get("title") or "").split())
        if not method:
            raise ValueError("Candidate data must include a 'method' or 'title' field.")
        candidate = NoveltyCandidate(
            id=uuid4().hex, project_id=project_id, run_id=run_id,
            gap_id=str(candidate_data.get("gap_id") or ""),
            strategy=str(candidate_data.get("strategy") or "imported"),
            method=method,
            causal_mechanism=" ".join(str(candidate_data.get("causal_mechanism") or candidate_data.get("mechanism") or "").split()),
            target_limitation=" ".join(str(candidate_data.get("target_limitation") or "").split()),
            difference_from_existing=" ".join(str(candidate_data.get("difference_from_existing") or "").split()),
            expected_effect=str(candidate_data.get("expected_effect") or ""),
            falsification_condition=" ".join(str(candidate_data.get("falsification_condition") or "").split()),
            minimum_validating_experiment=" ".join(str(candidate_data.get("minimum_validating_experiment") or "").split()),
            required_ablations=tuple(str(item) for item in candidate_data.get("required_ablations") or []),
            implementation_complexity=str(candidate_data.get("implementation_complexity") or "unknown"),
            expected_compute=str(candidate_data.get("expected_compute") or "unknown"),
            risks_and_objections=tuple(str(item) for item in candidate_data.get("risks_and_objections") or []),
            status=CandidateStatus.GENERATED,
            provenance=ArtifactProvenance.create(
                project_id=project_id, run_id=run_id,
                source_evidence=(), provider="manual_import", model="user",
                prompt_version="imported-candidate-v1",
                content={"method": method},
            ),
        )
        self.repository.save_novelty_candidate(candidate)
        return candidate

    def action_plan(self, spec: ExperimentSpec) -> tuple[ExecutionAction, ...]:
        if not spec.expected_outputs:
            raise ValueError("experiment_action_missing: expected_outputs is empty")
        scripts = [item for item in spec.expected_outputs if str(item).endswith(".py")]
        if scripts:
            return (ExecutionAction(ActionType.RUN_PYTHON_SCRIPT, str(scripts[0]), ()),)
        return (ExecutionAction(ActionType.RUN_PYTHON_SCRIPT, "scripts/run_experiment.py", ()),)

    def _execution_worker(self, spec: ExperimentSpec) -> str:
        """Bind approval to the worker that will actually execute.

        ``spec.target_worker`` is an LLM-chosen preference and may name a worker
        that is not configured (e.g. a hallucinated GPU worker). The checksum
        must use the real worker identity, or approval never matches execution.
        """
        if self.worker is not None:
            return self.worker.worker_id
        return spec.target_worker or "local"

    def approval_preview(self, run_id: str) -> dict[str, object]:
        run = self.repository.get_run(run_id)
        if not run:
            raise ValueError(f"Experiment run {run_id} not found")
        spec = self.repository.row("SELECT * FROM experiment_specs WHERE run_id=? ORDER BY rowid DESC LIMIT 1", (run_id,))
        if not spec:
            raise ValueError("approval_required: plan the experiment before approving execution")
        from autoresearch.storage.repository import _experiment_spec
        typed_spec = _experiment_spec(spec)
        actions = self.action_plan(typed_spec)
        worker = self._execution_worker(typed_spec)
        checksum = specification_checksum(typed_spec, actions, worker)
        return {"run_id": run_id, "spec_id": typed_spec.id, "candidate_id": typed_spec.candidate_id, "checksum": checksum, "worker": worker, "actions": [{"kind": action.kind.value, "target": action.target, "arguments": list(action.arguments)} for action in actions], "budget": 0.0, "expires_at": None}

    def approval_preview_for_candidate(self, candidate_id: str) -> dict[str, object]:
        spec = self.repository.get_experiment_spec(candidate_id)
        if not spec:
            raise ValueError("approval_required: plan the experiment before approving execution")
        actions = self.action_plan(spec)
        worker = self._execution_worker(spec)
        checksum = specification_checksum(spec, actions, worker)
        return {"run_id": spec.run_id, "spec_id": spec.id, "candidate_id": candidate_id, "checksum": checksum, "worker": worker, "actions": [{"kind": action.kind.value, "target": action.target, "arguments": list(action.arguments)} for action in actions], "budget": 3600.0, "expires_at": None}

    async def execute_candidate(self, candidate_id: str, *, run_id: str | None = None, mode: str = "plan") -> str:
        require_model_role(self.router, ModelRole.RESEARCH_REASONING)
        candidate_row = self.repository.row("SELECT * FROM novelty_candidates WHERE id=?", (candidate_id,))
        if not candidate_row:
            raise ValueError(f"Candidate {candidate_id} not found.")
        
        from autoresearch.storage.repository import _novelty_candidate
        candidate = _novelty_candidate(candidate_row)
        
        run = self.repository.get_run(run_id) if run_id else self.repository.create_run(candidate.project_id, PipelineKind.EXPERIMENT, {"candidate_id": candidate_id})
        if not run:
            raise ValueError(f"Experiment run {run_id} not found.")
        owns_run = run_id is None
        try:
            # mock candidate to use this run_id for tracking
            from dataclasses import replace
            run_candidate = replace(candidate, run_id=run.id)
            spec = self.repository.get_experiment_spec(candidate_id) if mode == "execute" else None
            if spec is None:
                spec = await self.pipeline.plan_for_candidate(run_candidate)
                self.repository.save_experiment_spec(spec)
            if mode in {"inspect", "validate"}:
                manifest = await self.pipeline.environment_builder.prepare(spec)
                self.repository.save_environment_manifest(manifest)
            if mode == "report":
                raise ValueError("insufficient_evidence: no completed statistical analysis exists for this candidate")
            if mode == "execute":
                if not self.worker:
                    raise RuntimeError("worker_unavailable: no policy-controlled execution worker is configured")
                preview = self.approval_preview_for_candidate(candidate_id)
                approval = self.repository.approval_by_checksum(str(preview["checksum"]))
                if not approval or approval.get("status") != "approved":
                    raise PermissionError("approval_required: exact experiment approval is required before execution")
                from datetime import datetime, timezone
                if approval.get("expires_at") and datetime.fromisoformat(str(approval["expires_at"])) < datetime.now(timezone.utc):
                    raise PermissionError("approval_expired: experiment approval has expired")
                result = await self.worker.run(spec, self.action_plan(spec), approved_checksum=str(preview["checksum"]), timeout_seconds=int(approval.get("approved_budget") or 3600))
                if result.get("state") != "COMPLETED":
                    raise RuntimeError("execution_failed: worker did not complete the approved scientific action")
                # Completion only proves that a command ended successfully.  It
                # deliberately creates no scientific result: artifact ingestion
                # and identity validation are required before analysis.
                self.repository.append_api_event(run.id, "experiment.execution_completed", {
                    "job_id": result.get("job_id"), "scientific_result": "INSUFFICIENT_EVIDENCE",
                    "reason": "no validated measurement artifact ingested",
                })
            if owns_run:
                self.repository.finish_run(run.id)
            return spec.id
        except Exception as exc:
            if owns_run:
                self.repository.finish_run(run.id, error=f"{type(exc).__name__}: {exc}")
            raise


class ReinventionService:
    def __init__(self, repository: ResearchRepository, router: ModelRouter) -> None:
        self.repository = repository
        self.router = router
        self.pipeline = ReinventionPipeline(repository, router)

    async def reinvent(self, candidate_id: str) -> dict[str, Any]:
        candidate = self.repository.get_novelty_candidate(candidate_id)
        if not candidate:
            raise ValueError(f"Novelty candidate '{candidate_id}' not found.")

        run = self.repository.create_run(candidate.project_id, PipelineKind.REINVENTION, input_data={"candidate_id": candidate_id})
        try:
            analysis, next_candidate = await self.pipeline.analyze_and_reinvent(candidate.id, run_id=run.id)
            self.repository.finish_run(run.id)
            return {
                "project_id": candidate.project_id,
                "run_id": run.id,
                "reinvention_id": analysis.id,
                "action": analysis.action.value,
                "root_cause_diagnosis": analysis.root_cause_diagnosis,
                "empirical_lessons": list(analysis.empirical_lessons),
                "next_generation_hypothesis": analysis.next_generation_hypothesis,
                "next_candidate_id": next_candidate.id if next_candidate else None,
                "next_candidate_method": next_candidate.method if next_candidate else None,
            }
        except Exception as exc:
            self.repository.finish_run(run.id, error=f"{type(exc).__name__}: {exc}")
            raise
