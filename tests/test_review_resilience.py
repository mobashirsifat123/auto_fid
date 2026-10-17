"""Review-skill resilience tests — the fixes for 'it never finishes'.

Covers, against the real _execute_review orchestration (offline, patched
collaborators):
- partial paper failures do NOT abort the run (top-5 + gap still produced),
- systemic provider errors (dead credential, exhausted budget) fail FAST
  instead of grinding through N identical failures,
- transient errors get one review-level retry,
- gap analysis retries transient failures,
- stale 'running' review runs from a dead process are retired (checkpoints
  and verified reviews are reusable across runs, so nothing is lost).
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import pytest

from autoresearch.application.services import LiteratureReviewService
from autoresearch.domain.models import (
    ArtifactProvenance,
    PaperRecord,
    PipelineKind,
    RunStatus,
    SeedPaper,
)
from autoresearch.providers.errors import ErrorKind, ProviderError
from autoresearch.providers.registry import ProviderRegistry
from autoresearch.providers.router import ModelRouter
from autoresearch.providers.testing import FakeLLMProvider
from autoresearch.storage import ResearchRepository


def _bundle(paper):
    return SimpleNamespace(review=SimpleNamespace(paper_id=paper.id, run_id=paper.run_id), claims=())


def _seed_environment(tmp_path, count=8):
    repository = ResearchRepository(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    project = repository.create_project("resilience")
    run = repository.create_run(project.id, PipelineKind.REVIEW, {})
    provenance = ArtifactProvenance.create(
        project_id=project.id, run_id=run.id, source_evidence=(), provider="test", model="test", prompt_version="v1", content={},
    )
    repository.save_seed(SeedPaper(
        id=uuid4().hex, project_id=project.id, run_id=run.id, input_type="pdf",
        input_value="seed.pdf", title="Seed", text="seed text", topic="t",
        research_question="q", checksum="c", provenance=provenance,
    ))
    for index in range(count):
        repository.save_paper(PaperRecord(
            id=f"paper-{index}", project_id=project.id, run_id=run.id,
            title=f"Paper {index}", abstract=f"abstract {index}", authors=("A",),
            year=2025, publication_date=None, doi=None, arxiv_id=None, url=None,
            source_names=("test",),
        ))
    return repository, project, run


def _router():
    registry = ProviderRegistry()
    registry.register_llm("fake", FakeLLMProvider(responder=lambda messages: "{}"))
    from autoresearch.providers.capabilities import ModelRole
    from autoresearch.providers.config import ModelAssignment

    assignments = {ModelRole.LITERATURE_EXTRACTION: ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "fake", "fake-model")}
    return ModelRouter(registry, assignments)


def _patch_collaborators(service, repository, gap_analyzer=None, ranker=None, writer=None):
    async def default_gap(**kwargs):
        return SimpleNamespace(id="gap-1", provenance=SimpleNamespace(checksum="c"))

    async def default_rank(**kwargs):
        return [SimpleNamespace(paper_id=p.id) for p in kwargs["papers"]]

    def default_writer(**kwargs):
        return SimpleNamespace(path="/tmp/r.json"), SimpleNamespace(path="/tmp/r.md")

    service.gap_analyzer = SimpleNamespace(analyze=gap_analyzer or default_gap)
    service.ranker = SimpleNamespace(rank=ranker or default_rank)
    service.writer = SimpleNamespace(write=writer or default_writer)
    repository.save_score = lambda score: None
    repository.save_gap = lambda gap: None
    return service


async def _run_review(service, run_id, project_id, top_k=5):
    return await service._execute_review(
        run_id, project_id, top_k=top_k, hierarchical=False,
        max_runtime_seconds=None, fresh=True,
    )


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_partial_paper_failures_do_not_abort_the_run(tmp_path):
    repository, project, run = _seed_environment(tmp_path)
    service = LiteratureReviewService(repository, _router())
    _patch_collaborators(service, repository)

    async def review_one(paper, *args, **kwargs):
        if paper.id in {"paper-2", "paper-5"}:
            raise ValueError("content_failure: unparseable source")
        return _bundle(paper)

    service._review_one = review_one
    outcome = await _run_review(service, run.id, project.id)
    assert outcome.reviewed == 6
    assert outcome.failed == 2
    assert outcome.gap is not None
    failures = repository.rows("SELECT paper_id FROM review_failures WHERE run_id=?", (run.id,))
    assert {row["paper_id"] for row in failures} == {"paper-2", "paper-5"}


@pytest.mark.anyio
async def test_systemic_provider_error_fails_fast(tmp_path):
    repository, project, run = _seed_environment(tmp_path)
    service = LiteratureReviewService(repository, _router())
    _patch_collaborators(service, repository)

    calls = {"count": 0}

    async def review_one(paper, *args, **kwargs):
        calls["count"] += 1
        if paper.id == "paper-0":
            raise ProviderError("credential expired", ErrorKind.AUTHENTICATION, "reasoning")
        await asyncio.sleep(2)  # would take minutes under the old behavior
        return _bundle(paper)

    service._review_one = review_one
    with pytest.raises(RuntimeError) as exc:
        await _run_review(service, run.id, project.id)
    message = str(exc.value)
    assert "review_aborted" in message
    assert "credential expired" in message
    assert calls["count"] < 8, "remaining papers must be cancelled, not processed"


@pytest.mark.anyio
async def test_transient_paper_error_retried_once(tmp_path):
    repository, project, run = _seed_environment(tmp_path)
    service = LiteratureReviewService(repository, _router())
    _patch_collaborators(service, repository)

    attempts = {"paper-1": 0}

    async def review_one(paper, *args, **kwargs):
        if paper.id == "paper-1":
            attempts["paper-1"] += 1
            if attempts["paper-1"] == 1:
                raise ProviderError("throttled", ErrorKind.RATE_LIMIT, "reasoning")
        return _bundle(paper)

    service._review_one = review_one
    outcome = await _run_review(service, run.id, project.id)
    assert outcome.reviewed == 8 and outcome.failed == 0
    assert attempts["paper-1"] == 2


@pytest.mark.anyio
async def test_gap_analysis_retries_transient_failure(tmp_path):
    repository, project, run = _seed_environment(tmp_path)
    service = LiteratureReviewService(repository, _router())

    attempts = {"gap": 0}

    async def flaky_gap(**kwargs):
        attempts["gap"] += 1
        if attempts["gap"] == 1:
            raise ProviderError("gateway timeout", ErrorKind.TIMEOUT, "reasoning")
        return SimpleNamespace(id="gap-1", provenance=SimpleNamespace(checksum="c"))

    _patch_collaborators(service, repository, gap_analyzer=flaky_gap)

    async def review_one(paper, *args, **kwargs):
        return _bundle(paper)

    service._review_one = review_one
    outcome = await _run_review(service, run.id, project.id)
    assert outcome.gap is not None
    assert attempts["gap"] == 2


@pytest.mark.anyio
async def test_stale_running_review_run_is_retired(tmp_path):
    repository, project, _seeding = _seed_environment(tmp_path)
    stale = repository.create_run(project.id, PipelineKind.REVIEW, {})
    service = LiteratureReviewService(repository, _router())
    _patch_collaborators(service, repository)

    async def review_one(paper, *args, **kwargs):
        return _bundle(paper)

    service._review_one = review_one
    await service.review(project.id, top_k=5)
    assert repository.get_run(stale.id).status is RunStatus.CANCELLED
