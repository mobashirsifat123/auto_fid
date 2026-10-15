"""Durable workflow coordination tests: stop stages, resume without repeats,
project isolation, approval gating, and cross-stage provenance.

Deterministic (FakeLLMProvider, fixture sources); no paid API, no network.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from uuid import uuid4

import pytest
from pypdf import PdfWriter

from autoresearch.application.services import (
    DiscoveryService,
    ExperimentService,
    LiteratureReviewService,
    NoveltyService,
    ReinventionService,
)
from autoresearch.application.workflow import STOP_STAGES, WorkflowService
from autoresearch.domain.models import DownloadStatus, content_checksum
from autoresearch.papers.metadata_sources import MetadataPaper, SourceSearchResult
from autoresearch.papers.pdf_resolver import DownloadedPdf, PdfResolution
from autoresearch.papers.search_pipeline import StudySearchPipeline
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.config import ModelAssignment
from autoresearch.providers.registry import ProviderRegistry
from autoresearch.providers.router import ModelRouter
from autoresearch.providers.testing import FakeEmbeddingProvider, FakeLLMProvider
from autoresearch.storage import ResearchRepository
from autoresearch.workers.local_typed import LocalTypedWorker


class DeterministicMetadataSource:
    name = "deterministic_source"

    async def search(self, query: str, *, limit: int, since_year: int | None = None) -> SourceSearchResult:
        papers = tuple(
            MetadataPaper(
                title=f"Adaptive Fusion Study {index}",
                abstract=f"Study {index} evaluates adaptive fusion on Dataset-A. Adaptive fusion improves robustness. Evaluation is limited to one dataset.",
                authors=(f"Researcher {index}",), year=2026 - (index % 4), publication_date=f"{2026 - (index % 4)}-01-01",
                doi=f"10.5555/study.{index}", url=f"https://example.org/study/{index}",
                open_access_pdf_url=f"https://oa.example.org/study-{index}.pdf", is_open_access=True,
                citation_count=index * 2, venue="Test Venue", source=self.name, external_id=str(index),
            )
            for index in range(25)
        )
        return SourceSearchResult(self.name, query, papers[:limit], content_checksum({"query": query, "count": len(papers[:limit])}))

    async def close(self) -> None:
        return None


class DeterministicPdfResolver:
    async def resolve(self, paper: MetadataPaper) -> PdfResolution:
        return PdfResolution(True, paper.open_access_pdf_url, "test_oa_registry", "fixture", "cc-by", None)

    async def download(self, resolution: PdfResolution, target: Path) -> DownloadedPdf:
        payload = b"%PDF-1.4\n% deterministic fixture\n%%EOF"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        return DownloadedPdf(target, content_checksum(payload), len(payload))

    async def close(self) -> None:
        return None


METHODOLOGY_KEYS = (
    "hypothesis_statement", "predicted_observations", "proposed_mechanism",
    "competing_explanations", "falsification_conditions", "closest_prior_methods",
    "algorithm_outline", "inputs", "outputs", "components", "procedure",
    "implementation_requirements", "expected_resource_demands",
    "supporting_evidence_claim_ids", "uncertainty_statement",
)


_WORKSPACE = str(Path.cwd())


def _responder(messages):
    prompt = str(messages[-1]["content"])
    if "seed-analysis-v1" in prompt:
        return {"title": "Seed Adaptive Fusion", "topic": "robust adaptive fusion", "research_question": "Can adaptive fusion improve cross-domain robustness?", "methods": ["adaptive fusion"], "datasets": ["Dataset-A"], "citations": [], "limitations": ["single-domain evaluation"], "keywords": ["robustness"]}
    if "search-strategies-v1" in prompt:
        return {"strategies": [{"name": "topic", "query": "adaptive fusion robustness"}, {"name": "method", "query": "adaptive fusion domain shift"}, {"name": "question", "query": "cross-domain robust fusion"}]}
    if "paper-review-v2" in prompt:
        return {
            "problem": "Cross-domain robustness", "research_question": "Does adaptive fusion improve robustness?", "method": "Adaptive fusion",
            "datasets": ["Dataset-A"], "experimental_protocol": "Single-dataset controlled evaluation", "results": ["Improved robustness"],
            "ablations": ["Fusion removed"], "limitations": ["Evaluation uses one dataset"], "failure_modes": ["Unverified domain transfer"], "confidence": 0.86,
            "evidence_spans": [
                {"claim_type": "result", "statement": "Adaptive fusion improves robustness.", "excerpt": "Adaptive fusion improves robustness.", "page": None, "section": "abstract", "confidence": 0.9},
                {"claim_type": "limitation", "statement": "Evaluation is limited to one dataset.", "excerpt": "Evaluation is limited to one dataset.", "page": None, "section": "abstract", "confidence": 0.95},
            ],
        }
    if "limitation-cluster-v1" in prompt:
        paper_ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
        return {"clusters": [{"theme": "single-domain evaluation", "frequency": 5, "paper_ids": paper_ids[:2], "description": "Most studies evaluate on a single dataset."}]}
    if "evidence-gap-v2" in prompt:
        claim_ids = re.findall(r'"claim_id":\s*"([^"]+)"', prompt)
        paper_ids = list(dict.fromkeys(re.findall(r'"paper_id":\s*"([^"]+)"', prompt)))
        return {"gap_statement": "Adaptive fusion lacks matched cross-domain validation.", "supporting_paper_ids": paper_ids[:2], "supporting_claim_ids": claim_ids[:2], "why_existing_work_does_not_solve_it": "The highest-priority studies evaluate only one dataset.", "missing_experiment": "A parameter-matched evaluation across three held-out domains.", "falsification_test": "Reject if matched multi-domain evidence shows consistent gains.", "confidence": 0.88, "counter_evidence_claim_ids": []}
    if "counter-evidence-v1" in prompt:
        return {"counter_evidence_claim_ids": [], "analysis": "No contradicting claims found."}
    if "gap-selection-v1" in prompt:
        return {"selected_index": 0, "selection_rationale": "Candidate 0 is precise and strongly supported."}
    if "methodology-freeze-v1" in prompt:
        bracketed = re.findall(r"\[([^\]]+)\]", prompt)
        claim_ids = [b for b in bracketed if len(b) == 32 and re.fullmatch(r"[a-f0-9]+", b)]
        paper_ids = [b for b in bracketed if b not in claim_ids]
        return {
            "hypothesis_statement": "Per-domain recalibration improves cross-domain robustness over fixed fusion.",
            "predicted_observations": ["+2 mAP on held-out domains", "no degradation on source domain"],
            "proposed_mechanism": "Domain-conditioned normalization removes covariate shift at the feature level.",
            "competing_explanations": ["more training data alone suffices", "gains are seed variance"],
            "falsification_conditions": ["no significant gain across three domains with three seeds"],
            "closest_prior_methods": ([{"paper_id": paper_ids[0], "method_summary": "fixed adaptive fusion", "proposed_difference": "we add per-domain recalibration"}] if paper_ids else []),
            "algorithm_outline": "1. encode x 2. estimate domain statistics mu_d, sigma_d 3. recalibrate z=(x-mu_d)/sigma_d 4. decode; loss = task loss + consistency term.",
            "inputs": "images, domain labels", "outputs": "detections, calibrated features",
            "components": ["encoder", "domain-statistics estimator", "decoder"],
            "procedure": ["pretrain encoder", "estimate per-domain statistics", "recalibrate", "evaluate on held-out domains"],
            "implementation_requirements": "PyTorch, single GPU",
            "expected_resource_demands": "2 GPU-hours per seed",
            "supporting_evidence_claim_ids": claim_ids[:1],
            "uncertainty_statement": "Novelty is retrieval-assessed only; per-domain statistics may be unstable with few samples per domain.",
        }
    if "Research Gap:" in prompt or "candidate-gen-v3" in prompt:
        return {
            "strategy": "limitation root cause elimination", "method": "New Method", "causal_mechanism": "Cause",
            "target_limitation": "Limitation", "difference_from_existing": "Diff", "expected_effect": "Effect",
            "falsification_condition": "Falsification", "minimum_validating_experiment": "Experiment",
            "required_ablations": [], "implementation_complexity": "Low", "expected_compute": "Low", "risks_and_objections": []
        }
    if "Proposed Method:" in prompt or "prior-art-v2" in prompt:
        return {"overlap_status": "potentially_novel", "mechanism_comparison": "Comparison", "supporting_excerpts": [], "unresolved_overlap": "None"}
    if "Seeds required:" in prompt:
        return {"selected_index": 0, "selection_rationale": "Design 0 is maximally falsifiable."}
    if "what_would_disprove" in prompt:
        return {"what_would_disprove": "No consistent improvement.", "confounding_factors": ["Seed variance"], "minimal_viable_experiment": "One dataset, three seeds.", "expected_null_outcome": "No gain.", "failure_modes": ["Implementation error"]}
    if "experiment-planner" in prompt:
        return {
            "hypothesis": "H1", "causal_mechanism": "C1", "falsification_condition": "F1", "repository": _WORKSPACE,
            "baseline_commit": "main", "dataset_identity": "data", "dataset_checksum": "checksum", "split_definition": "split",
            "baseline_configuration": "base", "proposed_change": "change", "independent_variables": [], "controlled_variables": [],
            "metrics": ["mAP"], "random_seeds": [42], "ablations": [], "robustness_tests": [], "statistical_plan": "paired t-test across seeds",
            "compute_budget": "budget", "stopping_conditions": [], "expected_outputs": ["scripts/run_experiment.py"], "artifact_contract": "contract",
            "interpretation_rules": [
                "positive: significant gain on all held-out domains",
                "negative: significant degradation",
                "null: no significant difference",
                "failed: execution or artifact-validation failure",
                "unavailable: metric artifact missing",
            ],
        }
    if "Method:" in prompt or "novelty-critic" in prompt:
        return {
            "novelty_score": 0.9, "significance_score": 0.9, "causal_plausibility_score": 0.9, "falsifiability_score": 0.9,
            "experimental_discriminability_score": 0.9, "feasibility_score": 0.9, "implementation_risk_score": 0.1,
            "expected_information_gain_score": 0.9, "compute_efficiency_score": 0.9, "publication_potential_score": 0.9,
            "objections": [], "failure_scenarios": []
        }
    if "reinvention-analysis-v1" in prompt:
        return {
            "action": "pivot",
            "root_cause_diagnosis": "The mechanism did not transfer across domains; the normalization statistics were estimated on the source domain only.",
            "empirical_lessons": ["source-only statistics do not generalize", "seed variance was controlled"],
            "next_generation_hypothesis": "Target-domain statistics estimated with a small calibration set remove the shift.",
            "revised_mechanism": "calibration-set domain normalization",
            "revised_method_name": "Recalibrated Fusion v2",
            "revised_difference_statement": "Uses target-domain calibration statistics instead of source-only.",
            "expected_effect": "+1.5 mAP on held-out domains",
            "actionable_next_steps": ["collect calibration samples", "re-run parameter-matched evaluation"],
        }
    raise AssertionError(f"Unexpected provider prompt: {prompt[:120]}")


def _router(repository, workspace: str):
    global _WORKSPACE
    _WORKSPACE = workspace
    registry = ProviderRegistry()
    registry.register_llm("fake", FakeLLMProvider(responder=_responder))
    registry.register_embeddings("fake_embeddings", FakeEmbeddingProvider())
    assignments = {
        ModelRole.LITERATURE_EXTRACTION: ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "fake", "fake-model"),
        ModelRole.RESEARCH_REASONING: ModelAssignment(ModelRole.RESEARCH_REASONING, "fake", "fake-model"),
        ModelRole.SCIENTIFIC_CRITIC: ModelAssignment(ModelRole.SCIENTIFIC_CRITIC, "fake", "fake-model"),
        ModelRole.EMBEDDINGS: ModelAssignment(ModelRole.EMBEDDINGS, "fake_embeddings", "fake-embedding-v1"),
    }
    return ModelRouter(registry, assignments, usage_sink=repository.record_provider_usage)


def _harness(tmp_path: Path):
    repository = ResearchRepository(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    repository.migrate()
    workspace = tmp_path / "workspace"
    (workspace / "scripts").mkdir(parents=True, exist_ok=True)
    (workspace / "scripts" / "run_experiment.py").write_text("print('deterministic fixture experiment')\n")
    worker = LocalTypedWorker(repository, workspace)
    router = _router(repository, str(workspace))
    pipeline = StudySearchPipeline(repository, [DeterministicMetadataSource()], DeterministicPdfResolver(), tmp_path / "downloads")
    services = {
        "discovery": DiscoveryService(repository, router, pipeline),
        "review": LiteratureReviewService(repository, router),
        "novelty": NoveltyService(repository, router, pipeline),
        "experiment": ExperimentService(repository, router, workspace, worker=worker),
        "reinvention": ReinventionService(repository, router),
    }
    workflow = WorkflowService(repository, router, **services)
    seed = tmp_path / "seed.pdf"
    writer = PdfWriter(); writer.add_blank_page(width=200, height=200); writer.add_metadata({"/Title": "Seed Adaptive Fusion"})
    with seed.open("wb") as handle:
        writer.write(handle)
    return repository, workflow, worker, seed, workspace


@pytest.mark.anyio
async def test_workflow_runs_to_design_and_links_provenance(tmp_path):
    repository, workflow, _, seed, _ = _harness(tmp_path)
    result = await workflow.run(seed=str(seed), until="design")

    assert result["stopped_at"] == "design"
    statuses = {s["stage"]: s["status"] for s in result["stages"]}
    assert statuses["discovery"] == "completed"
    assert statuses["review"] == "completed"
    assert statuses["question"] == "completed"
    assert statuses["hypotheses"] == "completed"
    assert statuses["methodology"] == "completed"
    assert statuses["design"] == "completed"

    # Cross-stage provenance chain: spec → freeze → candidate → gap → claims.
    design = next(s for s in result["stages"] if s["stage"] == "design")
    spec = repository.get_experiment_spec_by_id(design["spec_id"])
    assert spec.gap_id == next(s for s in result["stages"] if s["stage"] == "review")["gap_id"]
    assert spec.methodology_freeze_id == next(s for s in result["stages"] if s["stage"] == "methodology")["methodology_freeze_id"]
    assert spec.interpretation_rules, "pre-registered interpretation rules missing"

    freeze = repository.methodology_freeze(spec.methodology_freeze_id)
    assert freeze is not None and freeze.version == 1
    record = json.loads(Path(freeze.artifact_path).read_text())
    for key in METHODOLOGY_KEYS:
        assert key in record["methodology"], f"methodology missing {key}"
    assert len(record["methodology"]["uncertainty_statement"]) >= 20
    gap = repository.row("SELECT * FROM research_gaps WHERE id=?", (spec.gap_id,))
    assert gap and json.loads(gap["supporting_claim_ids_json"]), "question must trace to verified claims"
    # The frozen methodology's content must reach the design prompt, and the
    # design must carry executable outputs plus real interpretation rules.
    assert spec.expected_outputs and spec.expected_outputs[0].endswith(".py")
    assert any("positive" in rule for rule in spec.interpretation_rules), spec.interpretation_rules


@pytest.mark.anyio
async def test_workflow_resume_reuses_completed_stages(tmp_path):
    repository, workflow, _, seed, _ = _harness(tmp_path)
    first = await workflow.run(seed=str(seed), until="review")

    def run_kinds():
        return [r["kind"] for r in repository.rows("SELECT kind FROM pipeline_runs")]

    after_first = run_kinds()
    second = await workflow.run(project_id=first["project_id"], until="design")
    statuses = {s["stage"]: s["status"] for s in second["stages"]}
    assert statuses["review"] == "reused"
    assert statuses["question"] == "reused"
    assert statuses["hypotheses"] == "completed"
    assert statuses["methodology"] == "completed"
    assert statuses["design"] == "completed"

    third = await workflow.run(project_id=first["project_id"], until="design")
    reused = {s["stage"]: s["status"] for s in third["stages"]}
    assert reused["review"] == "reused" and reused["hypotheses"] == "reused"
    assert reused["methodology"] == "reused" and reused["design"] == "reused"
    # No repeated discovery/review/novelty runs after the first pass.
    assert run_kinds().count("study_search") == after_first.count("study_search")
    assert run_kinds().count("deep_literature_review") == after_first.count("deep_literature_review")


@pytest.mark.anyio
async def test_workflow_project_isolation(tmp_path):
    repository, workflow, _, seed, _ = _harness(tmp_path)
    first = await workflow.run(seed=str(seed), until="design")
    second = await workflow.run(seed=str(seed), until="design")
    assert first["project_id"] != second["project_id"]

    gap_a = next(s for s in first["stages"] if s["stage"] == "review")["gap_id"]
    gap_b = next(s for s in second["stages"] if s["stage"] == "review")["gap_id"]
    assert gap_a != gap_b
    spec_a = repository.get_experiment_spec(next(s for s in first["stages"] if s["stage"] == "methodology")["candidate_id"])
    assert spec_a.gap_id == gap_a, "project A design consumed project B's gap"
    # Project-scoped reuse: extending A never picks up B's newer outputs.
    extended = await workflow.run(project_id=first["project_id"], until="design")
    assert next(s for s in extended["stages"] if s["stage"] == "review")["gap_id"] == gap_a


@pytest.mark.anyio
async def test_workflow_execution_requires_exact_approval(tmp_path):
    repository, workflow, _, seed, _ = _harness(tmp_path)
    done = await workflow.run(seed=str(seed), until="design")
    project = done["project_id"]

    blocked = await workflow.run(project_id=project, until="experiment")
    experiment_stage = next(s for s in blocked["stages"] if s["stage"] == "experiment")
    assert experiment_stage["status"] == "awaiting_approval"
    assert experiment_stage["checksum"] and experiment_stage["actions"]
    assert "missing_requirement" in experiment_stage
    assert repository.rows("SELECT * FROM worker_jobs") == [], "unapproved execution must not create jobs"

    # Approve the exact checksum, then execution proceeds through the worker.
    candidate_id = next(s for s in blocked["stages"] if s["stage"] == "methodology")["candidate_id"]
    from autoresearch.application.services import ExperimentService
    preview = workflow.experiment.approval_preview_for_candidate(candidate_id)
    saved = repository.save_exact_approval({
        "id": uuid4().hex, "run_id": preview["run_id"], "specification_checksum": str(preview["checksum"]),
        "approver": "test-user", "approved_scope": "workspace", "approved_target": str(preview["worker"]),
        "approved_budget": 60.0, "expires_at": "2099-01-01T00:00:00+00:00", "status": "pending",
    })
    repository.approve_exact(saved["id"], approver="test-user")

    approved = await workflow.run(project_id=project, until="experiment")
    stage = next(s for s in approved["stages"] if s["stage"] == "experiment")
    assert stage["status"] in ("completed", "executed_analysis_pending")
    jobs = repository.rows("SELECT * FROM worker_jobs")
    assert jobs and jobs[0]["state"] == "COMPLETED", "approved job did not complete"
    # Duplicate dispatch: a second identical execution reuses the same job.
    job_ids = {job["id"] for job in repository.rows("SELECT id FROM worker_jobs")}
    again = await workflow.run(project_id=project, until="experiment")
    assert {job["id"] for job in repository.rows("SELECT id FROM worker_jobs")} == job_ids
    assert again["stopped_at"] == "experiment"


def test_stop_stage_vocabulary():
    assert STOP_STAGES == ("review", "question", "hypotheses", "methodology", "design", "experiment")


PROVENANCE_FIXTURE = '{"project_id":"t","run_id":"t","source_evidence":[],"provider":"t","model":"t","prompt_version":"v1","created_at":"2026-01-01T00:00:00Z","checksum":"s"}'


def _plant_analyzed_report(repository, workflow, candidate_id: str, decision: str = "refuted") -> str:
    """Plant a completed statistical analysis + report for an executed spec,
    standing in for the not-yet-wired artifact-ingestion step."""
    from uuid import uuid4 as _uuid4
    spec = repository.get_experiment_spec(candidate_id)
    assert spec is not None
    analysis_id = _uuid4().hex
    repository.execute(
        "INSERT INTO statistical_analyses(id,project_id,run_id,spec_id,baseline_metrics_json,proposed_metrics_json,"
        "effect_size_json,confidence_intervals_json,seed_level_results_json,ablation_contributions_json,"
        "parameter_count_diff,flops_latency_diff,training_inference_cost,failure_case_analysis,"
        "hypothesis_support_assessment,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (analysis_id, spec.project_id, spec.run_id, spec.id, "{}", "{}", "{}", "{}", "[]", "{}",
         0, 0.0, 0.0, "none", "not supported", PROVENANCE_FIXTURE),
    )
    report_id = _uuid4().hex
    repository.execute(
        "INSERT INTO experiment_reports(id,project_id,run_id,analysis_id,literature_motivation,research_gap,"
        "hypothesis,proposed_method,implementation_diff,lineage,dataset_split,exact_commands_json,"
        "baseline_results_json,aggregated_results_json,uncertainty,ablations,robustness_tests,failed_runs,"
        "cost,limitations,reproducibility_instructions,decision,decision_evidence,provenance_json)"
        " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (report_id, spec.project_id, spec.run_id, analysis_id, "m", "g", spec.hypothesis, "p", "d", "l", "s",
         "[]", "{}", "{}", "u", "a", "r", 0, 0.0, "lim", "repro", decision, "evidence", PROVENANCE_FIXTURE),
    )
    return report_id


async def _approved_run_to_experiment(repository, workflow, seed) -> str:
    """Drive a fresh project through design, approve it, and execute."""
    from uuid import uuid4 as _uuid4
    done = await workflow.run(seed=str(seed), until="design")
    project = done["project_id"]
    candidate_id = next(s for s in done["stages"] if s["stage"] == "methodology")["candidate_id"]
    preview = workflow.experiment.approval_preview_for_candidate(candidate_id)
    saved = repository.save_exact_approval({
        "id": _uuid4().hex, "run_id": preview["run_id"], "specification_checksum": str(preview["checksum"]),
        "approver": "test-user", "approved_scope": "workspace", "approved_target": str(preview["worker"]),
        "approved_budget": 60.0, "expires_at": "2099-01-01T00:00:00+00:00", "status": "pending",
    })
    repository.approve_exact(saved["id"], approver="test-user")
    result = await workflow.run(project_id=project, until="experiment")
    return next(s for s in result["stages"] if s["stage"] == "methodology")["candidate_id"], project, result


@pytest.mark.anyio
async def test_every_stop_stage_stops_before_later_stages(tmp_path):
    """Stopping at a stage must not execute later stages (regression: until=
    'question' used to run the whole novelty portfolio)."""
    kinds_after = {
        "review": set(),
        "question": set(),
        "hypotheses": {"novelty_generation"},
        "methodology": {"novelty_generation"},
        "design": {"novelty_generation", "experiment_design_execution"},
    }
    for stage in ("review", "question", "hypotheses", "methodology", "design"):
        repository, workflow, _, seed, _ = _harness(tmp_path / stage)
        result = await workflow.run(seed=str(seed), until=stage)
        assert result["stopped_at"] == stage
        ran = {row["kind"] for row in repository.rows("SELECT kind FROM pipeline_runs")}
        assert ran == {"study_search", "deep_literature_review"} | kinds_after[stage], (
            f"until={stage} executed unexpected stages: {ran}"
        )


@pytest.mark.anyio
async def test_revision_loop_runs_when_an_analyzed_report_exists(tmp_path):
    """Regression: the loop was unreachable and silently skipped when no
    report existed. With a planted analysis+report (decision=refuted) the loop
    must execute: reinvention → new freeze → new spec → approval gate."""
    repository, workflow, _, seed, _ = _harness(tmp_path)
    candidate_id, project, _ = await _approved_run_to_experiment(repository, workflow, seed)
    _plant_analyzed_report(repository, workflow, candidate_id, decision="refuted")

    result = await workflow.run(project_id=project, until="experiment", max_revisions=1)
    stages = result["stages"]
    experiment = next(s for s in stages if s["stage"] == "experiment")
    assert experiment["status"] == "completed" and experiment["decision"] == "refuted"
    revision = [s for s in stages if s["stage"] == "revision"]
    assert revision, "revision loop produced no disposition"
    assert revision[0]["status"] == "awaiting_approval"
    assert revision[0]["methodology_freeze_id"] and revision[0]["candidate_id"] != candidate_id
    # Prior attempt retained: original freeze/candidate/spec untouched.
    assert repository.get_novelty_candidate(candidate_id) is not None
    assert repository.latest_methodology_freeze_for_candidate(candidate_id) is not None


@pytest.mark.anyio
async def test_revision_reports_unavailable_without_an_analyzed_report(tmp_path):
    """Without an analysis, revision must be dispositioned unavailable — never
    silently skipped and never fabricated."""
    repository, workflow, _, seed, _ = _harness(tmp_path)
    _, project, _ = await _approved_run_to_experiment(repository, workflow, seed)
    result = await workflow.run(project_id=project, until="experiment", max_revisions=1)
    revision = [s for s in result["stages"] if s["stage"] == "revision"]
    assert revision and revision[0]["status"] == "unavailable"
    assert "ingestion" in revision[0]["reason"]


@pytest.mark.anyio
async def test_worker_unreported_failure_dispositioned(tmp_path):
    """worker_unavailable must disposition, not crash the workflow."""
    repository, workflow, _, seed, _ = _harness(tmp_path)
    done = await workflow.run(seed=str(seed), until="design")
    project = done["project_id"]
    candidate_id = next(s for s in done["stages"] if s["stage"] == "methodology")["candidate_id"]
    # Detach the worker to simulate an unconfigured executor.
    workflow.experiment.worker = None
    preview = workflow.experiment.approval_preview_for_candidate(candidate_id)
    from uuid import uuid4 as _uuid4
    saved = repository.save_exact_approval({
        "id": _uuid4().hex, "run_id": preview["run_id"], "specification_checksum": str(preview["checksum"]),
        "approver": "test-user", "approved_scope": "workspace", "approved_target": str(preview["worker"]),
        "approved_budget": 60.0, "expires_at": "2099-01-01T00:00:00+00:00", "status": "pending",
    })
    repository.approve_exact(saved["id"], approver="test-user")
    result = await workflow.run(project_id=project, until="experiment")
    experiment = next(s for s in result["stages"] if s["stage"] == "experiment")
    assert experiment["status"] == "worker_unavailable"
