"""Regression tests for literature-grounded candidate generation and
experiment planning (HKUDS AI-Researcher mechanism adaptation).

The grounding tests fail against the pre-change code: the candidate-generation
and planning prompts contained none of the project's verified evidence (see
docs/architecture.md, "Novelty and Experiment Pipelines"). The
backward-compatibility and import tests are defect guards rather than
grounding regressions.
"""

import json
from pathlib import Path

import pytest

from autoresearch.domain.models import (
    ArtifactProvenance,
    CandidateStatus,
    GapClassification,
    NoveltyCandidate,
    PipelineKind,
    PriorArtStatus,
    ResearchGap,
    RunStatus,
)
from autoresearch.experiments.pipeline import ExperimentPipeline
from autoresearch.experiments.planner import ExperimentSpecGenerator
from autoresearch.novelty.candidate_generator import LLMCandidateGenerator
from autoresearch.novelty.pipeline import NoveltyPipeline
from autoresearch.providers.config import ModelAssignment
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.registry import ProviderRegistry
from autoresearch.providers.router import ModelRouter
from autoresearch.providers.testing import FakeLLMProvider
from autoresearch.review.evidence_context import build_evidence_context, format_evidence_context
from autoresearch.storage.repository import ResearchRepository

PROVENANCE = '{"project_id":"t","run_id":"t","source_evidence":[],"provider":"t","model":"t","prompt_version":"v1","created_at":"2026-01-01T00:00:00Z","checksum":"s"}'
CLAIM_TEXT_1 = "Illumination-invariant features reduce night-time detection errors by 12 percent."
CLAIM_TEXT_2 = "Baseline YOLO models lose 30 percent recall on night imagery without adaptation."


def _router(responder):
    provider = FakeLLMProvider(responder=responder)
    registry = ProviderRegistry()
    registry.register_llm("fake", provider)
    assignments = {
        ModelRole.RESEARCH_REASONING: ModelAssignment(ModelRole.RESEARCH_REASONING, "fake", "fake-model"),
        ModelRole.SCIENTIFIC_CRITIC: ModelAssignment(ModelRole.SCIENTIFIC_CRITIC, "fake", "fake-model"),
    }
    return ModelRouter(registry, assignments)


def _seeded_repository(tmp_path: Path) -> ResearchRepository:
    repo = ResearchRepository(tmp_path / "state.db", tmp_path / "artifacts")
    repo.migrate()
    project = repo.create_project("Grounding Test")
    repo.execute(
        "INSERT INTO pipeline_runs(id,project_id,kind,status,input_json,started_at) VALUES(?,?,?,?,?,?)",
        ("run1", project.id, PipelineKind.NOVELTY.value, "completed", "{}", "2026-01-01"),
    )
    repo.execute(
        "INSERT INTO research_gaps(id,project_id,run_id,statement,supporting_paper_ids_json,"
        "supporting_claim_ids_json,why_unsolved,missing_experiment,falsification_test,confidence,"
        "counter_evidence_claim_ids_json,classification,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        ("gap1", project.id, "run1", "Camera-trap detectors degrade under night imagery.",
         '["paper1","paper2"]', '["claim1","claim2"]', "Domain shift.", "Night split eval.",
         "Fails if night mAP >= day mAP.", 0.9, "[]", GapClassification.EVIDENCE_SUPPORTED.value, PROVENANCE),
    )
    for pid, cid in (("paper1", "claim1"), ("paper2", "claim2")):
        repo.execute(
            "INSERT INTO papers(id,project_id,run_id,title,abstract,authors_json,year,source_names_json,"
            "is_open_access,citation_count,venue,checksum,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (pid, project.id, "run1", f"Paper {pid}", "", "[]", 2025, '["test"]', 0, 0, "", "x", "2026-01-01"),
        )
        text = CLAIM_TEXT_1 if cid == "claim1" else CLAIM_TEXT_2
        repo.execute(
            "INSERT INTO evidence_claims(id,project_id,run_id,paper_id,claim_type,statement,excerpt,page,section,"
            "confidence,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (cid, project.id, "run1", pid, "result", text, text, 4, "results", 0.9, PROVENANCE),
        )
        repo.execute(
            "INSERT INTO paper_reviews(id,project_id,run_id,paper_id,problem,research_question,method,"
            "datasets_json,experimental_protocol,results_json,ablations_json,limitations_json,failure_modes_json,"
            "evidence_claim_ids_json,confidence,provenance_json) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (f"rev{pid}", project.id, "run1", pid, "p", "q", f"Method-of-{pid}", "[]", "proto", "[]", "[]",
             "[]", "[]", f'["{cid}"]', 0.8, PROVENANCE),
        )
    return repo


def _gap(repo: ResearchRepository) -> ResearchGap:
    row = repo.row("SELECT * FROM research_gaps WHERE id='gap1'")
    return ResearchGap(
        row["id"], row["project_id"], row["run_id"], row["statement"],
        tuple(json.loads(row["supporting_paper_ids_json"])),
        tuple(json.loads(row["supporting_claim_ids_json"])),
        row["why_unsolved"], row["missing_experiment"], row["falsification_test"],
        float(row["confidence"]), tuple(json.loads(row["counter_evidence_claim_ids_json"])),
        GapClassification(row["classification"]), ArtifactProvenance(**json.loads(row["provenance_json"])),
    )


@pytest.mark.anyio
async def test_novelty_generation_prompt_is_literature_grounded(tmp_path):
    """The pipeline (not just the generator) must pass verified claims and
    paper methods into the candidate-generation prompt."""
    repo = _seeded_repository(tmp_path)
    captured: list[str] = []

    def responder(messages):
        prompt = str(messages[-1]["content"])
        if "Proposed Method:" in prompt or "Prior Art Abstracts" in prompt:
            return {
                "overlap_status": "potentially_novel", "mechanism_comparison": "distinct",
                "supporting_excerpts": [], "unresolved_overlap": "none",
            }
        if "Research Gap:" in prompt:
            captured.append(prompt)
            return {
                "strategy": "limitation root cause elimination",
                "method": "Night-adaptive normalization head",
                "causal_mechanism": "illumination-invariant features",
                "target_limitation": "night recall loss",
                "difference_from_existing": "Unlike [paper1] which uses fixed normalization, we adapt per-image.",
                "expected_effect": "+5 night recall",
                "falsification_condition": "night mAP unchanged",
                "minimum_validating_experiment": "night split eval",
                "required_ablations": ["norm"],
                "implementation_complexity": "medium",
                "expected_compute": "2 GPU hours",
                "risks_and_objections": ["overfitting"],
            }
        if "Method:" in prompt:
            return {
                "novelty_score": 0.8, "significance_score": 0.8, "causal_plausibility_score": 0.8,
                "falsifiability_score": 0.8, "experimental_discriminability_score": 0.8,
                "feasibility_score": 0.8, "implementation_risk_score": 0.2,
                "expected_information_gain_score": 0.8, "compute_efficiency_score": 0.9,
                "publication_potential_score": 0.9, "objections": [], "failure_scenarios": [],
            }
        raise AssertionError(f"Unexpected prompt: {prompt[:100]}")

    router = _router(responder)
    from autoresearch.novelty.novelty_critic import ScientificCritic
    from autoresearch.novelty.prior_art_checker import EvidenceBackedPriorArtChecker
    from autoresearch.novelty.selection import PortfolioSelector

    pipeline = NoveltyPipeline(
        repository=repo,
        generator=LLMCandidateGenerator(router),
        prior_art_checker=EvidenceBackedPriorArtChecker(router, None),
        critic=ScientificCritic(router),
        selector=PortfolioSelector(repo),
    )
    portfolio = await pipeline.generate_portfolio(_gap(repo))

    assert portfolio.id
    generation_prompts = [p for p in captured if CLAIM_TEXT_1 in p or CLAIM_TEXT_2 in p]
    assert generation_prompts, "candidate generation prompt contains no verified claim excerpts"
    assert any("Method-of-paper1" in p or "Method-of-paper2" in p for p in generation_prompts), \
        "candidate generation prompt contains no reviewed paper methods"

    # The evidence linkage is persisted as a checksummed artifact, not just prompted.
    artifacts = repo.artifacts_for_run("run1")
    grounding = [a for a in artifacts if a.artifact_type == "novelty_grounding_json"]
    assert grounding, "novelty grounding artifact was not saved"
    record = json.loads(Path(grounding[0].path).read_text())
    assert record["supplied"]["papers"] == ["paper1", "paper2"]
    assert record["supplied"]["claims"] == ["claim1", "claim2"]
    linked = [c["evidence_linked_paper_ids"] for c in record["candidates"]]
    assert any("paper1" in ids for ids in linked), f"no candidate cited supplied papers: {linked}"
    assert record["prompt_version"] == "candidate-gen-v3"


@pytest.mark.anyio
async def test_generator_backward_compatible_without_context(tmp_path):
    repo = _seeded_repository(tmp_path)

    def responder(messages):
        return {
            "strategy": "s", "method": "m", "causal_mechanism": "c", "target_limitation": "t",
            "difference_from_existing": "d", "expected_effect": "e",
            "falsification_condition": "f", "minimum_validating_experiment": "x",
            "required_ablations": ["a"], "implementation_complexity": "low",
            "expected_compute": "1h", "risks_and_objections": ["r"],
        }

    router = _router(responder)
    generator = LLMCandidateGenerator(router)
    candidates = await generator.generate(_gap(repo), count=1)
    assert len(candidates) == 1
    assert candidates[0].method == "m"


def test_cited_paper_ids_only_counts_supplied_papers():
    from autoresearch.novelty.candidate_generator import cited_paper_ids
    ctx = {"papers": [{"paper_id": "paper1"}, {"paper_id": "paper2"}]}
    text = "Unlike [paper1] and [bogus], extends [paper2]."
    assert cited_paper_ids(text, ctx) == ["paper1", "paper2"]
    assert cited_paper_ids("no citations", ctx) == []
    assert cited_paper_ids("[paper1]", None) == []


@pytest.mark.anyio
async def test_experiment_planning_prompt_is_literature_grounded(tmp_path):
    """plan_for_candidate must ground designs in the gap's falsification test,
    paper methods, and the stored prior-art assessment."""
    repo = _seeded_repository(tmp_path)
    gap = _gap(repo)
    candidate = NoveltyCandidate(
        id="cand1", project_id=gap.project_id, run_id=gap.run_id, gap_id=gap.id,
        strategy="s", method="Night-adaptive KAN detection head",
        causal_mechanism="illumination-invariant features", target_limitation="night recall loss",
        difference_from_existing="adaptive normalization", expected_effect="MODEL_HYPOTHESIS: +5 recall",
        falsification_condition="night mAP unchanged", minimum_validating_experiment="night split eval",
        required_ablations=("norm",), implementation_complexity="medium",
        expected_compute="2 GPU hours", risks_and_objections=("overfitting",),
        status=CandidateStatus.GENERATED,
        provenance=ArtifactProvenance.create(
            project_id=gap.project_id, run_id=gap.run_id, source_evidence=[gap.id],
            provider="fake", model="fake-model", prompt_version="t", content={},
        ),
    )
    repo.save_novelty_candidate(candidate)
    repo.execute(
        "INSERT INTO prior_art_assessments(id,project_id,run_id,candidate_id,closest_paper_ids_json,"
        "overlap_status,mechanism_comparison,supporting_excerpts_json,unresolved_overlap,provenance_json)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        ("pa1", gap.project_id, gap.run_id, "cand1", '["paper1"]', "partial_overlap",
         "Shares normalization pipeline", "[]", "uncertain", PROVENANCE),
    )

    design_prompts: list[str] = []

    def responder(messages):
        prompt = str(messages[-1]["content"])
        if "what_would_disprove" in prompt:
            return {
                "what_would_disprove": "w", "confounding_factors": ["c"],
                "minimal_viable_experiment": "m", "expected_null_outcome": "n",
                "failure_modes": ["f"],
            }
        if "Seeds required:" in prompt:
            return {"selected_index": 0, "selection_rationale": "most falsifiable"}
        if "Candidate Method:" in prompt:
            design_prompts.append(prompt)
            return {
                "hypothesis": "h", "causal_mechanism": "c", "falsification_condition": "f",
                "repository": str(tmp_path), "baseline_commit": "deadbeef",
                "dataset_identity": "d", "dataset_checksum": "x", "split_definition": "s",
                "baseline_configuration": "Method-of-paper1 baseline", "proposed_change": "p",
                "independent_variables": ["v"], "controlled_variables": ["c"],
                "metrics": ["m"], "random_seeds": [1], "ablations": ["a"],
                "robustness_tests": ["r"], "statistical_plan": "t-test",
                "compute_budget": "1h", "stopping_conditions": ["s"],
                "expected_outputs": ["scripts/run_experiment.py"], "artifact_contract": "json",
            }
        raise AssertionError(f"Unexpected prompt: {prompt[:100]}")

    router = _router(responder)
    pipeline = ExperimentPipeline(
        repository=repo, planner=ExperimentSpecGenerator(router),
        environment_builder=None, executor=None, analyzer=None, report_writer=None,
    )
    spec = await pipeline.plan_for_candidate(candidate)

    assert spec.hypothesis == "h"
    assert design_prompts, "no design prompts captured"
    for prompt in design_prompts:
        assert "Method-of-paper1" in prompt or "Method-of-paper2" in prompt, \
            "design prompt missing reviewed paper methods"
        assert "Fails if night mAP >= day mAP." in prompt, \
            "design prompt missing the gap's falsification test"
        assert CLAIM_TEXT_1 in prompt or CLAIM_TEXT_2 in prompt, \
            "design prompt missing verified claim excerpts"
        assert "experiment-planner-v4" in prompt
        assert "Shares normalization pipeline" in prompt, \
            "design prompt missing the closest-prior-method comparison"


def test_sparse_corpus_degrades_to_empty_context(tmp_path):
    repo = _seeded_repository(tmp_path)
    context = build_evidence_context(
        repo,
        project_id="missing-project",
        supporting_paper_ids=["nope"],
        supporting_claim_ids=["nada"],
    )
    assert context["papers"] == []
    assert context["claims"] == []
    assert format_evidence_context(context) == ""
    # and formatting None is safe
    assert format_evidence_context(None) == ""


@pytest.mark.anyio
async def test_import_candidate_roundtrip(tmp_path):
    """Regression: import_candidate referenced dataclass fields and enum values
    that do not exist (description/mechanism/prior_art_status/CandidateStatus.PROPOSED)
    and raised TypeError before the fix; run_id="imported" also violated the
    pipeline_runs foreign key."""
    repo = _seeded_repository(tmp_path)
    router = _router(lambda messages: {})
    from autoresearch.application.services import ExperimentService

    service = ExperimentService(repository=repo, router=router, workspace_root=tmp_path)
    candidate = await service.import_candidate({
        "project_id": repo.row("SELECT id FROM projects LIMIT 1")["id"],
        "gap_id": "gap1",
        "method": "Imported method",
        "causal_mechanism": "mechanism text",
        "falsification_condition": "fails",
        "required_ablations": ["a1"],
    })
    assert candidate.method == "Imported method"
    assert candidate.status == CandidateStatus.GENERATED
    assert repo.row("SELECT * FROM novelty_candidates WHERE id=?", (candidate.id,)) is not None
    # The import run must not linger as a phantom in-flight run.
    assert repo.list_runs(status=RunStatus.RUNNING) == []


@pytest.mark.anyio
async def test_import_gap_and_candidate_leave_no_zombie_runs(tmp_path):
    """Integration: the agent-recommended import path (import_gap when no gap
    exists) previously crashed on the research_gaps.run_id foreign key, and
    imports must not leak permanently-RUNNING runs."""
    repo = _seeded_repository(tmp_path)
    router = _router(lambda messages: {})
    from autoresearch.application.services import ExperimentService, NoveltyService

    project_id = repo.row("SELECT id FROM projects LIMIT 1")["id"]

    class _SearchPipeline:  # NoveltyService only stores it for the prior-art checker
        repository = repo
        sources = []

    novelty = NoveltyService(repository=repo, router=router, search_pipeline=_SearchPipeline())
    gap = await novelty.import_gap(project_id, {
        "statement": "Imported gap statement.",
        "why_unsolved": "No one tried.",
        "falsification_test": "Fails when resolved.",
        "supporting_paper_ids": ["paper1"],
        "supporting_claim_ids": ["claim1"],
    })
    assert gap.statement == "Imported gap statement."
    assert repo.get_run(gap.run_id) is not None

    experiment = ExperimentService(repository=repo, router=router, workspace_root=tmp_path)
    candidate = await experiment.import_candidate({
        "project_id": project_id, "gap_id": gap.id, "method": "Imported method",
    })
    assert repo.get_run(candidate.run_id) is not None
    assert repo.list_runs(status=RunStatus.RUNNING) == [], "import leaked a RUNNING run"


@pytest.mark.anyio
async def test_revision_path_keeps_grounding_linkage(tmp_path):
    """A partial-overlap candidate gets revised; the grounding artifact must
    cover the revised candidate and record its literature citation."""
    repo = _seeded_repository(tmp_path)
    prior_art_calls = {"n": 0}

    def responder(messages):
        prompt = str(messages[-1]["content"])
        if "Proposed Method:" in prompt or "Prior Art Abstracts" in prompt:
            prior_art_calls["n"] += 1
            status = "partial_overlap" if prior_art_calls["n"] == 1 else "potentially_novel"
            return {
                "overlap_status": status, "mechanism_comparison": "overlaps normalization",
                "supporting_excerpts": [], "unresolved_overlap": "uncertain",
            }
        if "Research Gap:" in prompt:
            return {
                "strategy": "limitation root cause elimination",
                "method": "Night-adaptive normalization head",
                "causal_mechanism": "illumination-invariant features",
                "target_limitation": "night recall loss",
                "difference_from_existing": "Unlike [paper1] fixed normalization, ours adapts.",
                "expected_effect": "+5 night recall", "falsification_condition": "night mAP unchanged",
                "minimum_validating_experiment": "night split eval", "required_ablations": ["norm"],
                "implementation_complexity": "medium", "expected_compute": "2 GPU hours",
                "risks_and_objections": ["overfitting"],
            }
        if "Revision round:" in prompt:
            return {
                "strategy": "limitation root cause elimination",
                "method": "Revised night-adaptive head",
                "causal_mechanism": "illumination-invariant features, bounded",
                "target_limitation": "night recall loss",
                "difference_from_existing": "Narrowed vs [paper2]: per-image, not dataset-level.",
                "expected_effect": "+3 night recall", "falsification_condition": "night mAP unchanged",
                "minimum_validating_experiment": "night split eval", "required_ablations": ["norm"],
                "implementation_complexity": "medium", "expected_compute": "2 GPU hours",
                "risks_and_objections": ["overfitting"],
            }
        if "Method:" in prompt:
            return {
                "novelty_score": 0.8, "significance_score": 0.8, "causal_plausibility_score": 0.8,
                "falsifiability_score": 0.8, "experimental_discriminability_score": 0.8,
                "feasibility_score": 0.8, "implementation_risk_score": 0.2,
                "expected_information_gain_score": 0.8, "compute_efficiency_score": 0.9,
                "publication_potential_score": 0.9, "objections": [], "failure_scenarios": [],
            }
        raise AssertionError(f"Unexpected prompt: {prompt[:100]}")

    router = _router(responder)
    from autoresearch.novelty.novelty_critic import ScientificCritic
    from autoresearch.novelty.prior_art_checker import EvidenceBackedPriorArtChecker
    from autoresearch.novelty.selection import PortfolioSelector

    pipeline = NoveltyPipeline(
        repository=repo,
        generator=LLMCandidateGenerator(router),
        prior_art_checker=EvidenceBackedPriorArtChecker(router, None),
        critic=ScientificCritic(router),
        selector=PortfolioSelector(repo),
    )
    await pipeline.generate_portfolio(_gap(repo))

    assert prior_art_calls["n"] >= 2, "revision loop did not run"
    artifacts = repo.artifacts_for_run("run1")
    grounding = [a for a in artifacts if a.artifact_type == "novelty_grounding_json"]
    assert grounding
    record = json.loads(Path(grounding[0].path).read_text())
    revised = repo.row(
        "SELECT id FROM novelty_candidates WHERE method='Revised night-adaptive head'"
    )
    assert revised is not None, "revision did not persist"
    entries = {c["id"]: c["evidence_linked_paper_ids"] for c in record["candidates"]}
    assert revised["id"] in entries, f"revised candidate missing from grounding record: {list(entries)}"
    assert entries[revised["id"]] == ["paper2"]
