"""Offline end-to-end smoke for the five-skill distribution.

Proves, with deterministic fixtures only (FakeLLMProvider, fixture sources,
no network, no paid API), that the full loop works:

    seed → review → novelty → methodology → design → implementation
    generation (statically verified) → approval → execution → measurement
    ingestion → analysis → report → Findings Memory → manuscript.

This is an ORCHESTRATION check — it verifies the skills run and stay honest;
it is not evidence of any scientific claim.
"""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

import pytest
from pypdf import PdfWriter

from autoresearch.application.manuscript import ManuscriptService
from autoresearch.application.services import (
    DiscoveryService,
    ExperimentService,
    LiteratureReviewService,
    NoveltyService,
)
from autoresearch.application.workflow import WorkflowService
from autoresearch.domain.models import DownloadStatus, content_checksum
from autoresearch.papers.metadata_sources import MetadataPaper, SourceSearchResult
from autoresearch.papers.pdf_resolver import DownloadedPdf, PdfResolution
from autoresearch.papers.search_pipeline import StudySearchPipeline
from autoresearch.providers.budget import ProviderBudgetGuard
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


_WORKSPACE = str(Path.cwd())


def _generated_script(prompt: str) -> str:
    """The 'model' implementation: follows the generation contract (copies the
    IDENTITY literal, iterates declared seeds, names every required artifact
    type, implements methodology components as named functions)."""
    identity = (re.search(r"IDENTITY = (\{.*?\})\n", prompt, flags=re.DOTALL) or [None, "{}"])[1]
    seeds = (re.search(r"Random seeds \(iterate ALL of these\): \[([0-9, ]+)\]", prompt) or [None, "42"])[1]
    artifacts = (re.search(r"\(types: \[(.*?)\]\)", prompt) or [None, "'seed_metrics'"])[1]
    return f'''
import json, pathlib, random

IDENTITY = {identity}

def encode_features(values):
    return [v for v in values]

def estimate_domain_statistics(values):
    return sum(values) / max(len(values), 1), (max(values) - min(values)) or 1e-6

def recalibrate_features(values):
    mean, spread = estimate_domain_statistics(values)
    return [(v - mean) / spread for v in values]

def decode_head(value):
    return min(max(value, 0.0), 1.0)

run_dir = pathlib.Path("run_outputs")
run_dir.mkdir(exist_ok=True)
seed_rows = []
for seed in [{seeds}]:
    rng = random.Random(seed)
    base_features = encode_features([rng.random() for _ in range(8)])
    baseline = decode_head(sum(base_features) / len(base_features) + 0.45)
    recalibrated = recalibrate_features(base_features)
    candidate = decode_head(sum(recalibrated) / len(recalibrated) + 0.45 + 0.03)
    seed_rows.append({{"seed": seed, "baseline": {{"accuracy": round(baseline, 6)}}, "candidate": {{"accuracy": round(candidate, 6)}}}})
artifact_rows = []
for artifact_type in [{artifacts}]:
    path = run_dir / (artifact_type + ".txt")
    path.write_text("generated artifact: " + artifact_type + chr(10))
    artifact_rows.append({{"artifact_type": artifact_type, "path": str(path)}})
print(json.dumps({{"identity": IDENTITY, "seeds": seed_rows, "artifacts": artifact_rows}}))
'''


def _responder(messages):
    prompt = str(messages[-1]["content"])
    if "implementation-generator-v1" in prompt:
        return _generated_script(prompt)
    if "seed-analysis-v1" in prompt:
        return {"title": "Seed Adaptive Fusion", "topic": "robust adaptive fusion", "research_question": "Can adaptive fusion improve cross-domain robustness?", "methods": ["adaptive fusion"], "datasets": ["Dataset-A"], "citations": [], "limitations": ["single-domain evaluation"], "keywords": ["robustness"]}
    if "search-strategies-v2" in prompt:
        return {"strategies": [{"name": "topic", "query": "adaptive fusion robustness"}, {"name": "method", "query": "adaptive fusion domain shift"}, {"name": "question", "query": "cross-domain robust fusion"}]}
    if "paper-review-v2" in prompt:
        source = prompt.split("SOURCE:\n", 1)[1].split("\n\nReturn one JSON object", 1)[0]
        sentences = re.findall(r"[^.!?]+[.!?]", source)
        return {
            "problem": "Cross-domain robustness", "research_question": "Does adaptive fusion improve robustness?", "method": "Adaptive fusion",
            "datasets": ["Dataset-A"], "experimental_protocol": "Single-dataset controlled evaluation", "results": ["Improved robustness"],
            "ablations": ["Fusion removed"], "limitations": ["Evaluation uses one dataset"], "failure_modes": ["Unverified domain transfer"], "confidence": 0.86,
            "evidence_spans": [
                {"claim_type": "finding", "statement": sentence.strip(), "excerpt": sentence.strip(), "page": None, "section": "abstract", "confidence": 0.9}
                for sentence in sentences[:2]
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
    if "methodology-freeze-v" in prompt:
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
            "metrics": ["accuracy"], "random_seeds": [42, 43, 44], "required_seed_count": 3,
            "minimum_meaningful_effect": {"accuracy": 0.01}, "metric_direction": {"accuracy": "maximize"},
            "ablations": [], "robustness_tests": [], "statistical_plan": "exact Wilcoxon signed-rank across seeds",
            "compute_budget": "budget", "stopping_conditions": [], "expected_outputs": ["scripts/run_experiment.py"], "artifact_contract": "contract",
            "interpretation_rules": [
                "positive: accuracy gate passes", "negative: accuracy gate fails",
                "null: no difference", "failed: execution or artifact-validation failure",
            ],
        }
    if "Method:" in prompt or "novelty-critic" in prompt:
        return {
            "novelty_score": 0.9, "significance_score": 0.9, "causal_plausibility_score": 0.9, "falsifiability_score": 0.9,
            "experimental_discriminability_score": 0.9, "feasibility_score": 0.9, "implementation_risk_score": 0.1,
            "expected_information_gain_score": 0.9, "compute_efficiency_score": 0.9, "publication_potential_score": 0.9,
            "objections": [], "failure_scenarios": []
        }
    raise AssertionError(f"Unexpected provider prompt: {prompt[:120]}")


@pytest.fixture
def anyio_backend():
    return "asyncio"


def _build_loop(tmp_path: Path):
    """Shared offline harness: repository + workflow + seed PDF, no network."""
    repository = ResearchRepository(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    workspace = tmp_path / "workspace"
    (workspace / "scripts").mkdir(parents=True, exist_ok=True)
    global _WORKSPACE
    _WORKSPACE = str(workspace.resolve())

    registry = ProviderRegistry()
    registry.register_llm("fake", FakeLLMProvider(responder=_responder))
    registry.register_embeddings("fake_embeddings", FakeEmbeddingProvider())
    assignments = {
        ModelRole.LITERATURE_EXTRACTION: ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "fake", "fake-model"),
        ModelRole.RESEARCH_REASONING: ModelAssignment(ModelRole.RESEARCH_REASONING, "fake", "fake-model"),
        ModelRole.SCIENTIFIC_CRITIC: ModelAssignment(ModelRole.SCIENTIFIC_CRITIC, "fake", "fake-model"),
        ModelRole.EMBEDDINGS: ModelAssignment(ModelRole.EMBEDDINGS, "fake_embeddings", "fake-embedding-v1"),
    }
    router = ModelRouter(registry, assignments, usage_sink=repository.record_provider_usage, budget_guard=ProviderBudgetGuard(repository))
    pipeline = StudySearchPipeline(repository, [DeterministicMetadataSource()], DeterministicPdfResolver(), tmp_path / "downloads")
    workflow = WorkflowService(
        repository, router,
        discovery=DiscoveryService(repository, router, pipeline),
        review=LiteratureReviewService(repository, router),
        novelty=NoveltyService(repository, router, pipeline),
        experiment=ExperimentService(repository, router, workspace, worker=LocalTypedWorker(repository, workspace)),
        manuscript=ManuscriptService(repository),
    )
    seed = tmp_path / "seed.pdf"
    writer = PdfWriter(); writer.add_blank_page(width=200, height=200); writer.add_metadata({"/Title": "Seed Adaptive Fusion"})
    with seed.open("wb") as handle:
        writer.write(handle)
    return repository, workflow, seed


@pytest.mark.anyio
async def test_five_skill_loop_runs_end_to_end_offline(tmp_path):
    repository, workflow, seed = _build_loop(tmp_path)

    # Pass 1: seed → … → design, with NO hand-written experiment script: the
    # implementation must be generated and statically verified.
    first = await workflow.run(seed=str(seed), until="design")
    implementation = next(s for s in first["stages"] if s["stage"] == "implementation")
    assert implementation["status"] == "generated", implementation
    assert implementation["fidelity"] == "STATIC_VERIFIED"

    # Approve the exact-spec package, then run the full loop to the manuscript.
    candidate_id = next(s for s in first["stages"] if s["stage"] == "methodology")["candidate_id"]
    preview = workflow.experiment.approval_preview_for_candidate(candidate_id)
    saved = repository.save_exact_approval({
        "id": uuid4().hex, "run_id": preview["run_id"], "specification_checksum": str(preview["checksum"]),
        "approver": "smoke-test", "approved_scope": "workspace", "approved_target": str(preview["worker"]),
        "approved_budget": 60.0, "expires_at": "2099-01-01T00:00:00+00:00", "status": "pending",
    })
    repository.approve_exact(saved["id"], approver="smoke-test")
    result = await workflow.run(project_id=first["project_id"], until="manuscript")

    stages = {s["stage"]: s for s in result["stages"]}
    assert stages["experiment"]["status"] == "completed", stages["experiment"]
    assert stages["manuscript"]["status"] == "complete", stages["manuscript"]
    record = repository.latest_manuscript(first["project_id"])
    assert record is not None and Path(record.markdown_path).is_file()
    findings = repository.findings_for_project(first["project_id"])
    assert findings, "Findings Memory must record the analyzed outcome"


@pytest.mark.anyio
async def test_manuscript_satisfies_fasttrack_template_end_to_end(tmp_path):
    """The generated manuscript must satisfy the FastTrack Academic Research
    Paper Template (template-spec.json) — enforced deterministically, with
    the audit persisted and the evidence bundle carrying keywords,
    limitations, exact commands, and the figure."""
    import json

    from autoresearch.application.manuscript import run_template_audit
    from autoresearch.review.matrix import project_keywords

    repository, workflow, seed = _build_loop(tmp_path)
    first = await workflow.run(seed=str(seed), until="design")
    candidate_id = next(s for s in first["stages"] if s["stage"] == "methodology")["candidate_id"]
    preview = workflow.experiment.approval_preview_for_candidate(candidate_id)
    saved = repository.save_exact_approval({
        "id": uuid4().hex, "run_id": preview["run_id"], "specification_checksum": str(preview["checksum"]),
        "approver": "smoke-test", "approved_scope": "workspace", "approved_target": str(preview["worker"]),
        "approved_budget": 60.0, "expires_at": "2099-01-01T00:00:00+00:00", "status": "pending",
    })
    repository.approve_exact(saved["id"], approver="smoke-test")
    result = await workflow.run(project_id=first["project_id"], until="manuscript")
    stages = {s["stage"]: s for s in result["stages"]}
    assert stages["manuscript"]["status"] == "complete", stages["manuscript"]

    record = repository.latest_manuscript(first["project_id"])
    markdown = Path(record.markdown_path).read_text(encoding="utf-8")
    root = Path(record.markdown_path).parent

    # 1. Template audit persisted and passing.
    audit_path = root / "template_audit.json"
    assert audit_path.is_file(), "template_audit.json must be persisted"
    audit_payload = json.loads(audit_path.read_text(encoding="utf-8"))
    assert audit_payload["passed"], audit_payload["violations"]
    assert audit_payload["inside_out_order"][:3] == ["Methodology", "Results", "Discussion"]
    assert audit_payload["inside_out_order"].index("Introduction") < audit_payload["inside_out_order"].index("Abstract")

    # 2. Abstract: six labeled parts, 150-250 words.
    for label in ("Background", "Gap in Knowledge", "Contribution", "Population & Methods", "Results", "Conclusion"):
        assert f"**{label}:**" in markdown, f"abstract missing part {label}"
    abstract_words = len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", markdown.split("## Keywords")[0].split("## Abstract")[1]))
    assert 150 <= abstract_words <= 250, abstract_words

    # 3. Keywords: 3-6 terms.
    keywords_section = markdown.split("## Keywords")[1].split("## 1. Introduction")[0]
    terms = [t.strip().rstrip(";,") for t in re.split(r"[,\n]", keywords_section) if t.strip().rstrip(";,")]
    assert 3 <= len(terms) <= 6, terms

    # 4. Discussion: at least 3 numbered limitations + required subsections.
    discussion = markdown.split("## 5. Discussion")[1].split("## References")[0]
    limitations_block = re.search(r"### Study Limitations[^\n]*\n(.*?)(?=\n### |\Z)", discussion, re.DOTALL).group(1)
    assert len(re.findall(r"^\d+\.\s", limitations_block, re.MULTILINE)) >= 3
    for part in ("### Recap of what we found", "### Strengths of this study", "### Implications for future research"):
        assert part in discussion

    # 5. Section order follows the template.
    headings = re.findall(r"^## (.+)$", markdown, re.MULTILINE)
    body = [re.sub(r"^\d+\.\s*", "", h) for h in headings]
    expected = ["Introduction", "Literature Review", "Methodology", "Results", "Discussion", "References", "Appendices"]
    assert [h for h in body if h in expected] == expected, body

    # 6. Results: table + deterministic figure artifact.
    assert "| Metric |" in markdown
    figure = root / "figures" / "effect-sizes.svg"
    assert figure.is_file()
    assert "figures/effect-sizes.svg" in markdown

    # 7. Title: <= 20 words, contains a keyword, no vague blocklist.
    title = markdown.splitlines()[0].removeprefix("# ").strip()
    assert len(re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", title)) <= 20, title
    assert any(re.search(rf"\b{re.escape(t.lower())}\b", title.lower()) for t in terms), title
    for vague in ("an analysis", "a study of", "an investigation", "new insights"):
        assert vague not in title.lower()

    # 8. Results contains no interpretation words.
    results_section = markdown.split("## 4. Results")[1].split("## 5. Discussion")[0].lower()
    for word in ("shows that", "suggests", "demonstrates the importance", "because", "indicates", "implies"):
        assert word not in results_section, word

    # 9. Evidence bundle carries keywords, >= 3 limitations, figures.
    bundle = json.loads(Path(record.evidence_bundle_path).read_text(encoding="utf-8"))
    assert 3 <= len(bundle["keywords"]) <= 6
    assert len(bundle["limitations"]) >= 3
    assert bundle["figures"] == ["figures/effect-sizes.svg"]

    # 10. Experiment report carries >= 3 limitation rows (list format).
    spec = repository.get_experiment_spec(candidate_id)
    analysis = repository.get_statistical_analysis(spec.id)
    report = repository.get_experiment_report(analysis.id)
    assert len(report.limitations) >= 3
    assert all(item.strip() for item in report.limitations)

    # 11. Review produced project keywords (3-6) in the evidence matrix.
    review_run_id = next(s for s in first["stages"] if s["stage"] == "review")["run_id"]
    keywords = project_keywords(repository, project_id=first["project_id"], run_id=review_run_id)
    assert 3 <= len(keywords) <= 6, keywords
