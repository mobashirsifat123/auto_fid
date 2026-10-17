"""Typed domain records shared by every AutoResearch pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Sequence


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def content_checksum(value: Any) -> str:
    """Return a stable SHA-256 checksum for bytes, text, or JSON-compatible data."""
    if isinstance(value, bytes):
        payload = value
    elif isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class PipelineKind(StrEnum):
    DISCOVERY = "study_search"
    REVIEW = "deep_literature_review"
    NOVELTY = "novelty_generation"
    EXPERIMENT = "experiment_design_execution"
    REINVENTION = "result_conditioned_reinvention"
    MANUSCRIPT = "manuscript_generation"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DownloadStatus(StrEnum):
    DOWNLOADED = "downloaded"
    UNAVAILABLE = "unavailable"
    REJECTED = "rejected"
    FAILED = "failed"


class GapClassification(StrEnum):
    EVIDENCE_SUPPORTED = "evidence_supported"
    HYPOTHESIS = "hypothesis"


@dataclass(frozen=True)
class ArtifactProvenance:
    project_id: str
    run_id: str
    source_evidence: tuple[str, ...]
    provider: str
    model: str
    prompt_version: str
    created_at: str
    checksum: str

    @classmethod
    def create(
        cls,
        *,
        project_id: str,
        run_id: str,
        source_evidence: Sequence[str],
        provider: str,
        model: str,
        prompt_version: str,
        content: Any,
    ) -> "ArtifactProvenance":
        return cls(
            project_id=project_id,
            run_id=run_id,
            source_evidence=tuple(source_evidence),
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            created_at=utc_now(),
            checksum=content_checksum(content),
        )


@dataclass(frozen=True)
class Project:
    id: str
    title: str
    created_at: str = field(default_factory=utc_now)
    # Exact database path this project's state lives in (recorded at
    # creation; lets later commands anchor to the project from any working
    # directory, even with a custom database path).
    state_db_path: str | None = None


@dataclass(frozen=True)
class PipelineRun:
    id: str
    project_id: str
    kind: PipelineKind
    status: RunStatus
    input_data: Mapping[str, Any]
    started_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class SeedPaper:
    id: str
    project_id: str
    run_id: str
    input_type: str
    input_value: str
    title: str
    text: str
    topic: str
    research_question: str
    checksum: str
    provenance: ArtifactProvenance
    doi: str | None = None
    url: str | None = None
    methods: tuple[str, ...] = ()
    datasets: tuple[str, ...] = ()
    citations: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class PaperRecord:
    id: str
    project_id: str
    run_id: str
    title: str
    abstract: str
    authors: tuple[str, ...]
    year: int | None
    publication_date: str | None
    doi: str | None
    arxiv_id: str | None
    url: str | None
    source_names: tuple[str, ...]
    open_access_pdf_url: str | None = None
    is_open_access: bool = False
    citation_count: int = 0
    venue: str = ""
    checksum: str = ""
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class SearchCandidate:
    id: str
    project_id: str
    run_id: str
    paper_id: str
    source: str
    strategy: str
    source_rank: int
    query: str
    raw_checksum: str
    retrieved_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class DownloadRecord:
    id: str
    project_id: str
    run_id: str
    paper_id: str
    status: DownloadStatus
    source: str
    url: str | None
    legal_basis: str | None
    license: str | None
    path: str | None
    reason: str | None
    checksum: str | None
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class EvidenceClaim:
    id: str
    project_id: str
    run_id: str
    paper_id: str
    claim_type: str
    statement: str
    excerpt: str
    page: int | None
    section: str | None
    confidence: float
    provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        if not self.excerpt.strip():
            raise ValueError("Evidence claims require a non-empty source excerpt.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Evidence confidence must be between 0 and 1.")


@dataclass(frozen=True)
class PaperReview:
    id: str
    project_id: str
    run_id: str
    paper_id: str
    problem: str
    research_question: str
    method: str
    datasets: tuple[str, ...]
    experimental_protocol: str
    results: tuple[str, ...]
    ablations: tuple[str, ...]
    limitations: tuple[str, ...]
    failure_modes: tuple[str, ...]
    evidence_claim_ids: tuple[str, ...]
    confidence: float
    provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Review confidence must be between 0 and 1.")
        if not self.evidence_claim_ids:
            raise ValueError("Paper reviews require at least one evidence claim.")


@dataclass(frozen=True)
class PriorityScore:
    id: str
    project_id: str
    run_id: str
    paper_id: str
    semantic_similarity: float
    methodological_similarity: float
    research_question_similarity: float
    recency: float
    influence: float
    evidence_quality: float
    total: float
    explanation: Mapping[str, Any]
    provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        values = (self.semantic_similarity, self.methodological_similarity, self.research_question_similarity, self.recency, self.influence, self.evidence_quality, self.total)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("Priority score components must be between 0 and 1.")


@dataclass(frozen=True)
class ResearchGap:
    id: str
    project_id: str
    run_id: str
    statement: str
    supporting_paper_ids: tuple[str, ...]
    supporting_claim_ids: tuple[str, ...]
    why_unsolved: str
    missing_experiment: str
    falsification_test: str
    confidence: float
    counter_evidence_claim_ids: tuple[str, ...]
    classification: GapClassification
    provenance: ArtifactProvenance

    def __post_init__(self) -> None:
        supported = bool(self.supporting_paper_ids and self.supporting_claim_ids)
        if self.classification is GapClassification.EVIDENCE_SUPPORTED and not supported:
            raise ValueError("An evidence-supported gap requires supporting papers and claims.")


@dataclass(frozen=True)
class ProviderUsage:
    id: str
    project_id: str | None
    run_id: str | None
    operation: str
    role: str
    provider: str
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    estimated_cost: float | None
    currency: str | None
    request_id: str | None
    latency_ms: float | None
    session_id: str | None = None
    prompt_version: str = ""
    retry_count: int = 0
    structured_output_status: str | None = None
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class ArtifactReference:
    id: str
    project_id: str
    run_id: str
    artifact_type: str
    path: str
    source_evidence: tuple[str, ...]
    provider: str
    model: str
    prompt_version: str
    checksum: str
    created_at: str = field(default_factory=utc_now)


class PriorArtStatus(StrEnum):
    UNSUPPORTED = "unsupported"
    POTENTIALLY_NOVEL = "potentially_novel"
    PARTIAL_OVERLAP = "partial_overlap"
    INCREMENTAL = "incremental"
    KNOWN_METHOD = "known_method"
    EXPERT_REVIEW_REQUIRED = "expert_review_required"


class CandidateStatus(StrEnum):
    GENERATED = "generated"
    REVISING = "revising"
    REJECTED = "rejected"
    PORTFOLIO_SELECTED = "portfolio_selected"


class ExperimentDecision(StrEnum):
    SUPPORTED = "supported"
    PARTIALLY_SUPPORTED = "partially_supported"
    REFUTED = "refuted"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    NO_EFFECT = "no_effect"
    IMPLEMENTATION_FAILURE = "implementation_failure"
    OPTIMIZATION_FAILURE = "optimization_failure"
    DATA_LIMITATION = "data_limitation"
    EVALUATION_ARTIFACT = "evaluation_artifact"
    COST_PROHIBITIVE = "cost_prohibitive"
    DOMAIN_SPECIFIC = "domain_specific"
    MECHANISM_CONTRADICTED = "mechanism_contradicted"


@dataclass(frozen=True)
class NoveltyCandidate:
    id: str
    project_id: str
    run_id: str
    gap_id: str
    strategy: str
    method: str
    causal_mechanism: str
    target_limitation: str
    difference_from_existing: str
    expected_effect: str
    falsification_condition: str
    minimum_validating_experiment: str
    required_ablations: tuple[str, ...]
    implementation_complexity: str
    expected_compute: str
    risks_and_objections: tuple[str, ...]
    status: CandidateStatus
    provenance: ArtifactProvenance


@dataclass(frozen=True)
class PriorArtAssessment:
    id: str
    project_id: str
    run_id: str
    candidate_id: str
    closest_paper_ids: tuple[str, ...]
    overlap_status: PriorArtStatus
    mechanism_comparison: str
    supporting_excerpts: tuple[str, ...]
    unresolved_overlap: str
    provenance: ArtifactProvenance


@dataclass(frozen=True)
class NoveltyCritique:
    id: str
    project_id: str
    run_id: str
    candidate_id: str
    novelty_score: float
    significance_score: float
    causal_plausibility_score: float
    falsifiability_score: float
    experimental_discriminability_score: float
    feasibility_score: float
    implementation_risk_score: float
    expected_information_gain_score: float
    compute_efficiency_score: float
    publication_potential_score: float
    objections: tuple[str, ...]
    failure_scenarios: tuple[str, ...]
    provenance: ArtifactProvenance
    rubric: Mapping[str, str] = field(default_factory=dict)
    evidence: Mapping[str, tuple[str, ...]] = field(default_factory=dict)
    confidence: float = 0.0

    def __post_init__(self) -> None:
        values = (
            self.novelty_score, self.significance_score, self.causal_plausibility_score,
            self.falsifiability_score, self.experimental_discriminability_score, self.feasibility_score,
            self.implementation_risk_score, self.expected_information_gain_score,
            self.compute_efficiency_score, self.publication_potential_score
        )
        if any(not 0.0 <= v <= 1.0 for v in values):
            raise ValueError("All critique scores must be between 0 and 1.")


@dataclass(frozen=True)
class CandidateRevision:
    id: str
    project_id: str
    run_id: str
    original_candidate_id: str
    revised_candidate_id: str
    critique_id: str
    revision_count: int
    changes_made: str
    provenance: ArtifactProvenance


@dataclass(frozen=True)
class NoveltyPortfolio:
    id: str
    project_id: str
    run_id: str
    low_risk_candidate_id: str | None
    balanced_candidate_id: str | None
    high_risk_candidate_id: str | None
    selection_reasons: Mapping[str, str]
    provenance: ArtifactProvenance


@dataclass(frozen=True)
class NoveltyClaimLedger:
    id: str
    project_id: str
    run_id: str
    candidate_id: str
    contribution_claim: str
    closest_methods: tuple[str, ...]
    overlap_status: PriorArtStatus
    supporting_evidence: tuple[str, ...]
    unresolved_uncertainty: str
    safe_wording: str
    prohibited_wording: tuple[str, ...]
    review_status: str
    provenance: ArtifactProvenance


@dataclass(frozen=True)
class MethodologyFreeze:
    """A versioned, frozen experimental methodology derived from one novelty
    candidate. Freezing records *what will be tested*; it never certifies
    novelty or scientific truth (``uncertainty`` must be preserved verbatim).
    """

    id: str
    project_id: str
    run_id: str
    candidate_id: str
    version: int
    selection_rationale: str
    prompt_version: str
    artifact_path: str
    checksum: str
    created_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class ExperimentSpec:
    id: str
    project_id: str
    run_id: str
    candidate_id: str
    hypothesis: str
    causal_mechanism: str
    falsification_condition: str
    repository: str
    baseline_commit: str
    dataset_identity: str
    dataset_checksum: str
    split_definition: str
    baseline_configuration: str
    proposed_change: str
    independent_variables: tuple[str, ...]
    controlled_variables: tuple[str, ...]
    metrics: tuple[str, ...]
    random_seeds: tuple[int, ...]
    ablations: tuple[str, ...]
    robustness_tests: tuple[str, ...]
    statistical_plan: str
    compute_budget: str
    stopping_conditions: tuple[str, ...]
    expected_outputs: tuple[str, ...]
    artifact_contract: str
    provenance: ArtifactProvenance
    metric_direction: Mapping[str, str] = field(default_factory=dict)
    minimum_meaningful_effect: Mapping[str, float] = field(default_factory=dict)
    required_seed_count: int = 1
    network_policy: str = "disabled"
    target_worker: str = "local"
    approval_required: bool = True
    # Cross-stage provenance: the question (gap) and frozen methodology this
    # design consumes, plus pre-registered interpretation rules.
    gap_id: str = ""
    methodology_freeze_id: str = ""
    interpretation_rules: tuple[str, ...] = ()
    # The contract is deliberately explicit: validation output is not evidence.
    required_artifacts: tuple[str, ...] = (
        "implementation_diff", "training_log", "checkpoint", "seed_metrics",
        "parameter_profile", "flop_profile", "latency_profile", "evaluation_report",
    )


@dataclass(frozen=True)
class EnvironmentManifest:
    id: str
    project_id: str
    run_id: str
    spec_id: str
    operating_system: str
    python_version: str
    environment_manager: str
    dependency_lock_checksum: str
    cuda_compatibility: str
    gpu_model: str
    gpu_memory: str
    disk_space: str
    dataset_paths: tuple[str, ...]
    sample_counts: Mapping[str, int]
    splits: tuple[str, ...]
    pretrained_checkpoint_identity: str
    reproducibility_flags: str
    provenance: ArtifactProvenance


@dataclass(frozen=True)
class ValidationReport:
    id: str
    project_id: str
    run_id: str
    manifest_id: str
    schema_validation_passed: bool
    static_checks_passed: bool
    unit_tests_passed: bool
    import_test_passed: bool
    one_batch_forward_passed: bool
    backward_gradient_passed: bool
    checkpoint_save_reload_passed: bool
    short_smoke_training_passed: bool
    evaluation_parsing_passed: bool
    budget_approval_passed: bool
    details: Mapping[str, Any]
    provenance: ArtifactProvenance

    @property
    def passed(self) -> bool:
        return all((
            self.schema_validation_passed, self.static_checks_passed, self.unit_tests_passed,
            self.import_test_passed, self.one_batch_forward_passed, self.backward_gradient_passed,
            self.checkpoint_save_reload_passed, self.short_smoke_training_passed,
            self.evaluation_parsing_passed, self.budget_approval_passed
        ))


@dataclass(frozen=True)
class ExecutionJob:
    id: str
    project_id: str
    run_id: str
    spec_id: str
    worker: str
    stage: str
    status: RunStatus
    started_at: str = field(default_factory=utc_now)
    completed_at: str | None = None
    timeout_seconds: int = 3600
    retry_policy: str = "none"


@dataclass(frozen=True)
class ExecutionResult:
    id: str
    project_id: str
    run_id: str
    job_id: str
    exit_code: int
    stdout: str
    stderr: str
    structured_metrics: Mapping[str, Any]
    gpu_telemetry: Mapping[str, Any]
    checkpoint_manifest: tuple[str, ...]
    failure_classification: str | None
    provenance: ArtifactProvenance
    identity: Mapping[str, Any] = field(default_factory=dict)
    artifacts: tuple[Mapping[str, Any], ...] = ()
    command_json: str = ""


@dataclass(frozen=True)
class StatisticalAnalysis:
    id: str
    project_id: str
    run_id: str
    spec_id: str
    baseline_metrics: Mapping[str, Any]
    proposed_metrics: Mapping[str, Any]
    effect_size: Mapping[str, float]
    confidence_intervals: Mapping[str, tuple[float, float]]
    seed_level_results: tuple[Mapping[str, Any], ...]
    ablation_contributions: Mapping[str, float]
    parameter_count_diff: int
    flops_latency_diff: float | None
    training_inference_cost: float
    failure_case_analysis: str
    hypothesis_support_assessment: str
    provenance: ArtifactProvenance
    decision_gates: Mapping[str, bool | None] = field(default_factory=dict)
    statistical_tests: Mapping[str, Any] = field(default_factory=dict)
    evidence_status: str = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class ExperimentReport:
    id: str
    project_id: str
    run_id: str
    analysis_id: str
    literature_motivation: str
    research_gap: str
    hypothesis: str
    proposed_method: str
    implementation_diff: str
    lineage: str
    dataset_split: str
    exact_commands: tuple[str, ...]
    baseline_results: Mapping[str, Any]
    aggregated_results: Mapping[str, Any]
    uncertainty: str
    ablations: str
    robustness_tests: str
    failed_runs: int
    cost: float
    limitations: tuple[str, ...]
    reproducibility_instructions: str
    decision: ExperimentDecision
    decision_evidence: str
    provenance: ArtifactProvenance


@dataclass(frozen=True)
class ManuscriptRecord:
    """A versioned manuscript over one exact research lineage.

    ``status`` is ``complete`` only when every lineage prerequisite (review
    gap, methodology freeze, experiment spec, statistical analysis, report)
    existed at generation time; otherwise ``incomplete`` and ``gaps_json``
    lists the precise missing evidence. Paths point at editable artifacts
    (Markdown, LaTeX, BibTeX, tables, manifest, optional compiled PDF).
    """

    id: str
    project_id: str
    run_id: str
    title: str
    status: str
    decision: str
    lineage_json: str
    evidence_bundle_path: str
    markdown_path: str
    latex_path: str
    bibtex_path: str
    manifest_path: str
    pdf_path: str | None
    gaps_json: str
    provenance: ArtifactProvenance
    created_at: str = field(default_factory=utc_now)


class FindingStatus(StrEnum):
    """Epistemic status of a research finding.

    The taxonomy keeps DeepScientist-style cumulative memory honest: a claim's
    status can only rise through the pipeline's deterministic gates, never
    through narrative confidence. ``confirmed_finding`` here means
    gate-confirmed (decision gates + recomputed metrics agree), NOT
    independently replicated.
    """

    UNVERIFIED_IDEA = "unverified_idea"
    IMPLEMENTED_METHOD = "implemented_method"
    MEASURED_OBSERVATION = "measured_observation"
    INTERPRETATION = "interpretation"
    CONFIRMED_FINDING = "confirmed_finding"
    REFUTED_HYPOTHESIS = "refuted_hypothesis"
    IMPLEMENTATION_FAILURE = "implementation_failure"
    INCONCLUSIVE = "inconclusive"


@dataclass(frozen=True)
class ResearchFinding:
    """One structured entry of the shared Findings Memory.

    Adapted (mechanism only) from DeepScientist's cumulative Findings Memory
    (ICLR 2026, Apache-2.0): every informative outcome — successful, failed,
    negative, or inconclusive — is persisted with its exact lineage so later
    stages retrieve failures and contradictions instead of rediscovering them.
    Observation and interpretation are stored in separate fields and never
    merged.
    """

    id: str
    project_id: str
    run_id: str
    status: FindingStatus
    question_id: str
    candidate_id: str
    methodology_freeze_id: str
    spec_id: str
    analysis_id: str
    report_id: str
    hypothesis: str
    expected_observation: str
    implementation_digest: str
    observed_outcome_json: str
    interpretation: str
    uncertainty: str
    fidelity: str
    failure_classification: str
    lesson: str
    applicability: str
    open_questions_json: str
    next_action: str
    keywords_json: str
    provenance: ArtifactProvenance
    created_at: str = field(default_factory=utc_now)


class ReinventionAction(StrEnum):
    REFINE = "refine"
    PIVOT = "pivot"
    REBUILD = "rebuild"
    ABORT = "abort"


@dataclass(frozen=True)
class ReinventionAnalysis:
    id: str
    project_id: str
    run_id: str
    spec_id: str
    report_id: str
    parent_candidate_id: str
    action: ReinventionAction
    root_cause_diagnosis: str
    empirical_lessons: tuple[str, ...]
    next_generation_hypothesis: str
    revised_mechanism: str
    actionable_next_steps: tuple[str, ...]
    provenance: ArtifactProvenance
    next_candidate_id: str | None = None


def dataclass_payload(value: Any) -> dict[str, Any]:
    """Serialize nested dataclasses and enums into a JSON-compatible mapping."""
    return json.loads(json.dumps(asdict(value), default=str))
