"""Purpose: Enforces type-checked dataflow between measurement artifacts and conclusions."""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping, Sequence


class ProvenanceType(StrEnum):
    MEASURED = "MEASURED"
    DERIVED = "DERIVED"
    EXTRACTED_FROM_SOURCE = "EXTRACTED_FROM_SOURCE"
    DATABASE_FACT = "DATABASE_FACT"
    CACHED_COMPATIBLE_RUN = "CACHED_COMPATIBLE_RUN"
    MODEL_ESTIMATE = "MODEL_ESTIMATE"
    MODEL_HYPOTHESIS = "MODEL_HYPOTHESIS"
    FIXTURE = "FIXTURE"
    MISSING = "MISSING"


class ExperimentStage(StrEnum):
    PLANNED = "PLANNED"
    VALIDATING = "VALIDATING"
    VALIDATED = "VALIDATED"
    APPROVED = "APPROVED"
    EXECUTING = "EXECUTING"
    EXECUTION_COMPLETED = "EXECUTION_COMPLETED"
    ARTIFACT_VALIDATION = "ARTIFACT_VALIDATION"
    ANALYZING = "ANALYZING"
    REPORTED = "REPORTED"
    INSUFFICIENT_EVIDENCE = "INSUFFICIENT_EVIDENCE"


class IntegrityFailure(ValueError):
    """A stable, user-visible reason why a scientific claim cannot be made."""


@dataclass(frozen=True)
class ScientificValue:
    value: Any
    provenance_type: ProvenanceType
    source_id: str | None
    source_artifact_id: str | None
    created_at: str
    verified: bool
    origin_run_id: str | None = None

    def require_observed(self) -> None:
        if self.provenance_type is ProvenanceType.FIXTURE:
            raise IntegrityFailure("FIXTURE_PROHIBITED")
        if self.provenance_type not in {ProvenanceType.MEASURED, ProvenanceType.CACHED_COMPATIBLE_RUN}:
            raise IntegrityFailure("INSUFFICIENT_EVIDENCE")
        if not self.verified or not self.source_artifact_id:
            raise IntegrityFailure("INSUFFICIENT_EVIDENCE")


@dataclass(frozen=True)
class PredictedMetric(ScientificValue):
    def __post_init__(self) -> None:
        if self.provenance_type is not ProvenanceType.MODEL_HYPOTHESIS:
            raise ValueError("PredictedMetric must be MODEL_HYPOTHESIS")


@dataclass(frozen=True)
class ObservedMetric(ScientificValue):
    def __post_init__(self) -> None:
        self.require_observed()


REQUIRED_ARTIFACT_TYPES = frozenset({
    "implementation_diff", "training_log", "checkpoint", "seed_metrics",
    "parameter_profile", "flop_profile", "latency_profile", "evaluation_report",
})


IDENTITY_FIELDS = (
    "candidate_id", "method_id", "repository_commit", "implementation_hash",
    "config_hash", "dataset_checksum", "split_checksum", "metric_version",
)


def validate_identity(identity: Mapping[str, Any], spec: Any, *, seed: int | None = None) -> None:
    expected = {
        "candidate_id": spec.candidate_id,
        "dataset_checksum": spec.dataset_checksum,
        "split_checksum": getattr(spec, "split_checksum", None) or spec.split_definition,
    }
    for field in IDENTITY_FIELDS:
        if not identity.get(field):
            raise IntegrityFailure("CACHE_INCOMPATIBLE")
        if expected.get(field) and str(identity[field]) != str(expected[field]):
            raise IntegrityFailure("CACHE_INCOMPATIBLE")
    if seed is not None and int(identity.get("seed", -1)) != seed:
        raise IntegrityFailure("CACHE_INCOMPATIBLE")


def validate_artifact_contract(spec: Any, artifacts: Sequence[Mapping[str, Any]], *, seed_rows: Sequence[Mapping[str, Any]]) -> None:
    required = set(getattr(spec, "required_artifacts", ()) or REQUIRED_ARTIFACT_TYPES)
    available = {str(a.get("artifact_type")) for a in artifacts if a.get("verified")}
    missing = required - available
    if missing:
        raise IntegrityFailure("MISSING_REQUIRED_ARTIFACTS:" + ",".join(sorted(missing)))
    for artifact in artifacts:
        if str(artifact.get("provenance_type", "")) == ProvenanceType.FIXTURE.value:
            raise IntegrityFailure("FIXTURE_PROHIBITED")
    if len(seed_rows) < int(spec.required_seed_count):
        raise IntegrityFailure("INCOMPLETE_EXPERIMENT")
    seeds = {int(row.get("seed", -1)) for row in seed_rows}
    if len(seeds) < int(spec.required_seed_count):
        raise IntegrityFailure("INCOMPLETE_EXPERIMENT")


def decision_from_gates(gates: Mapping[str, bool | None]) -> str:
    if not gates or any(value is None for value in gates.values()):
        return "INSUFFICIENT_EVIDENCE"
    if all(bool(value) for value in gates.values()):
        return "SUPPORTED"
    if not any(bool(value) for value in gates.values()):
        return "REFUTED"
    return "PARTIAL_SUPPORT"
