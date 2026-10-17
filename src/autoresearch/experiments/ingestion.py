"""Purpose: Ingests worker execution output into validated measurement state.

The contract between an executed experiment implementation and the analysis
layer is explicit and fail-closed:

* the implementation prints exactly one measurement document on stdout as
  single-line JSON (``json.dumps`` of a dict carrying ``seeds``);
* the document carries ``seeds`` (per-seed ``baseline``/``candidate`` metric
  values), optional per-seed ``identity``, ``artifacts`` descriptors for the
  spec's required artifact types, and optionally ``identity``;
* identity fields the specification already pins (``candidate_id``,
  ``dataset_checksum``, ``split_checksum``) may be omitted by the document
  and are asserted from the spec — the typed worker's approval-checksum
  binding establishes that the approved implementation produced this output.
  Script-provided values must match the spec exactly or ingestion fails;
* every artifact descriptor must point at a real file inside the spec's
  repository; checksums are recomputed and cross-checked when claimed;
* any FIXTURE provenance anywhere fails closed.

Nothing here interprets results: it only moves measured values into the typed
``ExecutionResult`` the ``StatisticalAnalyzer`` consumes. Analysis still
applies its own identity, artifact-contract, and seed-count gates.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, ArtifactReference, ExecutionResult, ExperimentSpec
from autoresearch.experiments.integrity import IntegrityFailure, ProvenanceType

MEASUREMENT_PROMPT_VERSION = "measurement-ingestion-v1"

# Metrics the ingester can independently recompute from raw outputs. A
# checksum proves file identity, not that the file holds a legitimate
# experiment; recomputation from predictions/labels closes that gap for the
# supported metrics. Unsupported metrics pass through unverified (the analysis
# still reports them; they are never labelled recomputed).
_RECOMPUTABLE_METRICS = ("accuracy", "acc")


def _recompute_metric(name: str, predictions: list, labels: list) -> float | None:
    if name.lower() not in _RECOMPUTABLE_METRICS:
        return None
    if len(predictions) != len(labels) or not labels:
        raise IntegrityFailure(f"METRIC_RECOMPUTATION_INVALID:{name}: predictions/labels length mismatch or empty")
    correct = sum(1 for predicted, actual in zip(predictions, labels) if predicted == actual)
    return correct / len(labels)


def _verify_recomputation(document: Mapping[str, Any]) -> dict[str, Any]:
    """Recompute supported metrics from raw per-seed outputs and reject any
    mismatch with the claimed values. Seeds without raw outputs are counted
    and reported honestly rather than silently trusted."""
    verified: dict[str, Any] = {"recomputed_metrics": [], "unverified_rows": 0}
    for row in document.get("seeds") or []:
        if not isinstance(row, dict):
            continue
        raw = row.get("raw_outputs") or {}
        predictions = raw.get("predictions")
        labels = raw.get("labels")
        if not isinstance(predictions, list) or not isinstance(labels, list):
            verified["unverified_rows"] += 1
            continue
        for arm in ("baseline", "candidate"):
            claimed = dict(row.get(arm) or {})
            for metric, value in claimed.items():
                recomputed = _recompute_metric(str(metric), predictions if arm == "candidate" else raw.get("baseline_predictions", predictions), labels)
                if recomputed is None:
                    continue
                if abs(float(recomputed) - float(value)) > 1e-6:
                    raise IntegrityFailure(
                        f"METRIC_RECOMPUTATION_MISMATCH:{arm}.{metric}: claimed {value}, recomputed {recomputed:.6f} from raw outputs"
                    )
                verified["recomputed_metrics"].append(f"{arm}.{metric}@seed{row.get('seed')}")
    return verified

# Identity fields the specification itself pins; the worker's checksum
# binding already proves these describe the executed implementation.
_SPEC_PINNED_IDENTITY = ("candidate_id", "dataset_checksum", "split_checksum")
# Identity fields the measurement document must provide itself.
_DOCUMENT_IDENTITY_FIELDS = ("method_id", "repository_commit", "implementation_hash", "config_hash", "metric_version")


class MeasurementIngester:
    def __init__(self, repository) -> None:
        self.repository = repository

    def extract_document(self, stdout: str) -> dict[str, Any]:
        """Return the last single-line JSON object on stdout carrying ``seeds``."""
        for line in reversed(stdout.splitlines()):
            stripped = line.strip()
            if not stripped.startswith("{"):
                continue
            try:
                candidate = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict) and "seeds" in candidate:
                return candidate
        try:  # fallback: the whole output is one pretty-printed document
            whole = json.loads(stdout)
            if isinstance(whole, dict) and "seeds" in whole:
                return whole
        except json.JSONDecodeError:
            pass
        raise IntegrityFailure(
            "MEASUREMENT_DOCUMENT_MISSING: the executed implementation printed no "
            "single-line JSON measurement document with a 'seeds' array on stdout"
        )

    def _resolve_identity(self, spec: ExperimentSpec, document: Mapping[str, Any]) -> dict[str, Any]:
        provided = {key: value for key, value in dict(document.get("identity") or {}).items() if value not in (None, "")}
        pinned = {
            "candidate_id": spec.candidate_id,
            "dataset_checksum": spec.dataset_checksum,
            "split_checksum": spec.split_definition,
        }
        for field, expected in pinned.items():
            if field in provided and str(provided[field]) != str(expected):
                raise IntegrityFailure(f"IDENTITY_MISMATCH:{field}")
        resolved = {**provided, **{field: str(value) for field, value in pinned.items()}}
        missing = [field for field in _DOCUMENT_IDENTITY_FIELDS if not resolved.get(field)]
        if missing:
            raise IntegrityFailure(
                "IDENTITY_INCOMPLETE:" + ",".join(missing)
            )
        return resolved

    def _register_artifacts(self, spec: ExperimentSpec, document: Mapping[str, Any], job_id: str) -> tuple[dict[str, Any], ...]:
        descriptors = [row for row in (document.get("artifacts") or []) if isinstance(row, dict)]
        if not descriptors:
            raise IntegrityFailure("ARTIFACTS_MISSING: the measurement document carries no artifact descriptors")
        repo_root = Path(spec.repository).expanduser().resolve()
        registered: list[dict[str, Any]] = []
        for descriptor in descriptors:
            artifact_type = str(descriptor.get("artifact_type") or "")
            if not artifact_type:
                raise IntegrityFailure("ARTIFACT_TYPE_MISSING")
            provenance_type = str(descriptor.get("provenance_type") or ProvenanceType.MEASURED.value)
            if provenance_type == ProvenanceType.FIXTURE.value:
                raise IntegrityFailure(f"FIXTURE_PROHIBITED:{artifact_type}")
            raw_path = Path(str(descriptor.get("path") or ""))
            resolved_path = raw_path if raw_path.is_absolute() else repo_root / raw_path
            try:
                resolved_path = resolved_path.resolve()
                resolved_path.relative_to(repo_root)
            except ValueError as exc:
                raise IntegrityFailure(f"ARTIFACT_OUTSIDE_REPOSITORY:{artifact_type}") from exc
            if not resolved_path.is_file():
                raise IntegrityFailure(f"ARTIFACT_FILE_MISSING:{descriptor.get('path')}")
            digest = hashlib.sha256(resolved_path.read_bytes()).hexdigest()
            claimed = str(descriptor.get("checksum") or "").strip().lower().removeprefix("sha256:")
            if claimed and claimed != digest:
                raise IntegrityFailure(f"ARTIFACT_CHECKSUM_MISMATCH:{artifact_type}")
            artifact = ArtifactReference(
                id=uuid4().hex, project_id=spec.project_id, run_id=spec.run_id,
                artifact_type=artifact_type, path=str(resolved_path),
                source_evidence=(job_id,), provider="deterministic",
                model="typed-worker", prompt_version=MEASUREMENT_PROMPT_VERSION,
                checksum=digest,
            )
            self.repository.save_artifact(artifact)
            registered.append({
                "id": artifact.id, "artifact_type": artifact_type, "path": str(resolved_path),
                "checksum": digest, "verified": True, "provenance_type": provenance_type,
            })
        return tuple(registered)

    def ingest(self, spec: ExperimentSpec, worker_result: Mapping[str, Any], *, job_id: str) -> ExecutionResult:
        """Convert a completed worker result into a typed, artifact-backed
        ExecutionResult. Raises IntegrityFailure with a precise reason when
        the output cannot support scientific analysis."""
        document = self.extract_document(str(worker_result.get("stdout") or ""))
        for row in document.get("seeds") or []:
            if isinstance(row, dict) and str(row.get("provenance_type", "")) == ProvenanceType.FIXTURE.value:
                raise IntegrityFailure("FIXTURE_PROHIBITED:seed_row")
        # Trust chain beyond checksums: when the implementation emits raw
        # predictions/labels, supported metrics are independently recomputed
        # and must match the claimed values exactly.
        document["recomputation"] = _verify_recomputation(document)
        identity = self._resolve_identity(spec, document)
        # Bind each seed row's declared seed into its identity so the
        # analyzer's per-seed validation can verify it (the document-level
        # identity intentionally has no single seed value).
        for row in document.get("seeds") or []:
            if isinstance(row, dict) and "identity" not in row and "seed" in row:
                row["identity"] = {**identity, "seed": row["seed"]}
        artifacts = self._register_artifacts(spec, document, job_id)
        provenance = ArtifactProvenance.create(
            project_id=spec.project_id, run_id=spec.run_id,
            source_evidence=[job_id, *[str(item["id"]) for item in artifacts]],
            provider="deterministic", model="measurement-ingester",
            prompt_version=MEASUREMENT_PROMPT_VERSION,
            content={"job_id": job_id, "exit_code": worker_result.get("exit_code")},
        )
        return ExecutionResult(
            id=uuid4().hex, project_id=spec.project_id, run_id=spec.run_id, job_id=job_id,
            exit_code=int(worker_result.get("exit_code") or 0),
            stdout=str(worker_result.get("stdout") or ""), stderr=str(worker_result.get("stderr") or ""),
            structured_metrics=document, gpu_telemetry=dict(document.get("gpu_telemetry") or {}),
            checkpoint_manifest=tuple(str(item) for item in (document.get("checkpoints") or [])),
            failure_classification=None, provenance=provenance,
            identity=identity, artifacts=artifacts,
        )
