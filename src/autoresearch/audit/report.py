"""TrustAuditReport v1: schemas, self-checking writer, and loader.

Implements order 1 of ``research/trust_audit/DESIGN.md`` (§8, §11): the
report dataclasses, the frozen JSON Schema of record (§8.1 transcribed
verbatim), the fail-closed writer with the §8.3 self-check, and the loader
that re-validates and refuses on drift.

No-fabrication guarantee (§8.3, mechanically enforced):

1. The report dict is built without ``report_checksum``.
2. Every ``rates`` entry is recomputed from the recorded ``counts`` via the
   frozen :data:`RATE_SPECS` registry; any mismatch — or any rate that does
   not follow from recorded counts — raises
   :class:`AuditRefusal` ``RATE_NOT_DERIVED``. Rates are *derived*, never
   asserted: :meth:`MetricSection.complete` computes them, and the writer
   and loader independently recompute them, so a hand-edited report cannot
   survive.
3. The report is validated against the frozen JSON Schema.
4. ``report_checksum = content_checksum(report_without_checksum)`` (the
   stable canonical SHA-256 from ``domain/models.py``).
5. The loader re-validates the schema, recomputes the checksum, and refuses
   on drift (``CHECKSUM_MISMATCH``).

Fail-closed ground rules (§4.3 asymmetry):

* A metric section is either ``RUN_COMPLETE`` (per-item verdicts inside,
  artifact-side failures count against the audited system) or ``NOT_RUN``
  with a reason (harness-side failure; every declared rate is ``null`` and
  a NOT_RUN section can never carry a non-null rate).
* The schema is embedded as :data:`REPORT_SCHEMA` and validated by a
  minimal built-in validator covering exactly the keyword subset the v1
  schema uses. Deviation from §8.3 step 3 (recorded rationale): the
  ``jsonschema`` package is not a project dependency and adding one for a
  single frozen schema is not worth the cost; the validator below
  implements the same keywords (``type/const/enum/pattern/required/
  properties/additionalProperties/items/$ref/$defs/minimum/maximum/
  format: date-time``) the schema actually uses, and the schema-of-record
  files under ``research/trust_audit/schemas/v1/`` (§8.2, §13 Q9) remain
  the publication artifact.

Count-key contract for the metric modules (frozen with this harness
version; §9 semver): each rate is defined over named counts that the
metric auditors must record — see :data:`RATE_SPECS`. A missing count key
is a refusal (``COUNT_MISSING``), never a silent zero.
"""

from __future__ import annotations

import json
import re
import sys
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _package_metadata_version
from pathlib import Path
from typing import Any

from autoresearch.audit.contract import _PREFERRED_PAPER, FileRole, PaperArtifactBundle
from autoresearch.domain.models import content_checksum, utc_now

HARNESS_VERSION = "autoresearch-audit/0.1.0"
REPORT_SCHEMA_ID = "trust-audit-report/v1"
METRIC_NAMES = ("citations", "claims", "numbers", "overclaim", "disclosure", "reproducibility")


def _package_version() -> str:
    try:
        return _package_metadata_version("autoresearch")
    except PackageNotFoundError:
        return "unknown"


class AuditRefusal(RuntimeError):
    """Fail-closed refusal: the harness refuses rather than fabricates."""

    def __init__(self, code: str, detail: str) -> None:
        super().__init__(f"{code}: {detail}")
        self.code = code
        self.detail = detail


# ----------------------------------------------------------------------
# Rate derivation registry (§8.3 step 2 — the no-fabrication guarantee)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class RateSpec:
    """One pre-registered rate: sum(numerator counts) / sum(denominator counts)."""

    metric: str
    name: str
    numerator: tuple[str, ...]
    denominator: tuple[str, ...]


RATE_SPECS: tuple[RateSpec, ...] = (
    # §5.1 citation resolution
    RateSpec("citations", "citation_trust_rate", ("resolved_exact",), ("total_entries",)),
    RateSpec("citations", "unverifiable_rate", ("unresolvable",), ("total_entries",)),
    RateSpec("citations", "misattribution_rate", ("resolved_misattributed",), ("total_entries",)),
    # §5.2 claim-evidence support
    RateSpec("claims", "strong_support_rate", ("verbatim_verified", "artifact_traced"), ("claims_examined",)),
    RateSpec("claims", "citation_backed_rate", ("citation_backed",), ("claims_examined",)),
    # §5.3 numeric provenance
    RateSpec("numbers", "numeric_provenance_rate", ("traced_artifact", "traced_source", "structural"), ("numeric_tokens_examined",)),
    RateSpec("numbers", "recomputation_match_rate", ("recomputed_match",), ("recomputable",)),
    # §5.4 overclaim detection
    RateSpec("overclaim", "overclaim_pattern_rate", ("pattern_hits",), ("sentences_examined",)),
    # §5.5 failure-disclosure honesty
    RateSpec("disclosure", "disclosure_honesty_rate", ("disclosed_claims",), ("audited_completeness_claims",)),
    # §5.6 reproducibility
    RateSpec("reproducibility", "reproduction_rate", ("rerun_match",), ("commands_expected",)),
)


def _specs_for(metric: str) -> tuple[RateSpec, ...]:
    return tuple(spec for spec in RATE_SPECS if spec.metric == metric)


def _derive_rate(spec: RateSpec, counts: Mapping[str, Any]) -> float | None:
    """Derive one rate from recorded counts; a zero denominator yields None."""
    for key in (*spec.numerator, *spec.denominator):
        if key not in counts:
            raise AuditRefusal(
                "COUNT_MISSING",
                f"rate {spec.metric}.{spec.name} requires count '{key}'",
            )
    denominator = sum(int(counts[key]) for key in spec.denominator)
    if denominator <= 0:
        return None
    return sum(int(counts[key]) for key in spec.numerator) / denominator


# ----------------------------------------------------------------------
# Report dataclasses (§8.1, mirrored field-for-field)
# ----------------------------------------------------------------------


@dataclass(frozen=True)
class BundleFileInfo:
    path: str
    role: str
    sha256: str
    bytes: int


@dataclass(frozen=True)
class TextExtractionInfo:
    mode: str  # latex_source | markdown_source | pdf_text
    quality: Mapping[str, int | float | str | bool] = field(default_factory=dict)


@dataclass(frozen=True)
class ReportBundleInfo:
    system_id: str
    adapter_name: str
    adapter_version: str
    root: str
    files: tuple[BundleFileInfo, ...] = ()
    excluded_files: tuple[str, ...] = ()
    text_extraction: TextExtractionInfo = field(default_factory=lambda: TextExtractionInfo("markdown_source"))
    ground_truth_tier: str = "none"


@dataclass(frozen=True)
class FetchCacheEntry:
    source: str
    key: str
    raw_checksum: str
    fetched_at: str
    cache_hit: bool


@dataclass(frozen=True)
class FetchCacheManifest:
    entries: tuple[FetchCacheEntry, ...] = ()
    checksum: str = ""


@dataclass(frozen=True)
class ReportInput:
    bundle: ReportBundleInfo
    fetch_cache_manifest: FetchCacheManifest | None = None


@dataclass(frozen=True)
class HarnessInfo:
    harness_version: str = HARNESS_VERSION
    harness_commit: str = "unknown"
    freeze_checksum: str | None = None
    frozen: bool = False
    python_version: str = field(default_factory=lambda: f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}")
    package_version: str = field(default_factory=_package_version)


@dataclass(frozen=True)
class LlmUsageRecord:
    metric: str
    purpose: str
    provider: str
    model: str
    prompt_version: str
    calls: int = 0
    sole_basis: bool = False  # must stay False: LLM passes are never the sole basis (§1)


@dataclass(frozen=True)
class FailureRecord:
    metric: str
    code: str
    detail: str


@dataclass(frozen=True)
class CoverageInfo:
    items_examined: int = 0
    items_skipped: int = 0
    skip_reasons: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class MetricSection:
    """One metric's envelope (§8.1 metricSection). Rates are derived data."""

    metric: str
    status: str  # RUN_COMPLETE | NOT_RUN
    llm_assisted: bool = False
    counts: Mapping[str, int] = field(default_factory=dict)
    rates: Mapping[str, float | None] = field(default_factory=dict)
    coverage: CoverageInfo = field(default_factory=CoverageInfo)
    details: tuple[Mapping[str, Any], ...] = ()
    not_run_reason: str | None = None

    @classmethod
    def complete(
        cls,
        metric: str,
        *,
        counts: Mapping[str, int],
        coverage: CoverageInfo | None = None,
        details: Sequence[Mapping[str, Any]] = (),
        llm_assisted: bool = False,
    ) -> MetricSection:
        """Build a RUN_COMPLETE section; rates are derived from counts."""
        for key, value in counts.items():
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AuditRefusal("COUNT_INVALID", f"count '{key}' of metric '{metric}' must be a non-negative integer")
        rates = {spec.name: _derive_rate(spec, counts) for spec in _specs_for(metric)}
        return cls(
            metric=metric,
            status="RUN_COMPLETE",
            llm_assisted=llm_assisted,
            counts=dict(counts),
            rates=rates,
            coverage=coverage or CoverageInfo(),
            details=tuple(details),
        )

    @classmethod
    def not_run(cls, metric: str, reason: str, *, llm_assisted: bool = False) -> MetricSection:
        """Build a NOT_RUN section: rates null, never a pass (§4.3)."""
        rates = {spec.name: None for spec in _specs_for(metric)}
        return cls(
            metric=metric,
            status="NOT_RUN",
            llm_assisted=llm_assisted,
            counts={},
            rates=rates,
            not_run_reason=reason,
        )


@dataclass(frozen=True)
class TrustAuditReport:
    input: ReportInput
    metrics: Mapping[str, MetricSection] = field(default_factory=dict)
    report_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    generated_at: str = field(default_factory=utc_now)
    harness: HarnessInfo = field(default_factory=HarnessInfo)
    llm_usage: tuple[LlmUsageRecord, ...] = ()
    failures: tuple[FailureRecord, ...] = ()
    report_checksum: str | None = None

    def to_dict(self, *, include_checksum: bool = False) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema": REPORT_SCHEMA_ID,
            "report_id": self.report_id,
            "generated_at": self.generated_at,
            "harness": {
                "harness_version": self.harness.harness_version,
                "harness_commit": self.harness.harness_commit,
                "freeze_checksum": self.harness.freeze_checksum,
                "frozen": self.harness.frozen,
                "python_version": self.harness.python_version,
                "package_version": self.harness.package_version,
            },
            "input": {
                "bundle": {
                    "system_id": self.input.bundle.system_id,
                    "adapter_name": self.input.bundle.adapter_name,
                    "adapter_version": self.input.bundle.adapter_version,
                    "root": self.input.bundle.root,
                    "files": [
                        {"path": f.path, "role": f.role, "sha256": f.sha256, "bytes": f.bytes}
                        for f in self.input.bundle.files
                    ],
                    "excluded_files": list(self.input.bundle.excluded_files),
                    "text_extraction": {
                        "mode": self.input.bundle.text_extraction.mode,
                        "quality": dict(self.input.bundle.text_extraction.quality),
                    },
                    "ground_truth_tier": self.input.bundle.ground_truth_tier,
                },
                "fetch_cache_manifest": (
                    None
                    if self.input.fetch_cache_manifest is None
                    else {
                        "entries": [
                            {
                                "source": e.source,
                                "key": e.key,
                                "raw_checksum": e.raw_checksum,
                                "fetched_at": e.fetched_at,
                                "cache_hit": e.cache_hit,
                            }
                            for e in self.input.fetch_cache_manifest.entries
                        ],
                        "checksum": self.input.fetch_cache_manifest.checksum,
                    }
                ),
            },
            "metrics": {
                name: {
                    "status": section.status,
                    "not_run_reason": section.not_run_reason,
                    "llm_assisted": section.llm_assisted,
                    "counts": dict(section.counts),
                    "rates": dict(section.rates),
                    "coverage": {
                        "items_examined": section.coverage.items_examined,
                        "items_skipped": section.coverage.items_skipped,
                        "skip_reasons": dict(section.coverage.skip_reasons),
                    },
                    "details": [dict(detail) for detail in section.details],
                }
                for name, section in self.metrics.items()
            },
            "llm_usage": [
                {
                    "metric": record.metric,
                    "purpose": record.purpose,
                    "provider": record.provider,
                    "model": record.model,
                    "prompt_version": record.prompt_version,
                    "calls": record.calls,
                    "sole_basis": record.sole_basis,
                }
                for record in self.llm_usage
            ],
            "failures": [
                {"metric": record.metric, "code": record.code, "detail": record.detail}
                for record in self.failures
            ],
        }
        if include_checksum and self.report_checksum:
            payload["report_checksum"] = self.report_checksum
        return payload

    def write(self, path: str | Path) -> str:
        return write_report(self, path)

    def checksum(self) -> str:
        """Self-checked canonical checksum of the report body."""
        payload = self.to_dict(include_checksum=False)
        _self_check_rates(payload)
        checksum = content_checksum(payload)
        payload["report_checksum"] = checksum
        _validate(payload, REPORT_SCHEMA)
        return checksum


# ----------------------------------------------------------------------
# Frozen JSON Schema of record (§8.1, transcribed verbatim)
# ----------------------------------------------------------------------


REPORT_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "$id": "autoresearch/trust-audit-report/v1",
    "type": "object",
    "additionalProperties": False,
    "required": ["schema", "report_id", "generated_at", "harness", "input",
                 "metrics", "llm_usage", "failures", "report_checksum"],
    "properties": {
        "schema": {"const": "trust-audit-report/v1"},
        "report_id": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
        "generated_at": {"type": "string", "format": "date-time"},
        "harness": {
            "type": "object", "additionalProperties": False,
            "required": ["harness_version", "harness_commit", "frozen", "python_version"],
            "properties": {
                "harness_version": {"type": "string"},
                "harness_commit": {"type": "string"},
                "freeze_checksum": {"type": ["string", "null"]},
                "frozen": {"type": "boolean"},
                "python_version": {"type": "string"},
                "package_version": {"type": "string"},
            },
        },
        "input": {
            "type": "object", "additionalProperties": False,
            "required": ["bundle"],
            "properties": {
                "bundle": {
                    "type": "object", "additionalProperties": False,
                    "required": ["system_id", "adapter_name", "adapter_version", "root",
                                 "files", "excluded_files", "text_extraction"],
                    "properties": {
                        "system_id": {"type": "string"},
                        "adapter_name": {"type": "string"},
                        "adapter_version": {"type": "string"},
                        "root": {"type": "string"},
                        "files": {"type": "array", "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["path", "role", "sha256", "bytes"],
                            "properties": {
                                "path": {"type": "string"},
                                "role": {"enum": ["paper_source", "paper_pdf", "bibliography",
                                                  "run_log", "artifact", "other"]},
                                "sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                                "bytes": {"type": "integer", "minimum": 0},
                            },
                        }},
                        "excluded_files": {"type": "array", "items": {"type": "string"}},
                        "text_extraction": {
                            "type": "object", "additionalProperties": False,
                            "required": ["mode", "quality"],
                            "properties": {
                                "mode": {"enum": ["latex_source", "markdown_source", "pdf_text"]},
                                "quality": {"type": "object", "additionalProperties": {
                                    "type": ["integer", "number", "string", "boolean"]}},
                            },
                        },
                        "ground_truth_tier": {"enum": ["structured", "log_derived", "none"]},
                    },
                },
                "fetch_cache_manifest": {
                    "type": ["object", "null"], "additionalProperties": False,
                    "required": ["entries", "checksum"],
                    "properties": {
                        "entries": {"type": "array", "items": {
                            "type": "object", "additionalProperties": False,
                            "required": ["source", "key", "raw_checksum", "fetched_at", "cache_hit"],
                            "properties": {
                                "source": {"type": "string"}, "key": {"type": "string"},
                                "raw_checksum": {"type": "string"},
                                "fetched_at": {"type": "string", "format": "date-time"},
                                "cache_hit": {"type": "boolean"},
                            },
                        }},
                        "checksum": {"type": "string"},
                    },
                },
            },
        },
        "metrics": {
            "type": "object", "additionalProperties": False,
            "required": ["citations", "claims", "numbers", "overclaim",
                         "disclosure", "reproducibility"],
            "properties": {
                "citations": {"$ref": "#/$defs/metricSection"},
                "claims": {"$ref": "#/$defs/metricSection"},
                "numbers": {"$ref": "#/$defs/metricSection"},
                "overclaim": {"$ref": "#/$defs/metricSection"},
                "disclosure": {"$ref": "#/$defs/metricSection"},
                "reproducibility": {"$ref": "#/$defs/metricSection"},
            },
        },
        "llm_usage": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["metric", "purpose", "provider", "model", "prompt_version",
                         "calls", "sole_basis"],
            "properties": {
                "metric": {"type": "string"}, "purpose": {"type": "string"},
                "provider": {"type": "string"}, "model": {"type": "string"},
                "prompt_version": {"type": "string"}, "calls": {"type": "integer", "minimum": 0},
                "sole_basis": {"const": False},
            },
        }},
        "failures": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["metric", "code", "detail"],
            "properties": {"metric": {"type": "string"}, "code": {"type": "string"},
                           "detail": {"type": "string"}},
        }},
        "report_checksum": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
    },
    "$defs": {
        "metricSection": {
            "type": "object", "additionalProperties": False,
            "required": ["status", "llm_assisted", "counts", "rates", "coverage", "details"],
            "properties": {
                "status": {"enum": ["RUN_COMPLETE", "NOT_RUN"]},
                "not_run_reason": {"type": ["string", "null"]},
                "llm_assisted": {"type": "boolean"},
                "counts": {"type": "object",
                           "additionalProperties": {"type": "integer", "minimum": 0}},
                "rates": {"type": "object",
                          "additionalProperties": {"type": ["number", "null"],
                                                   "minimum": 0, "maximum": 1}},
                "coverage": {"type": "object", "additionalProperties": False,
                             "required": ["items_examined", "items_skipped"],
                             "properties": {
                                 "items_examined": {"type": "integer", "minimum": 0},
                                 "items_skipped": {"type": "integer", "minimum": 0},
                                 "skip_reasons": {"type": "object",
                                                  "additionalProperties": {"type": "integer"}},
                             }},
                "details": {"type": "array", "items": {"type": "object"}},
            },
        },
    },
}


# ----------------------------------------------------------------------
# Minimal JSON-Schema validator (keyword subset of the v1 schema)
# ----------------------------------------------------------------------


def _validate(instance: Any, schema: Mapping[str, Any], path: str = "$") -> None:
    """Validate ``instance`` against the frozen schema subset; refuse on any violation."""
    ref = schema.get("$ref")
    if ref is not None:
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            raise AuditRefusal("SCHEMA_INVALID", f"{path}: unsupported $ref {ref!r}")
        target = REPORT_SCHEMA["$defs"].get(ref[len("#/$defs/"):])
        if not isinstance(target, dict):
            raise AuditRefusal("SCHEMA_INVALID", f"{path}: unresolvable $ref {ref!r}")
        _validate(instance, target, path)
        return
    if "const" in schema and instance != schema["const"]:
        raise AuditRefusal("SCHEMA_INVALID", f"{path}: expected const {schema['const']!r}, got {instance!r}")
    if "enum" in schema and instance not in schema["enum"]:
        raise AuditRefusal("SCHEMA_INVALID", f"{path}: {instance!r} not in {schema['enum']!r}")
    expected = schema.get("type")
    if expected is not None:
        allowed = expected if isinstance(expected, list) else [expected]
        if not any(_type_matches(instance, name) for name in allowed):
            raise AuditRefusal("SCHEMA_INVALID", f"{path}: expected type {expected!r}, got {type(instance).__name__}")
    if isinstance(instance, str):
        pattern = schema.get("pattern")
        if pattern is not None and re.search(pattern, instance) is None:
            raise AuditRefusal("SCHEMA_INVALID", f"{path}: {instance!r} does not match {pattern!r}")
        if schema.get("format") == "date-time" and not _is_iso_datetime(instance):
            raise AuditRefusal("SCHEMA_INVALID", f"{path}: {instance!r} is not an ISO-8601 date-time")
    if isinstance(instance, (int, float)) and not isinstance(instance, bool):
        if "minimum" in schema and instance < schema["minimum"]:
            raise AuditRefusal("SCHEMA_INVALID", f"{path}: {instance} < minimum {schema['minimum']}")
        if "maximum" in schema and instance > schema["maximum"]:
            raise AuditRefusal("SCHEMA_INVALID", f"{path}: {instance} > maximum {schema['maximum']}")
    if isinstance(instance, dict):
        for key in schema.get("required", []):
            if key not in instance:
                raise AuditRefusal("SCHEMA_INVALID", f"{path}: missing required property {key!r}")
        properties = schema.get("properties", {})
        additional = schema.get("additionalProperties", True)
        for key, value in instance.items():
            if key in properties:
                _validate(value, properties[key], f"{path}.{key}")
            elif additional is False:
                raise AuditRefusal("SCHEMA_INVALID", f"{path}: unexpected property {key!r}")
            elif isinstance(additional, dict):
                _validate(value, additional, f"{path}.{key}")
    if isinstance(instance, list) and "items" in schema:
        for index, item in enumerate(instance):
            _validate(item, schema["items"], f"{path}[{index}]")


def _type_matches(instance: Any, expected: str) -> bool:
    if expected == "object":
        return isinstance(instance, dict)
    if expected == "array":
        return isinstance(instance, list)
    if expected == "string":
        return isinstance(instance, str)
    if expected == "boolean":
        return isinstance(instance, bool)
    if expected == "null":
        return instance is None
    if expected == "integer":
        return isinstance(instance, int) and not isinstance(instance, bool)
    if expected == "number":
        return isinstance(instance, (int, float)) and not isinstance(instance, bool)
    return False


def _is_iso_datetime(value: str) -> bool:
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00") if value.endswith("Z") else value)
        return True
    except ValueError:
        return False


# ----------------------------------------------------------------------
# §8.3 self-check: every rate must follow from recorded counts
# ----------------------------------------------------------------------


def _self_check_rates(payload: Mapping[str, Any]) -> None:
    metrics = payload.get("metrics")
    if not isinstance(metrics, dict):
        raise AuditRefusal("SCHEMA_INVALID", "$: 'metrics' is not an object")
    for metric, section in metrics.items():
        if not isinstance(section, dict):
            raise AuditRefusal("SCHEMA_INVALID", f"$.metrics.{metric} is not an object")
        rates = section.get("rates")
        counts = section.get("counts")
        if not isinstance(rates, dict) or not isinstance(counts, dict):
            raise AuditRefusal("SCHEMA_INVALID", f"$.metrics.{metric}: rates/counts must be objects")
        specs = _specs_for(metric)
        declared = {spec.name for spec in specs}
        unknown = set(rates) - declared
        if unknown:
            raise AuditRefusal(
                "RATE_NOT_DERIVED",
                f"metric '{metric}' carries unregistered rates {sorted(unknown)}; only {sorted(declared)} are derivable from counts",
            )
        if section.get("status") == "RUN_COMPLETE":
            for spec in specs:
                if spec.name not in rates:
                    raise AuditRefusal(
                        "RATE_NOT_DECLARED",
                        f"metric '{metric}' is RUN_COMPLETE but rate '{spec.name}' is missing",
                    )
                derived = _derive_rate(spec, counts)
                stored = rates[spec.name]
                if stored != derived:
                    raise AuditRefusal(
                        "RATE_NOT_DERIVED",
                        f"rate {metric}.{spec.name}: stored {stored!r} != derived {derived!r} from counts {dict(counts)!r}",
                    )
        else:
            for name, value in rates.items():
                if value is not None:
                    raise AuditRefusal(
                        "NOT_RUN_WITH_RATE",
                        f"metric '{metric}' is not RUN_COMPLETE but rate '{name}' is {value!r}; NOT_RUN rates must be null (§4.3)",
                    )


# ----------------------------------------------------------------------
# Writer and loader
# ----------------------------------------------------------------------


def write_report(report: TrustAuditReport, path: str | Path) -> str:
    """Build, self-check, validate, checksum, and write one report (§8.3).

    Order per §8.3: (1) build the dict without ``report_checksum``,
    (2) self-check that every rate follows from counts, (3) validate
    against the frozen schema, (4) compute the checksum over the
    checksum-less body. Steps 3–4 are interleaved (validate the payload
    with the checksum key present, compute the checksum over the body
    without it) because the frozen schema marks ``report_checksum``
    required — the guarantee is identical either way.
    """
    payload = report.to_dict(include_checksum=False)
    _self_check_rates(payload)
    checksum = content_checksum(payload)
    payload["report_checksum"] = checksum
    _validate(payload, REPORT_SCHEMA)
    try:
        text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
    except TypeError as exc:
        raise AuditRefusal("DETAIL_NOT_SERIALIZABLE", str(exc)) from exc
    Path(path).write_text(text, encoding="utf-8")
    return checksum


def load_report(path: str | Path) -> TrustAuditReport:
    """Load, validate, checksum-verify, and self-check one report (§8.3 step 5)."""
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise AuditRefusal("REPORT_UNREADABLE", str(exc)) from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise AuditRefusal("NOT_JSON", f"{path}: {exc}") from exc
    if not isinstance(data, dict):
        raise AuditRefusal("NOT_JSON", f"{path}: top level is not an object")
    _validate(data, REPORT_SCHEMA)
    stored = data.pop("report_checksum")
    recomputed = content_checksum(data)
    if stored != recomputed:
        raise AuditRefusal(
            "CHECKSUM_MISMATCH",
            f"report checksum drift: stored {stored} != recomputed {recomputed}",
        )
    _self_check_rates(data)
    return _report_from_dict(data, stored)


def _report_from_dict(data: Mapping[str, Any], checksum: str) -> TrustAuditReport:
    bundle_data = data["input"]["bundle"]
    extraction = bundle_data["text_extraction"]
    bundle = ReportBundleInfo(
        system_id=bundle_data["system_id"],
        adapter_name=bundle_data["adapter_name"],
        adapter_version=bundle_data["adapter_version"],
        root=bundle_data["root"],
        files=tuple(
            BundleFileInfo(path=f["path"], role=f["role"], sha256=f["sha256"], bytes=f["bytes"])
            for f in bundle_data["files"]
        ),
        excluded_files=tuple(bundle_data["excluded_files"]),
        text_extraction=TextExtractionInfo(extraction["mode"], dict(extraction["quality"])),
        ground_truth_tier=bundle_data.get("ground_truth_tier", "none"),
    )
    manifest_data = data["input"].get("fetch_cache_manifest")
    manifest = None
    if isinstance(manifest_data, dict):
        manifest = FetchCacheManifest(
            entries=tuple(
                FetchCacheEntry(
                    source=e["source"], key=e["key"], raw_checksum=e["raw_checksum"],
                    fetched_at=e["fetched_at"], cache_hit=e["cache_hit"],
                )
                for e in manifest_data["entries"]
            ),
            checksum=manifest_data["checksum"],
        )
    harness_data = data["harness"]
    harness = HarnessInfo(
        harness_version=harness_data["harness_version"],
        harness_commit=harness_data["harness_commit"],
        freeze_checksum=harness_data.get("freeze_checksum"),
        frozen=harness_data["frozen"],
        python_version=harness_data["python_version"],
        package_version=harness_data.get("package_version", "unknown"),
    )
    metrics = {}
    for name, section in data["metrics"].items():
        coverage_data = section["coverage"]
        metrics[name] = MetricSection(
            metric=name,
            status=section["status"],
            llm_assisted=section["llm_assisted"],
            counts=dict(section["counts"]),
            rates=dict(section["rates"]),
            coverage=CoverageInfo(
                items_examined=coverage_data["items_examined"],
                items_skipped=coverage_data["items_skipped"],
                skip_reasons=dict(coverage_data.get("skip_reasons", {})),
            ),
            details=tuple(dict(detail) for detail in section["details"]),
            not_run_reason=section.get("not_run_reason"),
        )
    return TrustAuditReport(
        input=ReportInput(bundle=bundle, fetch_cache_manifest=manifest),
        metrics=metrics,
        report_id=data["report_id"],
        generated_at=data["generated_at"],
        harness=harness,
        llm_usage=tuple(
            LlmUsageRecord(
                metric=r["metric"], purpose=r["purpose"], provider=r["provider"],
                model=r["model"], prompt_version=r["prompt_version"],
                calls=r["calls"], sole_basis=r["sole_basis"],
            )
            for r in data["llm_usage"]
        ),
        failures=tuple(
            FailureRecord(metric=r["metric"], code=r["code"], detail=r["detail"])
            for r in data["failures"]
        ),
        report_checksum=checksum,
    )


# ----------------------------------------------------------------------
# Bundle → report-input conversion (integration helper for runner.py)
# ----------------------------------------------------------------------


def bundle_to_report_input(bundle: PaperArtifactBundle) -> ReportInput:
    """Convert a ``contract.PaperArtifactBundle`` into the report input block.

    Text-extraction mode is inferred from the bundle's file roles (source
    preferred over PDF, §4.1); quality records section/character counts so
    extraction weakness is disclosed (risk R4).
    """
    sources = [f for f in bundle.files if f.role == FileRole.PAPER_SOURCE]
    mode = "pdf_text"
    if sources:
        mode = "markdown_source" if _preferred_suffix(sources) == ".md" else "latex_source"
    elif not any(f.role == FileRole.PAPER_PDF for f in bundle.files):
        # No paper at all: the schema enum has no "none" mode; the runner
        # must record the failure explicitly (metric sections NOT_RUN).
        mode = "markdown_source"
    files = tuple(
        BundleFileInfo(path=f.path, role=f.role.value, sha256=f.sha256, bytes=f.bytes)
        for f in bundle.files
    )
    quality: dict[str, int | float | str | bool] = {
        "sections": len(bundle.paper_text),
        "characters": sum(len(section.text) for section in bundle.paper_text),
        "source_files": len(sources),
    }
    return ReportInput(
        bundle=ReportBundleInfo(
            system_id=bundle.system_id,
            adapter_name=bundle.adapter_name,
            adapter_version=bundle.adapter_version,
            root=bundle.root,
            files=files,
            excluded_files=tuple(bundle.excluded_files),
            text_extraction=TextExtractionInfo(mode, quality),
            ground_truth_tier=bundle.ground_truth_tier,
        ),
        fetch_cache_manifest=None,
    )


def _preferred_suffix(sources: list[Any]) -> str:
    for name in _PREFERRED_PAPER:
        for record in sources:
            if record.path == name:
                return ".md" if name.endswith(".md") else ".tex"
    for record in sources:
        if record.path.lower().endswith(".md"):
            return ".md"
    return ".tex"
