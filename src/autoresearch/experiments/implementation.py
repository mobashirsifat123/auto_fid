"""Purpose: Generates and statically verifies the experiment implementation.

Closes the methodology→code gap (mechanism adapted from AI-Researcher's
agentic implementation stage, NeurIPS 2025; and Co-Scientist's bounded
generate→critique→refine loop for the repair cycle): the frozen methodology
and exact spec become a working ``scripts/run_experiment.py`` through the
ModelRouter, then a DETERMINISTIC verifier checks that the intended mechanism
is present and that the measurement contract (ingestion.py) is emitted.

Fidelity levels (labelled, never conflated):
- ``STATIC_VERIFIED``  — deterministic checks pass (this module).
- ``SMOKE`` / ``EXPLORATORY`` / ``CONFIRMATORY`` — attach at execution time.

A program that merely runs can still implement the wrong method: the
verifier's mechanism-coverage check maps methodology components to code
symbols and reports coverage honestly; ``MECHANISM_PARTIAL`` stays visible
instead of being rounded up to verified. No hidden token/time caps are
applied to generation; the only bound is the repair-loop attempt count.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

from autoresearch.domain.models import ArtifactProvenance, ExperimentSpec, content_checksum
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.router import ModelRouter
from autoresearch.storage.repository import ResearchRepository

logger = logging.getLogger(__name__)

IMPLEMENTATION_PROMPT_VERSION = "implementation-generator-v1"
MAX_REPAIR_ATTEMPTS = 3

_FORBIDDEN_IMPORTS = {
    "socket", "urllib", "http", "requests", "subprocess", "ftplib", "telnetlib",
    "smtplib", "asyncio", "multiprocessing", "ctypes", "pickle", "shelve",
}

_STOPWORDS = {
    "the", "a", "an", "of", "and", "or", "to", "in", "on", "for", "with", "without",
    "by", "as", "is", "are", "can", "does", "do", "not", "how", "what", "which",
    "that", "this", "these", "those", "it", "its", "from", "at", "be", "using",
    "into", "than", "then", "when", "where", "who", "why", "will", "would", "there",
    "then", "each", "step", "steps", "value", "values", "compute", "calculate",
    "using", "used", "use", "given", "over", "between", "such", "must", "should",
}

IDENTITY_FIELDS = ("method_id", "repository_commit", "implementation_hash", "config_hash", "metric_version")


def identity_for_spec(spec: ExperimentSpec, freeze) -> dict[str, str]:
    """Deterministic document-identity values the script must embed.

    Spec-pinned identity (candidate/dataset/split) is asserted by the
    ingester from the worker's checksum binding; these five values describe
    the implementation itself.
    """
    freeze_digest = str(getattr(freeze, "checksum", "") or "") or "unfrozen"
    return {
        "method_id": re.sub(r"[^a-z0-9]+", "-", str(spec.proposed_change).lower()).strip("-")[:60] or "candidate-method",
        "repository_commit": str(spec.baseline_commit or "none"),
        "implementation_hash": f"{freeze_digest[:16]}",
        "config_hash": f"{content_checksum(str(spec.id))[:16]}",
        "metric_version": "-".join(str(m) for m in spec.metrics) + "-v1",
    }


def mechanism_components(methodology: dict[str, Any] | None, spec: ExperimentSpec) -> list[str]:
    """Extract the methodology's core component names (deterministic).

    Serves as the method→code map the verifier checks coverage against.
    """
    block = (methodology or {}).get("methodology", methodology or {})
    text = " ".join(
        str(block.get(key) or "") for key in ("algorithm_outline", "procedure", "causal_mechanism")
    ) or f"{spec.proposed_change} {spec.hypothesis}"
    tokens = [
        token for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{4,}", text.lower())
        if token not in _STOPWORDS
    ]
    ordered: list[str] = []
    for token in tokens:
        if token not in ordered:
            ordered.append(token)
    return ordered[:6]


@dataclass
class VerificationReport:
    level: str  # STATIC_VERIFIED | STATIC_FAILED
    checks: dict[str, bool] = field(default_factory=dict)
    mechanism_coverage: float = 0.0
    matched_components: list[str] = field(default_factory=list)
    missing_components: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "level": self.level,
            "checks": self.checks,
            "mechanism_coverage": round(self.mechanism_coverage, 3),
            "matched_components": self.matched_components,
            "missing_components": self.missing_components,
            "reasons": self.reasons,
        }


class ImplementationVerifier:
    """Deterministic static verification. No model calls, no execution."""

    def verify(self, source: str, spec: ExperimentSpec, methodology: dict[str, Any] | None, identity: dict[str, str]) -> VerificationReport:
        report = VerificationReport(level="STATIC_FAILED")
        reasons: list[str] = []

        # 1. Syntax
        try:
            tree = ast.parse(source)
            report.checks["syntax"] = True
        except SyntaxError as exc:
            report.checks["syntax"] = False
            reasons.append(f"SYNTAX_ERROR: line {exc.lineno}: {exc.msg}")
            report.reasons = reasons
            return report

        # 2. Forbidden imports (worker runs with network disabled; shell escapes are policy violations)
        imported: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        bad = imported & _FORBIDDEN_IMPORTS
        report.checks["no_forbidden_imports"] = not bad
        if bad:
            reasons.append(f"FORBIDDEN_IMPORTS: {sorted(bad)} (worker policy: stdlib-only, no network/shell/process escapes)")

        # 3. Identity block with the exact expected values
        expected_identity = {key: str(value) for key, value in identity.items()}
        identity_node = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "IDENTITY" for t in node.targets):
                identity_node = node.value
        identity_ok = False
        if isinstance(identity_node, ast.Dict):
            literal: dict[str, str] = {}
            for key_node, value_node in zip(identity_node.keys, identity_node.values):
                if isinstance(key_node, ast.Constant) and isinstance(value_node, ast.Constant):
                    literal[str(key_node.value)] = str(value_node.value)
            identity_ok = all(literal.get(key) == expected_identity.get(key) for key in IDENTITY_FIELDS)
        report.checks["identity_block"] = identity_ok
        if not identity_ok:
            reasons.append(
                "IDENTITY_BLOCK_MISSING: define module-level IDENTITY = {" + ", ".join(
                    f'"{k}": "{v}"' for k, v in expected_identity.items()
                ) + "}"
            )

        # 4. Declared seeds exercised
        seed_constants = {
            node.value for node in ast.walk(tree)
            if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value in set(spec.random_seeds)
        }
        seeds_ok = set(spec.random_seeds) <= seed_constants
        report.checks["declared_seeds_present"] = seeds_ok
        if not seeds_ok:
            reasons.append(f"SEEDS_MISSING: script must exercise the declared seeds {list(spec.random_seeds)}")

        # 5. Measurement contract: json.dumps print carrying a "seeds" document
        contract_ok = False
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "dumps":
                module = node.func.value
                if isinstance(module, ast.Name) and module.id == "json" and node.args:
                    arg = node.args[0]
                    if isinstance(arg, ast.Dict) and any(
                        isinstance(k, ast.Constant) and k.value == "seeds" for k in arg.keys
                    ):
                        contract_ok = True
                        break
                # also accept json.dumps(VAR) where VAR is built with a "seeds" key
                if isinstance(node.args[0], ast.Name):
                    var_name = node.args[0].id
                    for other in ast.walk(tree):
                        if isinstance(other, ast.Assign) and isinstance(other.value, ast.Dict):
                            if any(isinstance(t, ast.Name) and t.id == var_name for t in other.targets) and any(
                                isinstance(k, ast.Constant) and k.value == "seeds" for k in other.value.keys
                            ):
                                contract_ok = True
                                break
                if contract_ok:
                    break
        report.checks["measurement_contract"] = contract_ok
        if not contract_ok:
            reasons.append(
                'MEASUREMENT_CONTRACT_MISSING: end with print(json.dumps({"identity": IDENTITY, "seeds": seed_rows, "artifacts": artifact_rows}))'
            )

        # 6. Required artifact types written
        required = [str(item) for item in (getattr(spec, "required_artifacts", ()) or ())]
        missing_artifacts = [kind for kind in required if f"{kind}" not in source]
        report.checks["required_artifacts_named"] = not missing_artifacts
        if missing_artifacts:
            reasons.append(f"ARTIFACT_TYPES_MISSING: {missing_artifacts}")

        # 7. Mechanism coverage (methodology components → code symbols)
        components = mechanism_components(methodology, spec)
        defined = {
            node.name.lower() for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        }
        assigned = {
            t.id.lower() for node in tree.body if isinstance(node, ast.Assign)
            for t in node.targets if isinstance(t, ast.Name)
        }
        lowered = source.lower()
        matched, missing = [], []
        for component in components:
            symbol = component.replace("-", "_")
            if any(symbol in name or name in symbol for name in defined | assigned) or symbol in lowered:
                matched.append(component)
            else:
                missing.append(component)
        coverage = len(matched) / len(components) if components else 1.0
        report.mechanism_coverage = coverage
        report.matched_components = matched
        report.missing_components = missing
        report.checks["mechanism_present"] = coverage >= 0.75
        if coverage < 0.75:
            reasons.append(
                f"MECHANISM_PARTIAL: coverage {coverage:.0%}; missing components {missing} — "
                "implement each methodology component as a named function"
            )

        report.reasons = reasons
        report.level = "STATIC_VERIFIED" if all(report.checks.values()) else "STATIC_FAILED"
        return report


class ImplementationGenerator:
    """Generates the experiment implementation via ModelRouter with a bounded
    verify→repair loop (Co-Scientist generate→critique→refine, adapted)."""

    def __init__(self, repository: ResearchRepository, router: ModelRouter) -> None:
        self.repository = repository
        self.router = router
        self.verifier = ImplementationVerifier()

    async def implement(
        self,
        spec: ExperimentSpec,
        *,
        methodology: dict[str, Any] | None = None,
        findings: list[dict[str, Any]] | None = None,
        freeze=None,
    ) -> dict[str, Any]:
        script_path = Path(str(spec.repository)) / "scripts" / "run_experiment.py"
        script_path.parent.mkdir(parents=True, exist_ok=True)
        identity = identity_for_spec(spec, freeze)

        if script_path.is_file() and script_path.stat().st_size > 0:
            report = self.verifier.verify(script_path.read_text(encoding="utf-8"), spec, methodology, identity)
            return self._record(spec, script_path, report, attempts=0, status="reused")

        reasons: list[str] = []
        report = VerificationReport(level="STATIC_FAILED")
        for attempt in range(1, MAX_REPAIR_ATTEMPTS + 1):
            source = await self._generate_source(spec, methodology, findings, identity, previous_failures=reasons)
            if not source.strip():
                reasons.append("EMPTY_GENERATION: model returned no code")
                continue
            script_path.write_text(source, encoding="utf-8")
            report = self.verifier.verify(source, spec, methodology, identity)
            if report.level == "STATIC_VERIFIED":
                return self._record(spec, script_path, report, attempts=attempt, status="generated")
            reasons = list(report.reasons)
            logger.info("Implementation attempt %d failed verification: %s", attempt, reasons)
        return self._record(spec, script_path, report, attempts=MAX_REPAIR_ATTEMPTS, status="failed", extra_reasons=reasons)

    # ------------------------------------------------------------------

    def _record(
        self,
        spec: ExperimentSpec,
        script_path: Path,
        report: VerificationReport,
        *,
        attempts: int,
        status: str,
        extra_reasons: list[str] | None = None,
    ) -> dict[str, Any]:
        """Persist the implementation + verification report as checksummed
        artifacts and return the inspectable record."""
        from autoresearch.domain.models import ArtifactReference

        reasons = list(extra_reasons or []) if report.level != "STATIC_VERIFIED" else []
        source = script_path.read_text(encoding="utf-8") if script_path.is_file() else ""
        verification_payload = {
            "spec_id": spec.id,
            "status": status,
            "attempts": attempts,
            **report.as_dict(),
            "reasons": reasons or report.reasons,
        }
        root = self.repository.artifact_root / spec.project_id / spec.run_id
        root.mkdir(parents=True, exist_ok=True)
        verification_path = root / f"implementation-verification-{spec.id[:8]}.json"
        verification_path.write_text(json.dumps(verification_payload, indent=2), encoding="utf-8")
        if source:
            self.repository.save_artifact(ArtifactReference(
                uuid4().hex, project_id=spec.project_id, run_id=spec.run_id,
                artifact_type="experiment_implementation", path=str(script_path),
                source_evidence=(spec.id,), provider="model", model="implementation-generator",
                prompt_version=IMPLEMENTATION_PROMPT_VERSION,
                checksum=content_checksum(source),
            ))
        self.repository.save_artifact(ArtifactReference(
            uuid4().hex, project_id=spec.project_id, run_id=spec.run_id,
            artifact_type="implementation_verification", path=str(verification_path),
            source_evidence=(spec.id,), provider="deterministic", model="implementation-verifier",
            prompt_version=IMPLEMENTATION_PROMPT_VERSION,
            checksum=content_checksum(verification_path.read_text(encoding="utf-8")),
        ))
        return {
            "spec_id": spec.id,
            "candidate_id": spec.candidate_id,
            "script_path": str(script_path),
            "status": status,
            "verification": verification_payload,
            "fidelity": "STATIC_VERIFIED" if report.level == "STATIC_VERIFIED" else "UNVERIFIED",
        }

    async def _generate_source(
        self,
        spec: ExperimentSpec,
        methodology: dict[str, Any] | None,
        findings: list[dict[str, Any]] | None,
        identity: dict[str, str],
        *,
        previous_failures: list[str],
    ) -> str:
        block = (methodology or {}).get("methodology", methodology or {})
        required_artifacts = [str(item) for item in (getattr(spec, "required_artifacts", ()) or ())]
        identity_literal = json.dumps(identity, indent=4)
        findings_block = ""
        if findings:
            lines = [f"- [{f['finding_id']}] {f['status']}: {f['lesson']}" for f in findings]
            findings_block = "\n\nPrior findings for this question (do not repeat refuted mechanisms unchanged):\n" + "\n".join(lines)
        repair_block = (
            "\n\nPREVIOUS ATTEMPTS FAILED DETERMINISTIC VERIFICATION — fix ALL of these exactly:\n"
            + "\n".join(f"- {reason}" for reason in previous_failures)
            if previous_failures else ""
        )
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a research engineer who converts a frozen scientific methodology into a "
                    "complete, runnable Python experiment script. The script must faithfully implement "
                    "the methodology — a program that runs but implements a different method is worthless. "
                    "Stdlib only (json, math, random, pathlib, statistics, itertools, collections). "
                    "No network, no subprocess, no file reads outside the working directory. "
                    "Respond with ONE Python code block and nothing else."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"[{IMPLEMENTATION_PROMPT_VERSION}]\n"
                    f"Implement the experiment for this hypothesis: {spec.hypothesis}\n"
                    f"Baseline configuration: {spec.baseline_configuration}\n"
                    f"Proposed change (the candidate mechanism that must be exercised): {spec.proposed_change}\n"
                    f"Causal mechanism: {spec.causal_mechanism}\n"
                    f"Falsification condition: {spec.falsification_condition}\n"
                    f"Methodology (frozen — implement THIS): algorithm='{str(block.get('algorithm_outline') or '')[:1500]}' "
                    f"procedure='{str(block.get('procedure') or '')[:1200]}' "
                    f"predicted_observations='{block.get('predicted_observations') or ''}'\n"
                    f"Datasets: {spec.dataset_identity} | split: {spec.split_definition}\n"
                    f"Metrics to compute per arm and seed: {list(spec.metrics)} (direction {dict(spec.metric_direction)})\n"
                    f"Random seeds (iterate ALL of these): {list(spec.random_seeds)}\n"
                    f"Minimum meaningful effect: {dict(spec.minimum_meaningful_effect)}\n"
                    f"Statistical plan: {spec.statistical_plan}\n"
                    f"{findings_block}{repair_block}\n\n"
                    "HARD CONTRACT (a deterministic verifier checks every item):\n"
                    "1. Define at module level exactly:\n"
                    f"IDENTITY = {identity_literal}\n"
                    "2. Iterate every declared seed; for each seed compute baseline AND candidate values "
                    "of every metric by ACTUALLY running the method (no hardcoded result constants — "
                    "derive values from computation over the methodology's mechanism).\n"
                    "3. Write one artifact file per required artifact type under 'run_outputs/' "
                    f"(types: {required_artifacts}) and collect their descriptors.\n"
                    '4. End the script with exactly:\n'
                    'print(json.dumps({"identity": IDENTITY, "seeds": seed_rows, "artifacts": artifact_rows}))\n'
                    "   where seed_rows = [{\"seed\": s, \"baseline\": {metric: value}, \"candidate\": {metric: value}}, ...]\n"
                    "   and artifact_rows = [{\"artifact_type\": t, \"path\": str(p)}, ...]\n"
                    "5. Implement each methodology component as a named function whose name contains the component term."
                ),
            },
        ]
        result = await self.router.complete(
            ModelRole.RESEARCH_REASONING,
            messages=messages,
            project_id=spec.project_id,
            run_id=spec.run_id,
            prompt_version=IMPLEMENTATION_PROMPT_VERSION,
            operation="experiment_implementation",
        )
        return _extract_code(result.text)


def _extract_code(text: str) -> str:
    """Accept either a fenced code block or raw source."""
    fences = re.findall(r"```(?:python)?\s*\n(.*?)```", text, flags=re.DOTALL)
    if fences:
        return max(fences, key=len).strip() + "\n"
    return text.strip()
