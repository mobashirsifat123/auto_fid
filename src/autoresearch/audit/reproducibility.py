"""Reproducibility auditor (DESIGN §5.6).

Re-runs adapter-declared reproduction commands from the released artifacts
alone and scores ``reproduction_rate = rerun_match / commands_expected``.

Fail-closed rules (binding):
- No adapter commands ⇒ NOT_RUN / ``NOT_RUNNABLE/no_commands`` (rate None). A
  system that releases nothing runnable is *not* scored 1.0 — absence of
  evidence is recorded, never a pass.
- Command error, non-zero exit, or timeout ⇒ a rerun *mismatch*, never a skip.
- Shell features (pipes, redirects, ``&&``) are not supported: commands run as
  argv lists without a shell, so hostile meta-characters cannot execute.
- Containment is env-scrub policy (``PATH`` scrub, fixed ``PYTHONHASHSEED``,
  fresh process group), **not** an OS sandbox; ``containment_level`` is
  recorded in every detail record (soft spot #3 disclosure).

v1 check level is ``exit_code_clean_run``: a command matches when it exits 0
within the timeout in a pristine copy of the artifact tree and (if it emits a
single-line JSON measurement document) that document parses. Numeric
cross-checking of re-run outputs against reported values is handled by the
numbers metric's recomputation layer; this module records the parsed
measurement for the report so the runner can join them.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autoresearch.audit.contract import ReproductionCommand

__all__ = ["ReproducibilityAuditor", "ReproducibilityResult"]

_METRIC = "reproducibility"
_CONTAINMENT_LEVEL = "env_scrub_policy_not_os_sandbox"
_CHECK_LEVEL = "exit_code_clean_run_v1"
_DEFAULT_TIMEOUT = 600


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class ReproducibilityResult:
    status: str  # RUN_COMPLETE | NOT_RUN
    not_run_reason: str | None
    counts: Mapping[str, int]
    rates: Mapping[str, float | None]
    coverage: Mapping[str, Any]
    details: tuple[Mapping[str, Any], ...]

    def to_section(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "not_run_reason": self.not_run_reason,
            "llm_assisted": False,
            "counts": dict(self.counts),
            "rates": dict(self.rates),
            "coverage": dict(self.coverage),
            "details": [dict(detail) for detail in self.details],
        }


def _not_run(reason: str) -> ReproducibilityResult:
    return ReproducibilityResult(
        status="NOT_RUN",
        not_run_reason=reason,
        counts={},
        rates={"reproduction_rate": None},
        coverage={"items_examined": 0, "items_skipped": 0, "skip_reasons": {}},
        details=({"kind": "containment_level", "value": _CONTAINMENT_LEVEL},),
    )


def _parse_measurement(stdout: str) -> dict[str, Any] | None:
    """Parse a single-line JSON measurement document from stdout, mirroring
    the tool's own measurement contract (last non-empty line, JSON object)."""
    for line in reversed(stdout.strip().splitlines()):
        candidate = line.strip()
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


class ReproducibilityAuditor:
    """Re-runs released reproduction commands in a pristine artifact copy."""

    def __init__(self, *, timeout_seconds: int = _DEFAULT_TIMEOUT) -> None:
        self._timeout = int(timeout_seconds)

    def audit(
        self,
        commands: Sequence[ReproductionCommand],
        bundle_root: Path,
    ) -> ReproducibilityResult:
        if not commands:
            return _not_run("NOT_RUNNABLE/no_commands")

        rerun_match = 0
        details: list[dict[str, Any]] = []
        for command in commands:
            outcome = self._run_one(command, bundle_root)
            if outcome["matched"]:
                rerun_match += 1
            details.append(outcome)

        expected = len(commands)
        return ReproducibilityResult(
            status="RUN_COMPLETE",
            not_run_reason=None,
            counts={
                "commands_expected": expected,
                "rerun_match": rerun_match,
            },
            rates={},  # derived by MetricSection.complete in wiring
            coverage={
                "items_examined": expected,
                "items_skipped": 0,
                "skip_reasons": {},
            },
            details=(
                {
                    "kind": "containment_level",
                    "value": _CONTAINMENT_LEVEL,
                    "check_level": _CHECK_LEVEL,
                    "timeout_seconds": self._timeout,
                },
                *details,
            ),
        )

    def _run_one(
        self, command: ReproductionCommand, bundle_root: Path
    ) -> dict[str, Any]:
        base: dict[str, Any] = {
            "kind": "rerun",
            "command": list(command.command),
            "source_file": command.source_file,
            "description": command.description,
            "check_level": _CHECK_LEVEL,
        }
        if not command.command:
            return {**base, "matched": False, "outcome": "empty_command"}

        scrubbed_env = {
            "PATH": "/usr/bin:/bin",
            "PYTHONHASHSEED": "0",
            "HOME": str(Path(tempfile.gettempdir())),
        }
        try:
            with tempfile.TemporaryDirectory(prefix="audit_rerun_") as workdir_name:
                workdir = Path(workdir_name)
                self._copy_artifacts(bundle_root, workdir)
                try:
                    completed = subprocess.run(
                        list(command.command),
                        cwd=str(workdir),
                        env=scrubbed_env,
                        capture_output=True,
                        timeout=self._timeout,
                        start_new_session=True,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    return {
                        **base,
                        "matched": False,
                        "outcome": "timeout",
                        "timeout_seconds": self._timeout,
                    }
                except OSError as exc:
                    return {
                        **base,
                        "matched": False,
                        "outcome": f"spawn_error:{type(exc).__name__}",
                    }

                stdout = completed.stdout.decode("utf-8", errors="replace")
                stderr = completed.stderr.decode("utf-8", errors="replace")
                measurement = (
                    _parse_measurement(stdout) if completed.returncode == 0 else None
                )
                matched = completed.returncode == 0 and (
                    measurement is not None or not stdout.strip()
                )
                # Note: a clean-exit command that emits an unparseable trailing
                # line is scored a mismatch (check_level exit_code_clean_run_v1).
                return {
                    **base,
                    "matched": matched,
                    "outcome": "completed",
                    "exit_code": completed.returncode,
                    "stdout_sha256": _sha256(completed.stdout),
                    "stderr_sha256": _sha256(completed.stderr),
                    "measurement": measurement,
                }
        except Exception as exc:  # noqa: BLE001 - harness-side failure recorded, never raised past the auditor
            return {
                **base,
                "matched": False,
                "outcome": f"harness_error:{type(exc).__name__}",
            }

    @staticmethod
    def _copy_artifacts(bundle_root: Path, workdir: Path) -> None:
        """Copy the artifact tree (and only it) into the pristine workdir."""
        artifacts = bundle_root / "artifacts"
        if artifacts.is_dir():
            shutil.copytree(artifacts, workdir / "artifacts", dirs_exist_ok=True)
