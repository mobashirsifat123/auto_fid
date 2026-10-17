"""Stable Claude Code skill bridge and canonical skill management.

Architecture enforced here:

    Claude Skill -> `autoresearch skill ...` -> application service
        -> scientific pipeline -> structured artifact
        -> Claude Code interprets the --json result

The skills never drive an AgentRuntime conversational loop. Each command
calls a direct application service and returns a single JSON document.

Contract guarantees:
    * every command emits exactly one JSON document on stdout (no log noise)
      and structured errors on stderr
    * stable exit codes: 0 ok, 1 internal, 2 usage, 3 approval required,
      4 not found, 5 insufficient evidence, 6 worker unavailable
    * all output is passed through secret redaction
    * experiment planning never executes code; execution requires an exact
      specification approval that this module records only when --accept is
      passed explicitly
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable
from uuid import uuid4

import typer

from autoresearch.application.config import load_settings
from autoresearch.application.factory import ApplicationContainer
from autoresearch.domain.models import (
    EnvironmentManifest,
    ExperimentReport,
    ExperimentSpec,
    NoveltyCandidate,
    ResearchGap,
)
from autoresearch.providers.credentials import resolve_api_key
from autoresearch.providers.redaction import redact, redact_text

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_APPROVAL_REQUIRED = 3
EXIT_NOT_FOUND = 4
EXIT_INSUFFICIENT_EVIDENCE = 5
EXIT_WORKER_UNAVAILABLE = 6

# Exit codes for known machine-prefixed error messages ("code: detail").
_KNOWN_CODES = {
    "approval_required": EXIT_APPROVAL_REQUIRED,
    "approval_expired": EXIT_APPROVAL_REQUIRED,
    "checksum_mismatch": EXIT_USAGE,
    "insufficient_evidence": EXIT_INSUFFICIENT_EVIDENCE,
    "metric_schema_invalid": EXIT_INSUFFICIENT_EVIDENCE,
    "validation_failed": EXIT_INSUFFICIENT_EVIDENCE,
    "worker_unavailable": EXIT_WORKER_UNAVAILABLE,
}


@dataclass(frozen=True)
class SkillMeta:
    name: str
    version: str
    description: str
    entry: str


SKILLS = {
    "review": SkillMeta(
        name="review",
        version="1.0.0",
        description="Evidence-backed deep literature review: discover papers, extract evidence, rank by relevance, and surface a cited research gap. Primary skill — run first.",
        entry='autoresearch skill review --project <id>  # or: autoresearch skill workflow --seed-paper <pdf|doi|url>',
    ),
    "novelty": SkillMeta(
        name="novelty",
        version="1.0.0",
        description="Generate, prior-art-check, and adversarially critique novelty candidates for a validated gap. Run after review.",
        entry='autoresearch skill novelty --project <id>',
    ),
    "experiment": SkillMeta(
        name="experiment",
        version="1.0.0",
        description="Plan, approve, execute, inspect, and report a falsifiable experiment from a novelty candidate. Run after novelty.",
        entry='autoresearch skill experiment plan --candidate-id <id>',
    ),
    "workflow": SkillMeta(
        name="workflow",
        version="1.0.0",
        description="One durable cycle: seed → review → question → hypotheses → versioned methodology → design → approved execution → manuscript. Reuses completed stages; stops where asked.",
        entry='autoresearch skill workflow --seed-paper <pdf|doi|url> --until design  # resume: autoresearch skill workflow --project <id> --until manuscript',
    ),
    "manuscript": SkillMeta(
        name="manuscript",
        version="1.0.0",
        description="Stage-4 manuscript: deterministic, evidence-grounded draft (Markdown + LaTeX + BibTeX + tables + manifest) over the exact review→novelty→experiment lineage. Reports evidence gaps honestly; never fabricates results.",
        entry='autoresearch skill manuscript --project <id>  # or: autoresearch skill workflow --project <id> --until manuscript',
    ),
}

skill_app = typer.Typer(
    name="skill",
    help="Stable Claude Code skill bridge and canonical skill management.",
    no_args_is_help=True,
)


# --------------------------------------------------------------------------
# Skill management (canonical -> installed)
# --------------------------------------------------------------------------

def _canonical_root() -> Path:
    """Canonical skill definitions ship inside the package so wheel installs work."""
    configured = os.environ.get("AUTORESEARCH_SKILLS_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path(__file__).resolve().parents[1] / "skills"


def _target_roots(target: str) -> list[Path]:
    """Installation targets: Codex, Claude Code, and OpenCode skill dirs."""
    if target == "claude":
        configured = os.environ.get("CLAUDE_SKILLS_DIR", "").strip()
        return [Path(configured).expanduser() if configured else Path.home() / ".claude" / "skills"]
    if target == "opencode":
        configured = os.environ.get("OPENCODE_SKILLS_DIR", "").strip()
        return [Path(configured).expanduser() if configured else Path.home() / ".config" / "opencode" / "skills"]
    if target == "codex":
        configured = os.environ.get("CODEX_SKILLS_DIR", "").strip()
        return [Path(configured).expanduser() if configured else Path.home() / ".codex" / "skills"]
    if target == "all":
        roots: list[Path] = []
        for sub in ("codex", "claude", "opencode"):
            roots.extend(_target_roots(sub))
        return list(dict.fromkeys(roots))
    raise ValueError(f"unknown skill target '{target}'; use codex, claude, opencode, or all")


def _user_skills_dir() -> Path:
    return _target_roots("claude")[0]


def _canonical_dir(name: str) -> Path:
    return _canonical_root() / name


def _installed_dir(name: str, root: Path | None = None) -> Path:
    return (root or _user_skills_dir()) / name


def _skill_files(name: str) -> list[Path]:
    root = _canonical_dir(name)
    if not root.is_dir():
        return []
    return sorted(path for path in root.rglob("*") if path.is_file())


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_checksum(name: str) -> str:
    parts = b"\0".join(_sha256(path).encode("ascii") for path in _skill_files(name))
    return hashlib.sha256(parts).hexdigest()


def _provenance_file(name: str, root: Path | None = None) -> Path:
    return _installed_dir(name, root) / "provenance.json"


def _installed_provenance(name: str, root: Path | None = None) -> dict[str, Any] | None:
    try:
        value = json.loads(_provenance_file(name, root).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _install_skill(name: str, root: Path | None = None, *, force: bool = False) -> dict[str, Any]:
    meta = SKILLS.get(name)
    if not meta:
        raise ValueError(f"unknown skill '{name}'; available: {', '.join(sorted(SKILLS))}")
    source = _canonical_dir(name)
    files = _skill_files(name)
    if not files:
        raise ValueError(f"canonical skill '{name}' is missing SKILL.md (expected at {source / 'SKILL.md'})")
    root = root or _user_skills_dir()
    target = _installed_dir(name, root)
    # Collision safety: never silently overwrite a skill we did not install.
    # A foreign SKILL.md without our provenance marker is refused unless forced.
    if (target / "SKILL.md").is_file() and _installed_provenance(name, root) is None and not force:
        raise ValueError(
            f"skill_collision: {target} already exists and was not installed by AutoResearch; "
            "remove it or pass --force to overwrite"
        )
    target.mkdir(parents=True, exist_ok=True)
    for file in files:
        relative = file.relative_to(source)
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(file.read_bytes())
    checksum = _source_checksum(name)
    _provenance_file(name, root).write_text(json.dumps({
        "name": name,
        "version": meta.version,
        "source": str(source.resolve()),
        "source_checksum": checksum,
        "installed_at": datetime.now(timezone.utc).isoformat(),
    }, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "name": name,
        "version": meta.version,
        "installed": True,
        "path": str(target.resolve()),
        "source_checksum": checksum,
    }


@skill_app.command("list")
def skill_list() -> None:
    """List canonical skills and their install state per target as JSON."""
    rows = []
    for name in sorted(SKILLS):
        meta = SKILLS[name]
        canonical_present = (_canonical_dir(name) / "SKILL.md").is_file()
        targets = {}
        for target in ("codex", "claude", "opencode"):
            root = _target_roots(target)[0]
            provenance = _installed_provenance(name, root)
            targets[target] = {
                "installed": (_installed_dir(name, root) / "SKILL.md").is_file(),
                "synced_with_canonical": bool(provenance and provenance.get("source_checksum") == _source_checksum(name)),
                "path": str(_installed_dir(name, root)),
            }
        rows.append({
            "name": name,
            "version": meta.version,
            "description": meta.description,
            "entry": meta.entry,
            "canonical_present": canonical_present,
            "canonical_path": str(_canonical_dir(name)),
            "targets": targets,
        })
    typer.echo(json.dumps({"ok": True, "operation": "skill/list", "skills": rows}, indent=2))


def _selected_or_all(names: list[str]) -> list[str]:
    selected = sorted(names or SKILLS)
    for name in selected:
        if name not in SKILLS:
            raise typer.BadParameter(f"unknown skill '{name}'; available: {', '.join(sorted(SKILLS))}")
    return selected


@skill_app.command("install")
def skill_install(
    names: list[str] = typer.Argument(None, help="Skills to install (default: all)."),
    target: str = typer.Option("all", "--target", help="Install target: codex, claude, opencode, or all (default: all)."),
    force: bool = typer.Option(False, "--force", help="Overwrite a colliding skill we did not install."),
) -> None:
    """Install canonical skills into the Claude Code and/or OpenCode skill directories."""
    roots = _target_roots(target)
    results = []
    for name in _selected_or_all(names):
        for root in roots:
            entry = {"name": name, "target_root": str(root)}
            try:
                entry.update(_install_skill(name, root, force=force))
            except ValueError as exc:
                entry.update({"installed": False, "error": str(exc)})
            results.append(entry)
    ok = all(item.get("installed") for item in results)
    typer.echo(json.dumps({"ok": ok, "operation": "skill/install", "skills": results}, indent=2))
    if not ok:
        raise typer.Exit(1)


@skill_app.command("update")
def skill_update(
    names: list[str] = typer.Argument(None, help="Skills to refresh (default: all)."),
    target: str = typer.Option("all", "--target", help="Update target: codex, claude, opencode, or all (default: all)."),
) -> None:
    """Re-sync installed skills from the canonical package copies."""
    roots = _target_roots(target)
    results = []
    for name in _selected_or_all(names):
        for root in roots:
            entry = {"name": name, "target_root": str(root)}
            try:
                prior = _installed_provenance(name, root)
                result = _install_skill(name, root)
                result["changed"] = bool(prior is None or prior.get("source_checksum") != result["source_checksum"])
                entry.update(result)
            except ValueError as exc:
                entry.update({"installed": False, "error": str(exc)})
            results.append(entry)
    typer.echo(json.dumps({"ok": True, "operation": "skill/update", "skills": results}, indent=2))


@skill_app.command("uninstall")
def skill_uninstall(
    names: list[str] = typer.Argument(None, help="Skills to remove (default: all installed by this tool)."),
    target: str = typer.Option("all", "--target", help="Removal target: codex, claude, opencode, or all (default: all)."),
) -> None:
    """Remove only skill directories this tool installed (provenance-marked)."""
    import shutil

    roots = _target_roots(target)
    results = []
    for name in _selected_or_all(names):
        for root in roots:
            directory = _installed_dir(name, root)
            if _installed_provenance(name, root) is None:
                results.append({"name": name, "target_root": str(root), "removed": False,
                                "reason": "not installed by AutoResearch (no provenance marker); left untouched"})
                continue
            shutil.rmtree(directory)
            results.append({"name": name, "target_root": str(root), "removed": True, "path": str(directory)})
    typer.echo(json.dumps({"ok": True, "operation": "skill/uninstall", "skills": results}, indent=2))


@skill_app.command("doctor")
def skill_doctor() -> None:
    """Verify canonical skills, installs, repo hygiene, and backend/provider health."""
    checks: list[dict[str, Any]] = []
    for name in sorted(SKILLS):
        present = (_canonical_dir(name) / "SKILL.md").is_file()
        checks.append({
            "check": f"canonical/{name}",
            "ok": present,
            "detail": "SKILL.md present" if present else "missing canonical SKILL.md",
        })
    for name in sorted(SKILLS):
        for target in ("codex", "claude", "opencode"):
            root = _target_roots(target)[0]
            provenance = _installed_provenance(name, root)
            if not provenance:
                checks.append({"check": f"installed/{target}/{name}", "ok": False, "detail": f"not installed for {target}; run `autoresearch skill install --target {target}`"})
            elif provenance.get("source_checksum") != _source_checksum(name):
                checks.append({"check": f"installed/{target}/{name}", "ok": False, "detail": "out of date; run `autoresearch skill update`"})
            else:
                checks.append({"check": f"installed/{target}/{name}", "ok": True, "detail": "synced with canonical"})
    repo_skills = Path.cwd() / ".claude" / "skills"
    if repo_skills.is_dir():
        for entry in sorted(path.name for path in repo_skills.iterdir() if path.is_dir()):
            checks.append({
                "check": f"repo-copy/{entry}",
                "ok": False,
                "detail": "divergent copy under .claude/skills; canonicalize with `autoresearch skill install`",
            })

    required_roles = ("literature_extraction", "research_reasoning", "scientific_critic")

    async def provider_check() -> dict[str, Any]:
        container = ApplicationContainer(load_settings())
        try:
            missing_roles = [role for role in required_roles if role not in container.router.assignments]
            if missing_roles:
                return {"check": "providers", "ok": False, "detail": f"no model assigned to roles: {', '.join(missing_roles)}"}
            unavailable = []
            for name in sorted({container.router.assignments[role].provider for role in required_roles}):
                provider = container.router.registry.get(name)
                try:
                    resolve_api_key(provider.config)
                except Exception:
                    unavailable.append(name)
            if unavailable:
                return {"check": "providers", "ok": False, "detail": "credentials unavailable for: " + ", ".join(unavailable)}
            return {"check": "providers", "ok": True, "detail": "all research roles assigned; credentials resolved locally (live authentication not checked)"}
        finally:
            await container.close()

    try:
        checks.append(asyncio.run(provider_check()))
    except Exception as exc:  # container construction can fail without valid config
        checks.append({"check": "backend", "ok": False, "detail": redact_text(str(exc))})

    overall = all(check["ok"] for check in checks)
    typer.echo(json.dumps({
        "ok": overall,
        "operation": "skill/doctor",
        "overall": "healthy" if overall else "degraded",
        "exit_code": EXIT_OK if overall else EXIT_ERROR,
        "checks": checks,
    }, indent=2))
    raise typer.Exit(EXIT_OK if overall else EXIT_ERROR)


# --------------------------------------------------------------------------
# Research bridge (direct application services)
# --------------------------------------------------------------------------

def _classify(exc: Exception) -> tuple[str, int]:
    text = str(exc).strip()
    candidate: str | None = None
    if ":" in text:
        head = text.split(":", 1)[0].strip().lower()
        if re.fullmatch(r"[a-z][a-z0-9_]+", head):
            candidate = head
    if candidate and candidate.endswith("_not_found"):
        return candidate, EXIT_NOT_FOUND
    if candidate in _KNOWN_CODES:
        return candidate, _KNOWN_CODES[candidate]
    if candidate:
        return candidate, EXIT_ERROR
    lowered = text.lower()
    if "not found" in lowered:
        return "not_found", EXIT_NOT_FOUND
    return "internal_error", EXIT_ERROR


def _fail(code: str, exc: Exception, exit_code: int) -> None:
    typer.echo(json.dumps({
        "ok": False,
        "operation": None,
        "error": {
            "code": code,
            "message": redact_text(str(exc)),
            "type": type(exc).__name__,
        },
        "exit_code": exit_code,
    }, indent=2), err=True)
    raise typer.Exit(exit_code)


def _settings_for(project_id: str | None = None, *, candidate_id: str | None = None) -> Any:
    """Settings anchored to the project's recorded data root, so --project
    resume works from any working directory and projects never bleed into
    each other's state (or into the source checkout)."""
    from autoresearch.application.config import load_settings, resolve_settings

    settings = load_settings()
    if not project_id and candidate_id:
        from autoresearch.storage.repository import ResearchRepository

        probe = ResearchRepository(settings.database_path, settings.artifact_root)
        row = probe.row("SELECT project_id FROM novelty_candidates WHERE id=?", (candidate_id,))
        if row:
            project_id = str(row["project_id"])
    return resolve_settings(project_id)


def _invoke(operation: Callable[[ApplicationContainer], Awaitable[Any]], *, operation_name: str, project_id: str | None = None, candidate_id: str | None = None) -> None:
    from autoresearch.application.progress import reset_sink, set_sink, stderr_sink

    async def execute() -> Any:
        container = ApplicationContainer(_settings_for(project_id, candidate_id=candidate_id))
        token = set_sink(stderr_sink)  # live progress → stderr JSON lines
        try:
            return await operation(container)
        finally:
            reset_sink(token)
            await container.close()

    try:
        result = asyncio.run(execute())
    except TimeoutError as exc:
        _fail("runtime_limit", exc, EXIT_ERROR)
    except PermissionError as exc:
        _fail("approval_required", exc, EXIT_APPROVAL_REQUIRED)
    except (ValueError, RuntimeError) as exc:
        code, exit_code = _classify(exc)
        _fail(code, exc, exit_code)
    except Exception as exc:
        _fail("internal_error", exc, EXIT_ERROR)

    payload: dict[str, Any] = {"ok": True, "operation": operation_name}
    if isinstance(result, dict):
        payload.update(result)
    else:
        payload["result"] = result
    typer.echo(json.dumps(redact(payload), indent=2, default=str))


def _gap_summary(gap: ResearchGap) -> dict[str, Any]:
    return {
        "id": gap.id,
        "statement": gap.statement,
        "classification": gap.classification.value,
        "confidence": gap.confidence,
        "supporting_paper_ids": list(gap.supporting_paper_ids),
        "supporting_claim_ids": list(gap.supporting_claim_ids),
        "counter_evidence_claim_ids": list(gap.counter_evidence_claim_ids),
        "why_unsolved": gap.why_unsolved,
        "missing_experiment": gap.missing_experiment,
        "falsification_test": gap.falsification_test,
    }


def _discovery_summary(outcome) -> dict[str, Any]:
    return {
        "project_id": outcome.project_id,
        "run_id": outcome.run_id,
        "requested": outcome.requested,
        "retrieved": outcome.retrieved,
        "downloaded": outcome.downloaded,
        "unavailable": outcome.unavailable,
        "rejected": outcome.rejected,
        "failed": outcome.failed,
        "artifact_path": outcome.artifact_path,
    }


def _spec_summary(spec: ExperimentSpec) -> dict[str, Any]:
    return {
        "id": spec.id,
        "candidate_id": spec.candidate_id,
        "run_id": spec.run_id,
        "hypothesis": spec.hypothesis,
        "proposed_change": spec.proposed_change,
        "falsification_condition": spec.falsification_condition,
        "metrics": list(spec.metrics),
        "random_seeds": list(spec.random_seeds),
        "required_seed_count": spec.required_seed_count,
        "statistical_plan": spec.statistical_plan,
        "network_policy": spec.network_policy,
        "target_worker": spec.target_worker,
    }


def _manifest_summary(manifest: EnvironmentManifest | None) -> dict[str, Any] | None:
    if manifest is None:
        return None
    return {
        "manifest_id": manifest.id,
        "spec_id": manifest.spec_id,
        "environment_manager": manifest.environment_manager,
        "python_version": manifest.python_version,
        "dependency_lock_checksum": manifest.dependency_lock_checksum,
        "dataset_paths": list(manifest.dataset_paths),
    }
@skill_app.command("review")
def review(
    project: str = typer.Option(..., "--project", help="Project ID returned by discover."),
    top_k: int = typer.Option(5, "--top-k", min=1, max=15),
    hierarchical: bool = typer.Option(False, "--hierarchical/--flat", help="Chunk-level hierarchical review for long PDFs."),
    fresh: bool = typer.Option(False, "--fresh", help="Re-extract every paper review even when a verified one is reusable."),
    max_runtime_seconds: float | None = typer.Option(None, "--max-runtime-seconds", min=1, help="Optional total review deadline; unlimited when omitted."),
) -> None:
    """Run the deep review over an existing project corpus (reuses verified reviews by default)."""

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        outcome = await c.review.review(project, top_k=top_k, hierarchical=hierarchical, fresh=fresh, max_runtime_seconds=max_runtime_seconds)
        from autoresearch.review.matrix import project_keywords
        return {
            "project_id": outcome.project_id,
            "review_run_id": outcome.run_id,
            "reviewed": outcome.reviewed,
            "reused_reviews": outcome.reused,
            "failed_reviews": outcome.failed,
            "top_paper_ids": list(outcome.top_paper_ids),
            "keywords": project_keywords(c.repository, project_id=project, run_id=outcome.run_id),
            "gap": _gap_summary(outcome.gap),
            "json_report": outcome.json_report,
            "markdown_report": outcome.markdown_report,
        }

    _invoke(operation, operation_name="review", project_id=project)


@skill_app.command("novelty")
def novelty(
    project: str = typer.Option(..., "--project", help="Project ID with a validated research gap."),
    gap_id: str | None = typer.Option(None, "--gap-id", help="Specific gap; default is the highest-confidence gap."),
    max_runtime_seconds: float | None = typer.Option(None, "--max-runtime-seconds", min=1, help="Optional total novelty deadline; unlimited when omitted."),
) -> None:
    """Generate, prior-art-check, and adversarially critique novelty candidates."""

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        existing = c.repository.get_project(project)
        if not existing:
            raise ValueError(f"project_not_found: project '{project}' not found")
        portfolio_id = await c.novelty.generate_portfolio(project, gap_id=gap_id, max_runtime_seconds=max_runtime_seconds)
        portfolio = c.repository.get_novelty_portfolio_by_id(portfolio_id)
        if not portfolio:
            raise ValueError(f"portfolio_not_found: novelty portfolio '{portfolio_id}' not found")
        candidate_ids = tuple(
            value for value in (portfolio.low_risk_candidate_id, portfolio.balanced_candidate_id, portfolio.high_risk_candidate_id)
            if value
        )
        candidates = []
        resolved_gap_id = gap_id
        for candidate_id in candidate_ids:
            candidate = c.repository.get_novelty_candidate(candidate_id)
            if candidate is None:
                continue
            resolved_gap_id = resolved_gap_id or candidate.gap_id
            assessment = c.repository.get_prior_art_assessment(candidate_id)
            critique = c.repository.get_novelty_critique(candidate_id)
            candidates.append({
                "candidate_id": candidate_id, "method": candidate.method,
                "difference_from_existing": candidate.difference_from_existing,
                "falsification_condition": candidate.falsification_condition,
                "minimum_validating_experiment": candidate.minimum_validating_experiment,
                "overlap_status": assessment.overlap_status.value if assessment else "unassessed",
                "unresolved_overlap": assessment.unresolved_overlap if assessment else "",
                "objections": list(critique.objections) if critique else [],
                "failure_scenarios": list(critique.failure_scenarios) if critique else [],
            })
        return {
            "project_id": project,
            "portfolio_id": portfolio_id,
            "gap_id": resolved_gap_id,
            "candidate_ids": list(candidate_ids),
            "candidates": candidates,
            "low_risk_candidate_id": portfolio.low_risk_candidate_id,
            "balanced_candidate_id": portfolio.balanced_candidate_id,
            "high_risk_candidate_id": portfolio.high_risk_candidate_id,
        }

    _invoke(operation, operation_name="novelty", project_id=project)


@skill_app.command("workflow")
def workflow(
    seed_paper: str = typer.Option("", "--seed-paper", help="Seed PDF path, DOI, or HTTPS URL for a new project."),
    from_catalog: Path | None = typer.Option(None, "--from-catalog", help="Local JSON paper catalog for a new project."),
    project: str = typer.Option("", "--project", help="Existing project ID to resume/extend."),
    until: str = typer.Option("design", "--until", help="Stop after: review, question, hypotheses, methodology, design, experiment, or manuscript."),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    top_k: int = typer.Option(5, "--top-k", min=1, max=15),
    max_cost_usd: float | None = typer.Option(None, "--max-cost-usd", min=0.0, help="Cumulative project provider-cost hard stop (applies to this and later invocations)."),
    max_tokens: int | None = typer.Option(None, "--max-tokens", min=1, help="Project total-token ceiling (prompt+completion) across all provider calls."),
) -> None:
    """Advance one project through the full durable research cycle, reusing completed stages."""

    # Validate arguments BEFORE any side effect (importing a catalog creates a project).
    supplied = sum(1 for value in (seed_paper, from_catalog is not None, project) if value)
    if supplied != 1:
        raise typer.BadParameter("Provide exactly one of --seed-paper, --from-catalog, or --project.")

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        project_id: str | None = project or None
        seed: str | None = seed_paper or None
        if project_id is None and from_catalog is not None:
            outcome = await c.discovery.import_catalog(from_catalog)
            project_id = outcome.project_id
        result = await c.workflow.run(
            seed=seed, project_id=project_id, until=until,
            limit=limit, top_k=top_k, max_cost_usd=max_cost_usd,
            max_total_tokens=max_tokens,
        )
        return result

    _invoke(operation, operation_name="workflow", project_id=project)
experiment_app = typer.Typer(
    name="experiment",
    help="Falsifiable experiment lifecycle from a novelty candidate. Planning never executes code.",
    no_args_is_help=True,
)


def _require_candidate(c: ApplicationContainer, candidate_id: str) -> NoveltyCandidate:
    candidate = c.repository.get_novelty_candidate(candidate_id)
    if not candidate:
        raise ValueError(f"novelty_candidate_not_found: candidate '{candidate_id}' not found")
    return candidate


@experiment_app.command("plan")
def experiment_plan(
    candidate_id: str = typer.Option(..., "--candidate-id", help="Novelty candidate to translate into a spec."),
    max_runtime_seconds: float | None = typer.Option(None, "--max-runtime-seconds", min=1, help="Optional total planning deadline; unlimited when omitted."),
) -> None:
    """Translate a novelty candidate into an approval-gated ExperimentSpec (no code runs)."""

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        _require_candidate(c, candidate_id)
        spec_id = await c.experiment.execute_candidate(candidate_id, mode="plan", max_runtime_seconds=max_runtime_seconds)
        spec = c.repository.get_experiment_spec_by_id(spec_id)
        preview = c.experiment.approval_preview_for_candidate(candidate_id)
        return {
            "spec_id": spec_id,
            "candidate_id": candidate_id,
            "run_id": spec.run_id,
            "status": "planned",
            "executed": False,
            "approval_required": True,
            "approval": {
                "worker": preview["worker"],
                "actions": preview["actions"],
                "checksum": preview["checksum"],
                "budget": preview["budget"],
            },
            "spec": _spec_summary(spec),
        }

    _invoke(operation, operation_name="experiment/plan", candidate_id=candidate_id)


@experiment_app.command("validate")
def experiment_validate(
    candidate_id: str = typer.Option(..., "--candidate-id", help="Novelty candidate to validate."),
) -> None:
    """Prepare and verify the execution environment for a spec (no code runs)."""

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        _require_candidate(c, candidate_id)
        spec_id = await c.experiment.execute_candidate(candidate_id, mode="validate")
        spec = c.repository.get_experiment_spec_by_id(spec_id)
        manifest = c.repository.get_environment_manifest(spec_id)
        return {
            "spec_id": spec_id,
            "candidate_id": candidate_id,
            "run_id": spec.run_id,
            "status": "validated",
            "executed": False,
            "environment": _manifest_summary(manifest),
        }

    _invoke(operation, operation_name="experiment/validate", candidate_id=candidate_id)


@experiment_app.command("implement")
def experiment_implement(
    candidate_id: str = typer.Option(..., "--candidate-id", help="Novelty candidate whose planned spec needs a working implementation."),
) -> None:
    """Generate and statically verify the experiment implementation (no execution)."""

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        _require_candidate(c, candidate_id)
        return await c.experiment.implement_candidate(candidate_id)

    _invoke(operation, operation_name="experiment/implement", candidate_id=candidate_id)


@experiment_app.command("approve")
def experiment_approve(
    candidate_id: str = typer.Option(..., "--candidate-id", help="Novelty candidate whose spec is being approved."),
    accept: bool = typer.Option(False, "--accept", help="Record the exact-spec approval. Without it nothing is written."),
) -> None:
    """Preview the exact-spec approval; with --accept record it to unblock execution."""

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        _require_candidate(c, candidate_id)
        preview = c.experiment.approval_preview_for_candidate(candidate_id)
        if not accept:
            return {
                "approved": False,
                "reason": "approval not recorded; review the exact spec preview and re-run with --accept",
                "preview": _approval_preview(preview),
            }
        run_id = str(preview["run_id"])
        approval = c.repository.save_exact_approval({
            "id": uuid4().hex,
            "run_id": run_id,
            "specification_checksum": str(preview["checksum"]),
            "approver": "claude-code-skill",
            "approved_scope": "local-experiment",
            "approved_target": str(preview["worker"]),
            "approved_budget": float(preview["budget"]),
            "expires_at": "9999-12-31T23:59:59+00:00",
            "status": "pending",
            "approved_actions": preview["actions"],
        })
        approved = c.repository.approve_exact(str(approval["id"]), approver="claude-code-skill") or approval
        return {
            "approved": True,
            "approval_id": approval["id"],
            "status": "approved",
            "checksum": preview["checksum"],
            "preview": _approval_preview(preview),
        }

    _invoke(operation, operation_name="experiment/approve", candidate_id=candidate_id)


def _approval_preview(preview: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": preview["run_id"],
        "spec_id": preview["spec_id"],
        "candidate_id": preview["candidate_id"],
        "checksum": preview["checksum"],
        "worker": preview["worker"],
        "actions": preview["actions"],
        "budget": preview["budget"],
    }


@experiment_app.command("execute")
def experiment_execute(
    candidate_id: str = typer.Option(..., "--candidate-id", help="Novelty candidate whose approved spec to execute."),
) -> None:
    """Execute an approved exact-spec action plan through the policy-controlled worker."""

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        _require_candidate(c, candidate_id)
        spec = c.repository.get_experiment_spec(candidate_id)
        if not spec:
            raise ValueError("approval_required: plan the experiment before execution")
        preview = c.experiment.approval_preview_for_candidate(candidate_id)
        existing = c.repository.approval_by_checksum(str(preview["checksum"]))
        if not existing or existing.get("status") != "approved":
            raise PermissionError(
                "approval_required: exact-spec approval not granted; run "
                "`autoresearch skill experiment approve --candidate-id <id> --accept` first"
            )
        spec_id = await c.experiment.execute_candidate(candidate_id, mode="execute")
        run = c.repository.get_run(spec.run_id)
        jobs = c.repository.worker_jobs_for_run(spec.run_id)
        return {
            "spec_id": spec_id,
            "candidate_id": candidate_id,
            "run_id": spec.run_id,
            "run_status": run.status.value if run else None,
            "jobs": jobs,
        }

    _invoke(operation, operation_name="experiment/execute", candidate_id=candidate_id)


@experiment_app.command("inspect")
def experiment_inspect(
    candidate_id: str = typer.Option(..., "--candidate-id", help="Novelty candidate whose spec to inspect."),
) -> None:
    """Inspect the durable environment manifest for a spec (no code runs)."""

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        _require_candidate(c, candidate_id)
        spec_id = await c.experiment.execute_candidate(candidate_id, mode="inspect")
        spec = c.repository.get_experiment_spec_by_id(spec_id)
        manifest = c.repository.get_environment_manifest(spec_id)
        return {
            "spec_id": spec_id,
            "candidate_id": candidate_id,
            "run_id": spec.run_id,
            "status": "inspected",
            "executed": False,
            "environment": _manifest_summary(manifest),
        }

    _invoke(operation, operation_name="experiment/inspect", candidate_id=candidate_id)


@experiment_app.command("report")
def experiment_report(
    candidate_id: str = typer.Option(..., "--candidate-id", help="Novelty candidate whose executed experiment to report."),
) -> None:
    """Write (or reuse) the deterministic report for an analyzed experiment."""

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        _require_candidate(c, candidate_id)
        return await c.experiment.report_for_candidate(candidate_id)

    _invoke(operation, operation_name="experiment/report", candidate_id=candidate_id)


@skill_app.command("manuscript")
def skill_manuscript(
    project: str = typer.Option(..., "--project", help="Project ID whose validated lineage to write the manuscript from."),
) -> None:
    """Generate the stage-4 manuscript from the exact review→novelty→experiment lineage."""

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        return await c.manuscript.generate(project)

    _invoke(operation, operation_name="skill/manuscript", project_id=project)


@experiment_app.command("cancel")
def experiment_cancel(
    run_id: str = typer.Option(..., "--run-id", help="Experiment run ID to cancel."),
) -> None:
    """Cancel an experiment run that has not reached a terminal state."""

    async def operation(c: ApplicationContainer) -> dict[str, Any]:
        run = c.repository.get_run(run_id)
        if not run:
            raise ValueError(f"run_not_found: run '{run_id}' not found")
        c.repository.cancel_run(run_id)
        return {"run_id": run_id, "status": "cancelled"}

    _invoke(operation, operation_name="experiment/cancel", candidate_id=candidate_id)


skill_app.add_typer(experiment_app)
