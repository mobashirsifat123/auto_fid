"""AutoResearch milestone CLI."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import typer

from autoresearch.application.config import load_settings
from autoresearch.application.factory import ApplicationContainer
from autoresearch.application.audit import write_audit
from autoresearch.cli.skills import skill_app
from autoresearch.domain.models import PipelineKind
from autoresearch.providers.redaction import redact, redact_text
from autoresearch.release import (
    ProductConfig,
    ProductPaths,
    activate_staged,
    api_request,
    create_diagnostics,
    daemon_environment,
    doctor as run_doctor,
    launch_tui,
    remove_installation,
    stage_update,
    start_daemon,
    stream_sse,
    verify_release_artifact,
    write_config,
    rollback_update,
)
from autoresearch.versioning import API_VERSION, CLI_VERSION, DATABASE_SCHEMA_VERSION, PRODUCT_VERSION, WORKER_PROTOCOL_VERSION

app = typer.Typer(name="autoresearch", help="AutoResearch local-first coding and scientific workflows.", no_args_is_help=False)
audit_app = typer.Typer(help="Inspect auditable scientific provenance.")


def _paths() -> ProductPaths:
    return ProductPaths.current()


def _product_config(paths: ProductPaths | None = None) -> ProductConfig:
    return ProductConfig.load(paths or _paths())


def _claude_session_state(paths: ProductPaths, repository_root: Path) -> Path:
    """Return a local, secret-free state file shared by Claude Code skills."""
    digest = hashlib.sha256(str(repository_root.resolve()).encode("utf-8")).hexdigest()[:20]
    return paths.state_dir / "claude-code-sessions" / f"{digest}.json"


def _backend_repository_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _read_claude_sessions(path: Path) -> dict[str, str]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {str(key): str(item) for key, item in value.items() if isinstance(key, str) and isinstance(item, str)}


def _write_claude_sessions(path: Path, sessions: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(sessions, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _release_output(value: Any, *, json_output: bool = True) -> None:
    if json_output:
        typer.echo(json.dumps(redact(value), indent=2, default=str))
    else:
        for item in value:
            typer.echo(f"{item['status']:7} {item['name']}: {item['detail']}" + (f"\n         {item['remedy']}" if item.get("remedy") else ""))


@app.callback(invoke_without_command=True)
def root(ctx: typer.Context) -> None:
    """Launch the packaged AutoResearch terminal application."""
    if ctx.invoked_subcommand is None:
        paths = _paths()
        try:
            exit_code = launch_tui(_product_config(paths), paths, repository_root=Path.cwd())
        except Exception as exc:
            typer.echo(json.dumps({"error": redact_text(str(exc)), "next": "Run `autoresearch doctor` for actionable checks."}, indent=2), err=True)
            raise typer.Exit(1) from exc
        raise typer.Exit(exit_code)


@app.command()
def version() -> None:
    """Show product and protocol versions."""
    _release_output({"product_version": PRODUCT_VERSION, "cli_version": CLI_VERSION, "api_version": API_VERSION, "database_schema_version": DATABASE_SCHEMA_VERSION, "worker_protocol_version": WORKER_PROTOCOL_VERSION})


@app.command()
def setup(
    backend_url: str | None = typer.Option(None, help="Local backend URL without embedded credentials."),
    daemon_port: int | None = typer.Option(None, min=1024, max=65535),
    provider_config: Path | None = typer.Option(None, help="Secret-free provider TOML path."),
    allowed_root: list[Path] = typer.Option([], "--allowed-root", help="Allowed repository root; repeat as needed."),
    tui_executable: Path | None = typer.Option(None, help="Installed packaged TUI binary."),
    auth_token_env: str | None = typer.Option(None, help="Existing backend-auth token environment-variable name."),
    cost_budget_usd: float | None = typer.Option(None, min=0),
    telemetry: bool | None = typer.Option(None, "--telemetry/--no-telemetry"),
) -> None:
    """Create safe local configuration and product directories; never stores provider keys."""
    paths = _paths()
    paths.create()
    if provider_config and not provider_config.expanduser().is_file():
        raise typer.BadParameter("provider_config must be an existing secret-free TOML file.")
    updates: dict[str, Any] = {
        "backend_url": backend_url,
        "daemon_port": daemon_port,
        "provider_config": str(provider_config.expanduser()) if provider_config else None,
        "allowed_repository_roots": tuple(str(path.expanduser().resolve()) for path in allowed_root) if allowed_root else None,
        "tui_executable": str(tui_executable.expanduser()) if tui_executable else None,
        "authentication_token_environment_variable": auth_token_env,
        "cost_budget_usd": cost_budget_usd,
        "telemetry_enabled": telemetry,
    }
    file = write_config(paths, _product_config(paths).with_updates(**updates))
    _release_output({"config": str(file), "data_dir": str(paths.data_dir), "state_dir": str(paths.state_dir), "cache_dir": str(paths.cache_dir), "provider_keys_stored": False})


@app.command()
def login(
    auth_token_env: str | None = typer.Option(None, help="Existing environment variable containing a backend-auth token."),
    provider_config: Path | None = typer.Option(None, help="Secret-free provider TOML path."),
) -> None:
    """Record only credential references; actual provider keys remain in the backend environment."""
    if not auth_token_env and not provider_config:
        raise typer.BadParameter("Specify --auth-token-env and/or --provider-config; never pass raw credentials.")
    setup(auth_token_env=auth_token_env, provider_config=provider_config)


@app.command()
def doctor(check_provider: bool = typer.Option(False, "--check-provider"), json_output: bool = typer.Option(False, "--json")) -> None:
    """Diagnose local installation, daemon, configuration, and safe capability prerequisites."""
    checks = run_doctor(_product_config(), _paths(), check_provider=check_provider)
    _release_output(checks, json_output=json_output)
    if any(item["status"] == "FAIL" for item in checks):
        raise typer.Exit(1)


@app.command()
def serve() -> None:
    """Run the local backend daemon in the foreground."""
    import uvicorn
    paths, config = _paths(), _product_config()
    os.environ.update(daemon_environment(config, paths, repository_root=Path.cwd()))
    uvicorn.run("autoresearch.api.app:create_app", factory=True, host=config.daemon_host, port=config.daemon_port, log_level="warning")


@app.command()
def status(json_output: bool = typer.Option(False, "--json")) -> None:
    """Show local daemon and API compatibility status."""
    paths, config = _paths(), _product_config()
    try:
        health, started = start_daemon(config, paths, repository_root=Path.cwd())
        payload: dict[str, Any] = {"daemon": "started" if started else "already_running", "health": health}
    except RuntimeError as exc:
        payload = {"daemon": "unavailable", "error": redact_text(str(exc))}
    if json_output:
        _release_output(payload)
    else:
        _release_output([{"status": "FAIL" if payload["daemon"] == "unavailable" else "PASS", "name": "daemon", "detail": str(payload), "remedy": "Run `autoresearch doctor`." if payload["daemon"] == "unavailable" else ""}], json_output=False)
    if payload["daemon"] == "unavailable":
        raise typer.Exit(1)


@app.command()
def diagnostics(output: Path = typer.Option(Path("autoresearch-diagnostics.zip"), "--output")) -> None:
    """Write a privacy-safe diagnostic archive without research file contents."""
    archive = create_diagnostics(_product_config(), _paths(), output)
    _release_output({"diagnostic_archive": str(archive), "includes_file_contents": False, "includes_provider_keys": False})


@app.command()
def update(source: Path = typer.Option(..., help="Local release artifact directory containing manifest.json."), activate: bool = typer.Option(False, "--activate", help="Activate the verified TUI and retain the previous binary for rollback.")) -> None:
    """Verify checksums, stage a release, and optionally activate its host TUI."""
    manifest = verify_release_artifact(source)
    staged = stage_update(source, _paths())
    payload: dict[str, Any] = {"staged_release": str(staged), "product_version": manifest.get("product_version"), "signed": bool(manifest.get("signature")), "activation": "staged"}
    if activate:
        target, previous = activate_staged(staged, _paths())
        write_config(_paths(), _product_config().with_updates(tui_executable=str(target)))
        payload.update({"activation": "active", "tui_executable": str(target), "previous_release": str(previous) if previous else None, "health_check": "backend compatibility remains checked by launcher startup"})
    _release_output(payload)


@app.command()
def rollback() -> None:
    """Restore the most recent retained TUI release after an unsuccessful update."""
    target = rollback_update(_paths())
    write_config(_paths(), _product_config().with_updates(tui_executable=str(target / ("autoresearch-tui.exe" if os.name == "nt" else "autoresearch-tui"))))
    _release_output({"rollback": "completed", "tui_executable": str(target)})


@app.command()
def uninstall(yes: bool = typer.Option(False, "--yes"), purge_data: bool = typer.Option(False, "--purge-data")) -> None:
    """Remove known product files; data removal requires explicit `--purge-data`."""
    paths = _paths()
    if not yes:
        _release_output({"would_remove": str(paths.install_dir), "would_purge_data": purge_data, "next": "Repeat with --yes to uninstall."})
        return
    _release_output({"removed": [str(path) for path in remove_installation(paths, purge_data=purge_data)], "purged_data": purge_data})


@app.command()
def run(
    prompt: str = typer.Option(..., "--prompt", help="Prompt for one durable agent turn."),
    json_output: bool = typer.Option(False, "--json"),
    jsonl: bool = typer.Option(False, "--jsonl"),
    timeout_seconds: float = typer.Option(120.0, min=1, max=600),
) -> None:
    """Run one durable headless turn and emit plain text, JSON, or JSONL events."""
    if json_output and jsonl:
        raise typer.BadParameter("Use either --json or --jsonl, not both.")
    paths, config = _paths(), _product_config()
    start_daemon(config, paths, repository_root=Path.cwd())
    session = api_request(config, "/v1/agent/sessions", method="POST", payload={"project_title": Path.cwd().name or "AutoResearch workspace", "active_repository": str(Path.cwd()), "working_directory": str(Path.cwd()), "token_budget": 32_000, "cost_budget": config.cost_budget_usd}, timeout=timeout_seconds)
    session_id = str(session["session"]["id"])
    turn = api_request(config, f"/v1/agent/sessions/{session_id}/turns", method="POST", payload={"message": prompt, "capability_override": None, "token_budget": None, "cost_budget": config.cost_budget_usd, "time_budget_seconds": timeout_seconds}, timeout=timeout_seconds)
    events = list(stream_sse(config, f"/v1/agent/sessions/{session_id}/events?last_event_id=0", timeout=timeout_seconds))
    snapshot = api_request(config, f"/v1/agent/sessions/{session_id}", timeout=timeout_seconds)
    if jsonl:
        for event in events:
            typer.echo(json.dumps(event, default=str))
    elif json_output:
        _release_output({"session_id": session_id, "turn": turn, "events": events, "session": snapshot})
    else:
        answer = next((str(item.get("content", "")) for item in reversed(snapshot.get("messages", [])) if item.get("role") == "assistant"), "No assistant response was recorded.")
        typer.echo(answer)


@app.command("agent")
def agent(
    mode: str = typer.Option(..., "--mode", help="One of review, novelty, or experiment."),
    prompt: str | None = typer.Option(None, "--prompt", help="Research request for the selected durable pipeline."),
    project: str | None = typer.Option(None, "--project", help="Use an existing discover project ID for the paper set."),
    wait: bool = typer.Option(False, "--wait", help="Stream the session events until the current turn reaches a terminal state."),
) -> None:
    """Start or continue a Claude Code-facing research mode without exposing provider keys."""
    mode_aliases = {"review": "literature_review", "novelty": "novelty_generation", "experiment": "experiment"}
    normalized = mode_aliases.get(mode.strip().lower())
    if not normalized:
        raise typer.BadParameter("mode must be one of: review, novelty, experiment.")
    if not prompt and not wait:
        raise typer.BadParameter("Provide --prompt, or use --wait only with an already-started turn.")

    paths, config = _paths(), _product_config()
    workspace_root = Path.cwd().resolve()
    backend_root = _backend_repository_root()
    start_daemon(config, paths, repository_root=backend_root)
    state_file = _claude_session_state(paths, workspace_root)
    sessions = _read_claude_sessions(state_file)
    session_id = sessions.get(normalized)
    session_payload: dict[str, Any] | None = None
    if session_id:
        try:
            session_payload = api_request(config, f"/v1/agent/sessions/{session_id}", timeout=5.0)
        except RuntimeError:
            session_id = None
    if not session_id:
        created = api_request(
            config,
            "/v1/agent/sessions",
            method="POST",
            payload={
                "project_title": workspace_root.name or "AutoResearch workspace",
                "active_repository": str(backend_root),
                "working_directory": str(backend_root),
                "token_budget": 32_000,
                "cost_budget": config.cost_budget_usd,
            },
            timeout=10.0,
        )
        session_id = str(created["session"]["id"])
        sessions[normalized] = session_id
        _write_claude_sessions(state_file, sessions)

    api_request(config, f"/v1/agent/sessions/{session_id}/mode", method="POST", payload={"mode": normalized, "auto_routing": True}, timeout=5.0)

    # If --project is provided, link the discover project's papers to the session's project
    if project:
        try:
            api_request(config, f"/v1/agent/sessions/{session_id}/link-project", method="POST", payload={"project_id": project}, timeout=5.0)
        except RuntimeError as exc:
            _release_output({"mode": normalized, "session_id": session_id, "workspace": str(workspace_root), "error": f"Failed to link project '{project}': {exc}", "provider_keys_exposed": False})
            raise typer.Exit(1)

    response: dict[str, Any] = {"mode": normalized, "session_id": session_id, "workspace": str(workspace_root), "started": False, "provider_keys_exposed": False}
    if project:
        response["linked_project"] = project
    if prompt:
        overrides = {"literature_review": "@review", "novelty_generation": "@invent", "experiment": "@experiment"}
        turn = api_request(config, f"/v1/agent/sessions/{session_id}/turns", method="POST", payload={"message": f"{overrides[normalized]} {prompt}", "capability_override": None, "token_budget": None, "cost_budget": config.cost_budget_usd, "time_budget_seconds": 600}, timeout=10.0)
        response.update({"started": True, "turn_id": turn["turn_id"], "status": turn["status"], "background": not wait})
    if wait:
        response["events"] = list(stream_sse(config, f"/v1/agent/sessions/{session_id}/events?last_event_id=0", timeout=600.0))
        response["session"] = api_request(config, f"/v1/agent/sessions/{session_id}", timeout=10.0)
    _release_output(response)


@app.command()
def discover(
    seed_paper: str = typer.Option("", "--seed-paper", help="Seed PDF path, DOI, or HTTPS URL."),
    from_catalog: Path | None = typer.Option(None, "--from-catalog", help="Import papers from a local JSON catalog instead of external search."),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    since_year: int = typer.Option(datetime.now(timezone.utc).year - 3, "--since-year", min=1900),
    max_cost_usd: float | None = typer.Option(None, "--max-cost-usd", min=0.0, help="Cumulative project provider-cost hard stop."),
    max_tokens: int | None = typer.Option(None, "--max-tokens", min=1, help="Project total-token ceiling (prompt+completion) across all provider calls."),
    max_runtime_seconds: float | None = typer.Option(None, "--max-runtime-seconds", min=1.0, help="Runtime limit for this invocation."),
) -> None:
    """Retrieve recent related papers and legally accessible PDFs. Use --from-catalog to import local papers."""
    if from_catalog:
        _run(lambda container: container.discovery.import_catalog(from_catalog))
    elif seed_paper:
        _run(lambda container: container.discovery.discover(seed_paper, limit=limit, since_year=since_year, max_cost_usd=max_cost_usd, max_runtime_seconds=max_runtime_seconds))
    else:
        raise typer.BadParameter("Provide --seed-paper for external search or --from-catalog for local import.")


@app.command()
def review(
    project: str = typer.Option(..., "--project", help="Project ID returned by discover."),
    top_k: int = typer.Option(5, "--top-k", min=1, max=15),
    hierarchical: bool = typer.Option(False, "--hierarchical/--flat", help="Use chunk-level hierarchical review for long legal PDFs."),
    max_cost_usd: float | None = typer.Option(None, "--max-cost-usd", min=0.0, help="Cumulative project provider-cost hard stop."),
    max_runtime_seconds: float | None = typer.Option(None, "--max-runtime-seconds", min=1.0, help="Runtime limit for this invocation."),
) -> None:
    """Review the corpus, rank top papers, and write one cited gap report."""
    _run(lambda container: container.review.review(project, top_k=top_k, hierarchical=hierarchical, max_cost_usd=max_cost_usd, max_runtime_seconds=max_runtime_seconds))


@app.command()
def workflow(
    seed: str = typer.Option("", "--seed", help="Seed PDF path, DOI, or HTTPS URL for a new project."),
    project: str = typer.Option("", "--project", help="Existing project ID to resume/extend."),
    until: str = typer.Option("design", "--until", help="Stop after: review, question, hypotheses, methodology, design, or experiment."),
    limit: int = typer.Option(20, "--limit", min=1, max=100),
    top_k: int = typer.Option(5, "--top-k", min=1, max=15),
    max_cost_usd: float | None = typer.Option(None, "--max-cost-usd", min=0.0, help="Cumulative project provider-cost hard stop."),
    max_revisions: int = typer.Option(0, "--max-revisions", min=0, max=5, help="Bounded adaptive revision cycles after an authorized experiment (0 = off)."),
) -> None:
    """Advance one project through review → question → hypotheses → methodology → design → experiment.

    Completed stages are reused, not repeated: re-invoke with --project to
    resume. Execution still requires an exact-specification approval; without
    one the command emits the execution package and the missing requirement.
    """
    if not seed and not project:
        raise typer.BadParameter("Provide --seed for a new project or --project to resume.")
    _run(lambda container: container.workflow.run(
        seed=seed or None,
        project_id=project or None,
        until=until,
        limit=limit,
        top_k=top_k,
        max_cost_usd=max_cost_usd,
        max_total_tokens=max_tokens,
        max_revisions=max_revisions,
    ))


@app.command()
def resume(
    run: str = typer.Option(..., "--run", help="Interrupted or partially failed run ID."),
) -> None:
    """Resume a discovery or review run from durable checkpoints."""
    async def operation(container: ApplicationContainer):
        stored = container.repository.get_run(run)
        if not stored:
            raise ValueError(f"Pipeline run '{run}' was not found.")
        if stored.kind is PipelineKind.DISCOVERY:
            return await container.discovery.resume(run)
        if stored.kind is PipelineKind.REVIEW:
            return await container.review.resume(run)
        raise ValueError(f"Pipeline kind '{stored.kind.value}' is not resumable in this milestone.")
    _run(operation)


@audit_app.command("project")
def audit_project(
    project: str = typer.Option(..., "--project", help="Project ID to validate."),
    run: str | None = typer.Option(None, "--run", help="Optional run ID to associate with the validation artifact."),
) -> None:
    """Validate provenance, evidence, usage, recovery, and secret redaction."""
    async def _operation(container) -> dict[str, Any]:
        return {"project": project, "validation_path": str(write_audit(container.repository, project, run_id=run))}
    _run(_operation)


@audit_app.command("run")
def audit_run(
    run_id: str = typer.Argument(..., metavar="RUN_ID"),
    json_output: bool = typer.Option(False, "--json"),
) -> None:
    """Emit claim/value/provenance/source verification records for one run."""
    async def operation(container: ApplicationContainer) -> dict[str, Any]:
        run = container.repository.get_run(run_id)
        if not run:
            raise ValueError(f"run_not_found: {run_id}")
        claims: list[dict[str, Any]] = []
        for artifact in container.repository.rows("SELECT id,artifact_type,path,source_evidence_json,checksum,created_at FROM artifacts WHERE run_id=?", (run_id,)):
            claims.append({"claim": artifact["artifact_type"], "value": artifact["path"], "provenance": "EXTRACTED_FROM_SOURCE", "source_artifact": artifact["id"], "source_run": run_id, "verification": "checksum_recorded"})
        for analysis in container.repository.rows("SELECT id,hypothesis_support_assessment,provenance_json FROM statistical_analyses WHERE run_id=?", (run_id,)):
            claims.append({"claim": "experiment_decision", "value": analysis["hypothesis_support_assessment"], "provenance": "DERIVED", "source_artifact": analysis["id"], "source_run": run_id, "verification": "deterministic_gates_required"})
        for report in container.repository.rows("SELECT id,decision,decision_evidence FROM experiment_reports WHERE run_id=?", (run_id,)):
            claims.append({"claim": "experiment_report", "value": report["decision"], "provenance": "DERIVED", "source_artifact": report["id"], "source_run": run_id, "verification": report["decision_evidence"]})
        return {"run_id": run_id, "kind": run.kind.value, "status": run.status.value, "claims": claims}
    _run(operation)


def _run(operation: Any) -> None:
    async def execute() -> Any:
        container = ApplicationContainer(load_settings())
        try:
            return await operation(container)
        finally:
            await container.close()
    try:
        result = asyncio.run(execute())
    except Exception as exc:
        typer.echo(json.dumps({"error": redact_text(str(exc)), "type": type(exc).__name__}, indent=2), err=True)
        raise typer.Exit(1) from exc
    typer.echo(json.dumps(asdict(result) if hasattr(result, "__dataclass_fields__") else result, indent=2, default=str))


app.add_typer(skill_app)
app.add_typer(audit_app, name="audit")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
