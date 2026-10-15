from typer.testing import CliRunner
from pathlib import Path

from autoresearch.cli.app import app


def test_cli_exposes_required_commands():
    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "discover" in result.stdout
    assert "review" in result.stdout
    assert "resume" in result.stdout
    discover_help = runner.invoke(app, ["discover", "--help"]).stdout
    review_help = runner.invoke(app, ["review", "--help"]).stdout
    assert "--seed-paper" in discover_help and "--max-runtime-seconds" in discover_help
    assert "--top-k" in review_help and "--hierarchical" in review_help and "--max-cost-usd" in review_help
    assert "--run" in runner.invoke(app, ["resume", "--help"]).stdout


def test_product_commands_are_exposed_without_raw_secret_input():
    runner = CliRunner()
    help_text = runner.invoke(app, ["--help"]).stdout
    for command in ("version", "setup", "login", "doctor", "serve", "status", "diagnostics", "update", "rollback", "uninstall", "run", "agent"):
        assert command in help_text
    assert "--auth-token-env" in runner.invoke(app, ["login", "--help"]).stdout
    assert "api-key" not in runner.invoke(app, ["setup", "--help"]).stdout.lower()


def test_claude_code_research_skills_are_available():
    from autoresearch.cli.skills import _canonical_root

    canonical = _canonical_root()
    expected = {"deep-research": "deep-research", "novelty": "novelty", "experiment": "experiment"}
    for name, command in expected.items():
        content = (canonical / name / "SKILL.md").read_text(encoding="utf-8")
        # Bridge skills call the stable skill CLI, never the conversational agent.
        assert "agent --mode" not in content
        assert "skill " + command in content
        assert "Exit codes" in content


def test_skill_cli_classifies_not_found_exit_codes(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTORESEARCH_DATA_ROOT", str(tmp_path / "data"))
    runner = CliRunner()
    result = runner.invoke(app, ["skill", "novelty", "--project", "missing-project"])
    assert result.exit_code == 4
    assert "project_not_found" in (result.stdout + result.stderr)
    result2 = runner.invoke(app, ["skill", "experiment", "plan", "--candidate-id", "missing-candidate"])
    assert result2.exit_code == 4
    assert "novelty_candidate_not_found" in (result2.stdout + result2.stderr)


def test_skill_cli_experiment_plan_requires_candidate(monkeypatch, tmp_path):
    monkeypatch.setenv("AUTORESEARCH_DATA_ROOT", str(tmp_path / "data"))
    runner = CliRunner()
    result = runner.invoke(app, ["skill", "experiment", "plan"])
    assert result.exit_code == 2, "missing required --candidate-id is a usage error"


def test_no_divergent_skill_copies_remain():
    root = Path(__file__).resolve().parents[1]
    repo_copies = root / ".claude" / "skills"
    assert not (repo_copies / "deep-research").exists()
    assert not (repo_copies / "novelty").exists()
    assert not (repo_copies / "experiment").exists()
    assert not (repo_copies / "review").exists(), "divergent review skill must be removed"


def test_audit_project_command_runs(tmp_path, monkeypatch):
    """Regression: the audit command passed a plain dict to the async runner
    and crashed with "'dict' object can't be awaited"."""
    import asyncio

    from autoresearch.application.audit import write_audit
    from autoresearch.storage.repository import ResearchRepository

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTORESEARCH_DATA_ROOT", str(tmp_path / "cli-state"))
    repository = ResearchRepository(tmp_path / "cli-state" / "autoresearch.sqlite3", tmp_path / "cli-state" / "artifacts")
    repository.migrate()
    project = repository.create_project("Audit CLI")

    from typer.testing import CliRunner
    from autoresearch.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["audit", "project", "--project", project.id])
    assert result.exit_code == 0, result.output
    assert "validation_path" in result.output


def test_resume_command_reports_unknown_run_cleanly(tmp_path, monkeypatch):
    """The resume CLI wrapper must emit one JSON error document, not a traceback."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AUTORESEARCH_DATA_ROOT", str(tmp_path / "state"))

    from typer.testing import CliRunner
    from autoresearch.cli.app import app

    runner = CliRunner()
    result = runner.invoke(app, ["resume", "--run", "nonexistent"])
    assert result.exit_code == 1
    assert "was not found" in result.output
    assert result.output.strip().startswith("{")
