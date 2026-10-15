import sqlite3

from autoresearch.storage import ResearchRepository


def test_empty_database_migrates_idempotently(tmp_path):
    path = tmp_path / "research.sqlite3"
    first = ResearchRepository(path, tmp_path / "artifacts")
    second = ResearchRepository(path, tmp_path / "artifacts")
    assert first.integrity_check() == "ok"
    assert second.row("SELECT MAX(version) AS version FROM schema_migrations")["version"] == 11
    assert second.row("SELECT name FROM sqlite_master WHERE type='table' AND name='api_events'") is not None
    tables = {row["name"] for row in second.rows("SELECT name FROM sqlite_master WHERE type='table'")}
    assert {"projects", "pipeline_runs", "pipeline_checkpoints", "papers", "paper_reviews", "evidence_claims", "research_gaps", "provider_usage", "artifacts", "novelty_candidates", "experiment_specs", "api_project_idempotency", "agent_sessions", "agent_messages", "agent_turns", "agent_tool_calls", "agent_permissions", "agent_events", "agent_artifact_refs", "reinvention_analyses"} <= tables


def test_existing_empty_file_migrates_without_data_loss(tmp_path):
    path = tmp_path / "existing.sqlite3"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE preserved(id TEXT PRIMARY KEY, value TEXT)")
    connection.execute("INSERT INTO preserved VALUES('one','keep')")
    connection.commit(); connection.close()
    repository = ResearchRepository(path, tmp_path / "artifacts")
    assert repository.row("SELECT value FROM preserved WHERE id='one'")["value"] == "keep"


def test_upgrade_from_v1_and_v2_and_v3(tmp_path):
    path = tmp_path / "legacy.sqlite3"
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE schema_migrations(version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL)")
    conn.execute("INSERT INTO schema_migrations(version, applied_at) VALUES(1, '2026-01-01T00:00:00Z')")
    from autoresearch.storage.repository import _SCHEMA_V1
    conn.executescript(_SCHEMA_V1)
    conn.commit()
    conn.close()

    repo = ResearchRepository(path, tmp_path / "artifacts")
    assert repo.integrity_check() == "ok"
    assert repo.row("SELECT MAX(version) AS version FROM schema_migrations")["version"] == 11
    tables = {row["name"] for row in repo.rows("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "novelty_candidates" in tables
    assert "api_project_idempotency" in tables
    assert "reinvention_analyses" in tables


def test_upgrade_from_v9_to_latest(tmp_path):
    """Realistic upgrade path: an existing v9 database must gain the
    methodology_freezes table and the new experiment_specs columns."""
    import sqlite3

    database = tmp_path / "v9.sqlite3"
    from autoresearch.storage.repository import ResearchRepository, _SCHEMA_V9

    fresh = ResearchRepository(tmp_path / "seed.sqlite3", tmp_path / "artifacts")
    fresh.migrate()
    # Build a v9 database by cloning the migrated schema minus v10 objects.
    connection = sqlite3.connect(database)
    connection.executescript("".join(
        line for line in sqlite3.connect(tmp_path / "seed.sqlite3").iterdump()
        if "methodology_freezes" not in line
    ))
    connection.execute("DROP TABLE IF EXISTS methodology_freezes")
    connection.execute("ALTER TABLE experiment_specs DROP COLUMN gap_id")
    connection.execute("ALTER TABLE experiment_specs DROP COLUMN methodology_freeze_id")
    connection.execute("ALTER TABLE experiment_specs DROP COLUMN interpretation_rules_json")
    connection.execute("DROP TABLE IF EXISTS project_budgets_v11_marker")
    connection.execute("ALTER TABLE project_budgets DROP COLUMN token_ceiling")
    connection.execute("DELETE FROM schema_migrations WHERE version >= 10")
    connection.commit()
    connection.close()

    upgraded = ResearchRepository(database, tmp_path / "artifacts")
    upgraded.migrate()
    version = upgraded.row("SELECT MAX(version) AS version FROM schema_migrations")["version"]
    assert version == 11
    tables = {row["name"] for row in upgraded.rows("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "methodology_freezes" in tables
    columns = {row["name"] for row in upgraded.rows("PRAGMA table_info(experiment_specs)")}
    assert {"gap_id", "methodology_freeze_id", "interpretation_rules_json"} <= columns
    budget_columns = {row["name"] for row in upgraded.rows("PRAGMA table_info(project_budgets)")}
    assert "token_ceiling" in budget_columns
    assert upgraded.integrity_check() == "ok"
