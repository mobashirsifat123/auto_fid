from __future__ import annotations

import hashlib
import json
import socket
import time
import zipfile
from pathlib import Path

import pytest

from autoresearch.release import (
    ProductConfig,
    ProductPaths,
    create_diagnostics,
    activate_staged,
    rollback_update,
    start_daemon,
    stage_update,
    stop_daemon,
    verify_release_artifact,
    write_config,
)
from autoresearch.versioning import API_VERSION, DATABASE_SCHEMA_VERSION, PRODUCT_VERSION


def _paths(tmp_path: Path) -> ProductPaths:
    return ProductPaths(
        config_dir=tmp_path / "config",
        data_dir=tmp_path / "data",
        state_dir=tmp_path / "state",
        cache_dir=tmp_path / "cache",
        install_dir=tmp_path / "install" / "autoresearch",
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def test_product_configuration_round_trips_without_credentials(tmp_path: Path):
    paths = _paths(tmp_path)
    config = ProductConfig.from_mapping(
        {
            "daemon_port": 8123,
            "allowed_repository_roots": [str(tmp_path)],
            "authentication_token_environment_variable": "AUTORESEARCH_BACKEND_TOKEN",
        }
    )
    updated = config.with_updates(allowed_repository_roots=(str(tmp_path / "workspace"),), cost_budget_usd=1.5)
    write_config(paths, updated)
    loaded = ProductConfig.load(paths)
    assert loaded.allowed_repository_roots == (str(tmp_path / "workspace"),)
    assert loaded.cost_budget_usd == 1.5
    assert "AUTORESEARCH_BACKEND_TOKEN" in paths.config_file.read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="credential"):
        ProductConfig.from_mapping({"api_key": "not-allowed"})
    with pytest.raises(ValueError, match="embedded credentials"):
        ProductConfig.from_mapping({"backend_url": "http://user:password@127.0.0.1:8000"})
    with pytest.raises(ValueError, match="provider credential"):
        ProductConfig.from_mapping({"authentication_token_environment_variable": "OPENAI_API_KEY"})


def test_daemon_starts_with_versioned_health_and_stops_cleanly(tmp_path: Path):
    paths = _paths(tmp_path)
    port = _free_port()
    config = ProductConfig(
        backend_url=f"http://127.0.0.1:{port}",
        daemon_port=port,
        allowed_repository_roots=(str(tmp_path),),
        daemon_persistence="stop_on_exit",
    )
    try:
        health, started = start_daemon(config, paths, timeout_seconds=12, repository_root=tmp_path)
        assert started
        assert health["status"] == "ready"
        assert health["product_version"] == PRODUCT_VERSION
        assert health["api_version"] == API_VERSION
        assert health["database_schema_version"] == DATABASE_SCHEMA_VERSION
        second_health, second_started = start_daemon(config, paths, repository_root=tmp_path)
        assert second_health == health
        assert not second_started
    finally:
        assert stop_daemon(paths)
        for _ in range(30):
            if not paths.pid_file.exists():
                break
            time.sleep(0.02)


def test_diagnostics_and_release_staging_redact_and_verify(tmp_path: Path, monkeypatch):
    paths = _paths(tmp_path)
    token = "diagnostic-token-value"
    monkeypatch.setenv("AUTORESEARCH_BACKEND_TOKEN", token)
    paths.create()
    paths.daemon_log.write_text(f"Authorization: Bearer {token}\n", encoding="utf-8")
    config = ProductConfig(authentication_token_environment_variable="AUTORESEARCH_BACKEND_TOKEN")
    archive = create_diagnostics(config, paths, tmp_path / "diagnostics.zip")
    with zipfile.ZipFile(archive) as contents:
        text = "\n".join(contents.read(name).decode("utf-8", errors="replace") for name in contents.namelist())
    assert token not in text
    assert "[REDACTED]" in text

    release = tmp_path / "release"
    release.mkdir()
    payload = release / "autoresearch-tui"
    payload.write_text("safe local payload", encoding="utf-8")
    manifest = {"product_version": PRODUCT_VERSION, "files": {payload.name: hashlib.sha256(payload.read_bytes()).hexdigest()}}
    (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    assert verify_release_artifact(release)["product_version"] == PRODUCT_VERSION
    staged = stage_update(release, paths)
    assert (staged / payload.name).read_text(encoding="utf-8") == "safe local payload"
    payload.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        verify_release_artifact(release)


def test_verified_tui_activation_retains_previous_release_for_rollback(tmp_path: Path):
    paths = _paths(tmp_path)
    paths.install_dir.mkdir(parents=True)
    (paths.install_dir / "autoresearch-tui").write_text("old", encoding="utf-8")
    release = tmp_path / "release-with-tui"
    release.mkdir()
    tui = release / "autoresearch-tui"
    tui.write_text("new", encoding="utf-8")
    manifest = {"product_version": PRODUCT_VERSION, "files": {tui.name: hashlib.sha256(tui.read_bytes()).hexdigest()}}
    (release / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    target, previous = activate_staged(release, paths)
    assert previous is not None
    assert target.read_text(encoding="utf-8") == "new"
    rollback_update(paths)
    assert target.read_text(encoding="utf-8") == "old"


def test_stale_pid_file_does_not_signal_unrelated_process(tmp_path: Path):
    paths = _paths(tmp_path)
    paths.create()
    paths.pid_file.write_text(json.dumps({"pid": __import__("os").getpid()}), encoding="utf-8")
    assert stop_daemon(paths)
    assert not paths.pid_file.exists()
