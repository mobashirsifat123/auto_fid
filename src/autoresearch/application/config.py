"""Filesystem and provider configuration derived from environment variables."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_root: Path
    database_path: Path
    artifact_root: Path
    download_root: Path
    provider_config: Path
    search_sources: tuple[str, ...]
    agent_roots: tuple[Path, ...] = ()
    # Project-budget defaults applied by the guard when a project has no
    # explicit --max-tokens / --max-cost-usd. None leaves that dimension uncapped.
    default_max_tokens: int | None = None
    default_max_cost_usd: float | None = None
    # "deterministic" (default): manuscripts are field-assembled, zero model
    # calls. "model": model-assisted scientific argument gated by the
    # deterministic numeric/citation/overclaim audit (falls back on any
    # surviving violation).
    manuscript_writer: str = "deterministic"
    # When true, the skills follow the OpenCode session's own model provider
    # (auto-detected from the session credential store) instead of the
    # provider-config file. Also used automatically when no config exists.
    auto_provider: bool = False
    # Quality-width knobs: parallel expansion adds tokens (width), not
    # wall-clock (depth). All defaults are ON; every one degrades gracefully.
    discovery_query_perspectives: int = 3
    discovery_citation_expansion: bool = True
    discovery_citation_anchors: int = 5
    discovery_citation_neighbors: int = 10
    review_ensemble: bool = True


def _positive_int(value: str | int | None) -> int | None:
    try:
        parsed = int(str(value or "").strip())
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _positive_float(value: str | None) -> float | None:
    try:
        parsed = float(str(value or "").strip())
        return parsed if parsed > 0 else None
    except (TypeError, ValueError):
        return None


def _boolean(value: object) -> bool | None:
    text = str(value or "").strip().lower()
    if text in ("1", "true", "yes", "on"):
        return True
    if text in ("0", "false", "no", "off"):
        return False
    return None


def load_settings() -> Settings:
    data_root = Path(os.getenv("AUTORESEARCH_DATA_ROOT", ".autoresearch/data")).expanduser().resolve()
    config = Path(os.getenv("AUTORESEARCH_PROVIDER_CONFIG", ".autoresearch/config.toml")).expanduser().resolve()
    research = {}
    if config.is_file():
        with config.open("rb") as handle:
            research = tomllib.load(handle).get("research", {})
    sources = tuple(item.strip().lower() for item in os.getenv("AUTORESEARCH_SEARCH_SOURCES", "semantic_scholar,openalex,crossref,arxiv,openreview").split(",") if item.strip())
    citation_expansion = _boolean(research.get("discovery_citation_expansion", os.getenv("AUTORESEARCH_DISCOVERY_CITATION_EXPANSION")))
    review_ensemble = _boolean(research.get("review_ensemble", os.getenv("AUTORESEARCH_REVIEW_ENSEMBLE")))
    return Settings(
        data_root=data_root,
        database_path=Path(os.getenv("AUTORESEARCH_DATABASE_PATH", str(data_root / "autoresearch.sqlite3"))).expanduser().resolve(),
        artifact_root=Path(os.getenv("AUTORESEARCH_ARTIFACT_ROOT", str(data_root / "artifacts"))).expanduser().resolve(),
        download_root=Path(os.getenv("AUTORESEARCH_DOWNLOAD_ROOT", str(data_root / "downloads"))).expanduser().resolve(),
        provider_config=config,
        search_sources=sources,
        agent_roots=tuple(Path(item).expanduser().resolve() for item in os.getenv("AUTORESEARCH_AGENT_ROOTS", str(Path.cwd())).split(os.pathsep) if item.strip()),
        # A project config can explicitly request no token ceiling (0), even
        # when the launching shell inherited a finite global default.
        default_max_tokens=_positive_int(research.get("max_tokens", os.getenv("AUTORESEARCH_DEFAULT_MAX_TOKENS"))),
        default_max_cost_usd=_positive_float(os.getenv("AUTORESEARCH_DEFAULT_MAX_COST_USD")),
        manuscript_writer=str(research.get("manuscript_writer", os.getenv("AUTORESEARCH_MANUSCRIPT_WRITER", "deterministic"))).strip().lower() or "deterministic",
        auto_provider=bool(str(research.get("auto_provider", os.getenv("AUTORESEARCH_AUTO_PROVIDER", "false"))).strip().lower() in ("1", "true", "yes", "on")),
        discovery_query_perspectives=_positive_int(research.get("discovery_query_perspectives", os.getenv("AUTORESEARCH_DISCOVERY_QUERY_PERSPECTIVES"))) or 3,
        discovery_citation_expansion=True if citation_expansion is None else citation_expansion,
        discovery_citation_anchors=_positive_int(research.get("discovery_citation_anchors")) or 5,
        discovery_citation_neighbors=_positive_int(research.get("discovery_citation_neighbors")) or 10,
        review_ensemble=True if review_ensemble is None else review_ensemble,
    )


def resolve_settings(project_id: str | None = None) -> Settings:
    """Settings anchored to a project's recorded data root.

    Projects remember the directory their state lives in (recorded at
    creation). Resuming with --project therefore works from ANY working
    directory: the command re-opens the database where the project actually
    lives instead of silently starting a fresh one.
    """
    from pathlib import Path as _Path

    settings = load_settings()
    if not project_id:
        return settings
    root_raw: str | None = None
    # 1) machine-local registry: works from any working directory.
    registry_path = _Path(os.environ.get(
        "AUTORESEARCH_REGISTRY_PATH", str(_Path.home() / ".autoresearch" / "project-registry.json"),
    )).expanduser()
    try:
        if registry_path.is_file():
            payload = json.loads(registry_path.read_text(encoding="utf-8"))
            root_raw = str(payload.get(str(project_id))) if isinstance(payload, dict) and payload.get(str(project_id)) else None
    except (OSError, ValueError):
        root_raw = None
    # 2) fallback: the project row in the current default database. If
    # found there, lazily register it so the NEXT resume works from anywhere.
    if not root_raw:
        from autoresearch.storage.repository import ResearchRepository, _register_project_root

        probe = ResearchRepository(settings.database_path, settings.artifact_root)
        row = probe.row("SELECT state_db_path FROM projects WHERE id=?", (project_id,))
        if row:
            root_raw = str(row.get("state_db_path") or settings.database_path)
            try:
                _register_project_root(str(project_id), root_raw)
            except Exception:
                pass
    if not root_raw:
        return settings
    database = _Path(root_raw).expanduser().resolve()
    if database == settings.database_path.resolve():
        return settings
    from dataclasses import replace

    return replace(
        settings,
        data_root=database.parent,
        database_path=database,
        artifact_root=database.parent / "artifacts",
        download_root=database.parent / "downloads",
    )
