"""Filesystem and provider configuration derived from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
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
    # explicit --max-tokens / --max-cost-usd: every project stays bounded.
    default_max_tokens: int | None = None
    default_max_cost_usd: float | None = None


def _positive_int(value: str | None) -> int | None:
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


def load_settings() -> Settings:
    data_root = Path(os.getenv("AUTORESEARCH_DATA_ROOT", ".autoresearch/data")).expanduser().resolve()
    config = Path(os.getenv("AUTORESEARCH_PROVIDER_CONFIG", ".autoresearch/config.toml")).expanduser().resolve()
    sources = tuple(item.strip().lower() for item in os.getenv("AUTORESEARCH_SEARCH_SOURCES", "semantic_scholar,openalex,crossref,arxiv,openreview").split(",") if item.strip())
    return Settings(
        data_root=data_root,
        database_path=Path(os.getenv("AUTORESEARCH_DATABASE_PATH", str(data_root / "autoresearch.sqlite3"))).expanduser().resolve(),
        artifact_root=Path(os.getenv("AUTORESEARCH_ARTIFACT_ROOT", str(data_root / "artifacts"))).expanduser().resolve(),
        download_root=Path(os.getenv("AUTORESEARCH_DOWNLOAD_ROOT", str(data_root / "downloads"))).expanduser().resolve(),
        provider_config=config,
        search_sources=sources,
        agent_roots=tuple(Path(item).expanduser().resolve() for item in os.getenv("AUTORESEARCH_AGENT_ROOTS", str(Path.cwd())).split(os.pathsep) if item.strip()),
        default_max_tokens=_positive_int(os.getenv("AUTORESEARCH_DEFAULT_MAX_TOKENS")),
        default_max_cost_usd=_positive_float(os.getenv("AUTORESEARCH_DEFAULT_MAX_COST_USD")),
    )
