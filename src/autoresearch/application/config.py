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
    )
