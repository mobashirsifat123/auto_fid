"""Durable run-status projection for the future HTTP API."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PipelineRunStatus:
    run_id: str
    project_id: str
    status: str
    artifact_ids: tuple[str, ...]
