"""Source-of-truth registry for general and scientific capabilities."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from autoresearch.providers.capabilities import ModelRole

from .models import CapabilityDescriptor, CapabilityState


def _schema(properties: dict[str, Any], required: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": list(required), "additionalProperties": False}


class CapabilityRegistry:
    def __init__(self, container: Any) -> None:
        self.container = container

    def descriptors(self) -> tuple[CapabilityDescriptor, ...]:
        assigned = self.container.router.assignments
        chat_ready = any(role in assigned for role in (ModelRole.CHAT, ModelRole.CODING, ModelRole.RESEARCH_REASONING))
        worker_ready = bool(getattr(self.container, "worker", None))
        common_chat_reason = None if chat_ready else "No chat, coding, or research-reasoning model role is configured."
        return (
            CapabilityDescriptor("general_agent_chat", "Default Agent", "General conversational coding and repository assistance.", CapabilityState.ENABLED if chat_ready else CapabilityState.DISABLED, "default", ("message",), ("ModelRouter",), (), "model-token usage", False, common_chat_reason, (), _schema({"message": {"type": "string"}}, ("message",))),
            CapabilityDescriptor("repository_file_tools", "Repository/File Tools", "Contained directory listing, file reading, code search, and approval-bound edits.", CapabilityState.ENABLED, "available in default agent", ("active_repository",), ("AgentToolRegistry",), ("repository_read", "file_modify"), "local only", True, None, ("/tools",), _schema({"path": {"type": "string"}})),
            CapabilityDescriptor("terminal_tools", "Policy-controlled Terminal", "Short-lived allowlisted development and test commands.", CapabilityState.CONDITIONAL, "agent tool request plus permission", ("command_vector",), ("GenericTerminalExecutor",), ("local_command",), "local runtime", True, "Each command requires a scoped permission; GPU and long scientific runs are excluded.", ("/tools",), _schema({"command": {"type": "string"}, "arguments": {"type": "array", "items": {"type": "string"}}}, ("command",))),
            CapabilityDescriptor("study_search", "Study Search", "Discover and deduplicate recent studies and resolve legal PDFs. Use --from-catalog to import local papers.", CapabilityState.ENABLED, "explicit mode, direct shortcut, or confirmed suggestion", ("seed_paper_or_query",), ("DiscoveryService",), ("research_provider_calls",), "metadata calls plus configured model usage", False, None, ("/use study-search", "/discover", "@discover"), _schema({"seed_paper": {"type": "string"}, "limit": {"type": "integer", "minimum": 1, "maximum": 100}}, ("seed_paper",))),
            CapabilityDescriptor("local_import", "Local Paper Import", "Import papers from a local JSON catalog without external APIs or providers.", CapabilityState.ENABLED, "explicit activation", ("catalog_path",), ("ResearchRepository", "DiscoveryService"), (), "local only; no model usage", False, None, ("/import", "@import"), _schema({"catalog_path": {"type": "string"}}, ("catalog_path",))),
            CapabilityDescriptor("literature_review", "Deep Literature Review", "Evidence-verified structured review, ranking, and cited gap analysis. Requires a paper set (from discover, local_import, or --project).", CapabilityState.ENABLED, "explicit mode, direct shortcut, or confirmed suggestion", ("paper_set",), ("LiteratureReviewService",), ("research_provider_calls",), "model and embedding usage", False, None, ("/use review", "/review", "@review"), _schema({"top_k": {"type": "integer", "minimum": 1, "maximum": 15}})),
            CapabilityDescriptor("novelty_generation", "Novelty Generation", "Generate and criticize candidate mechanisms against prior art. Requires a gap (from review or --gap-file).", CapabilityState.ENABLED, "explicit activation or confirmed suggestion", ("validated_gap",), ("NoveltyService",), ("research_provider_calls",), "search and model usage", True, None, ("/use novelty", "/invent", "@invent"), _schema({"gap_id": {"type": "string"}})),
            CapabilityDescriptor("experiment_design", "Experiment Design", "Create a declarative, falsifiable ExperimentSpec. Requires a candidate (from novelty or --candidate-file).", CapabilityState.ENABLED, "explicit activation or confirmed suggestion", ("novelty_candidate",), ("ExperimentService",), ("experiment_planning",), "model usage; no execution", True, None, ("/use experiment", "/experiment plan", "@experiment"), _schema({"candidate_id": {"type": "string"}}, ("candidate_id",))),
            CapabilityDescriptor("experiment_execution_local", "Local Experiment Execution", "Execute an exact approved typed action plan in the configured workspace.", CapabilityState.CONDITIONAL if worker_ready else CapabilityState.DISABLED, "exact specification approval", ("approved_experiment_spec",), ("LocalTypedWorker",), ("experiment_execution",), "approval-bound local compute", True, "Workspace-contained typed worker; not an OS sandbox." if worker_ready else "No policy-controlled local worker is configured.", ("/experiment execute", "/approve"), _schema({"candidate_id": {"type": "string"}, "specification_checksum": {"type": "string"}}, ("candidate_id", "specification_checksum")), True),
            CapabilityDescriptor("experiment_execution_remote", "Remote GPU Execution", "Run approved scientific jobs on a configured remote worker.", CapabilityState.DISABLED, "explicit approval and configured worker", ("approved_experiment_spec", "remote_worker"), ("RemoteWorker",), ("remote_worker_execution",), "remote GPU and API cost", True, "Remote GPU workers are not configured or implemented.", (), _schema({}), True),
            CapabilityDescriptor("result_conditioned_reinvention", "Result-Conditioned Reinvention", "Diagnose experiment evidence and propose the next discriminating method.", CapabilityState.ENABLED, "explicit activation or confirmed suggestion", ("experiment_result",), ("ReinventionService",), ("research_provider_calls",), "model usage", True, None, ("/use reinvention", "/reinvent", "@reinvent"), _schema({"candidate_id": {"type": "string"}}, ("candidate_id",))),
        )

    def get(self, capability_id: str) -> CapabilityDescriptor:
        value = next((item for item in self.descriptors() if item.id == capability_id), None)
        if value is None:
            raise ValueError(f"Unknown capability '{capability_id}'.")
        return value

    def payload(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self.descriptors()]
