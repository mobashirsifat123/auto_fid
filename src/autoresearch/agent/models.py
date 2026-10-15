"""Provider-neutral domain models for conversational agent state."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from autoresearch.domain.models import utc_now


class CapabilityState(StrEnum):
    ENABLED = "enabled"
    CONDITIONAL = "conditional"
    DISABLED = "disabled"


class SessionMode(StrEnum):
    DEFAULT = "default"
    STUDY_SEARCH = "study_search"
    LITERATURE_REVIEW = "literature_review"
    NOVELTY = "novelty_generation"
    EXPERIMENT = "experiment"


class AgentIntent(StrEnum):
    GENERAL = "general"
    CODING = "coding"
    STUDY_SEARCH = "study_search"
    LITERATURE_REVIEW = "literature_review"
    NOVELTY = "novelty_generation"
    EXPERIMENT_DESIGN = "experiment_design"
    EXPERIMENT_EXECUTION = "experiment_execution"
    REINVENTION = "result_conditioned_reinvention"


class TurnStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    AWAITING_PERMISSION = "awaiting_permission"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class CapabilityDescriptor:
    id: str
    display_name: str
    description: str
    state: CapabilityState
    activation_policy: str
    required_inputs: tuple[str, ...]
    required_services: tuple[str, ...]
    required_permissions: tuple[str, ...]
    cost: str
    approval_required: bool
    reason: str | None
    supported_commands: tuple[str, ...]
    tool_schema: dict[str, Any]
    worker_required: bool = False


@dataclass(frozen=True)
class RoutingDecision:
    intent: AgentIntent
    recommended_capability: str | None
    confidence: float
    reason: str
    required_inputs: tuple[str, ...] = ()
    confirmation_required: bool = False
    explicit: bool = False


@dataclass(frozen=True)
class AgentSession:
    id: str
    project_id: str
    active_repository: str
    working_directory: str
    selected_mode: SessionMode = SessionMode.DEFAULT
    auto_routing: bool = False
    provider_role: str = "chat"
    token_budget: int = 32_000
    cost_budget: float | None = None
    active_run_id: str | None = None
    cancellation_requested: bool = False
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)


@dataclass(frozen=True)
class AgentTurn:
    id: str
    session_id: str
    message: str
    status: TurnStatus
    override_mode: str | None
    routing: RoutingDecision | None
    max_tool_iterations: int
    max_model_calls: int
    time_budget_seconds: float
    token_budget: int
    cost_budget: float | None
    model_calls: int = 0
    tool_iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    estimated_cost: float = 0.0
    error: str | None = None
    created_at: str = field(default_factory=utc_now)
    completed_at: str | None = None


@dataclass(frozen=True)
class ToolDefinition:
    id: str
    description: str
    permission: str
    mutating: bool
    schema: dict[str, Any]


@dataclass(frozen=True)
class ToolObservation:
    tool_id: str
    ok: bool
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PermissionDecision:
    allowed: bool
    permission_id: str | None = None
    reason: str | None = None
