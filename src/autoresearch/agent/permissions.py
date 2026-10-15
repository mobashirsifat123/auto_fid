"""Scoped permission checks for agent tools."""

from __future__ import annotations

from .models import AgentSession, PermissionDecision, ToolDefinition
from .store import AgentStore


class AgentPermissionService:
    def __init__(self, store: AgentStore) -> None:
        self.store = store

    def check(self, session: AgentSession, turn_id: str, tool: ToolDefinition, arguments: dict[str, object]) -> PermissionDecision:
        if not tool.mutating:
            return PermissionDecision(True)
        if self.store.permission_allows(session.id, tool.id, tool.permission, session.active_repository):
            return PermissionDecision(True)
        permission = self.store.create_permission(session.id, turn_id, tool.id, tool.permission, session.active_repository, tool.id, arguments)
        return PermissionDecision(False, permission["id"], f"{tool.display_name if hasattr(tool, 'display_name') else tool.id} requires {tool.permission} permission.")
