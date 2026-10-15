"""Deterministic intent routing with conservative scientific activation."""

from __future__ import annotations

import re

from .models import AgentIntent, RoutingDecision, SessionMode


OVERRIDES = {
    "@discover": SessionMode.STUDY_SEARCH,
    "@import": SessionMode.STUDY_SEARCH,
    "@review": SessionMode.LITERATURE_REVIEW,
    "@invent": SessionMode.NOVELTY,
    "@experiment": SessionMode.EXPERIMENT,
}

LOCAL_IMPORT_OVERRIDE = {"@import"}


class AgentIntentRouter:
    """Classify intent without granting permission or starting a pipeline."""

    threshold = 0.72

    def route(self, message: str, *, mode: SessionMode, auto_routing: bool, override: str | None = None) -> RoutingDecision:
        explicit_mode = self._explicit_mode(message, override)
        selected = explicit_mode or (mode if mode is not SessionMode.DEFAULT else None)
        if selected:
            return self._selected(selected, message)

        inferred = self._infer(message)
        if not auto_routing or inferred.confidence < self.threshold or inferred.intent in {AgentIntent.GENERAL, AgentIntent.CODING}:
            return RoutingDecision(inferred.intent, None, inferred.confidence, inferred.reason)
        return inferred

    def strip_override(self, message: str) -> str:
        trimmed = message.strip()
        first = trimmed.split(maxsplit=1)[0].lower() if trimmed else ""
        return trimmed[len(first):].strip() if first in OVERRIDES else trimmed

    def _explicit_mode(self, message: str, override: str | None) -> SessionMode | None:
        if override:
            return SessionMode(override)
        first = message.strip().split(maxsplit=1)[0].lower() if message.strip() else ""
        return OVERRIDES.get(first)

    def _selected(self, mode: SessionMode, message: str) -> RoutingDecision:
        # Detect local import intent (catalog path provided)
        first = message.strip().split(maxsplit=1)[0].lower() if message.strip() else ""
        if first == "@import" and re.search(r"(?:paper_catalog|\.json)", message, re.I):
            return RoutingDecision(AgentIntent.STUDY_SEARCH, "local_import", 1.0, "Explicit local catalog import.", ("catalog_path",), False, True)

        mapping = {
            SessionMode.STUDY_SEARCH: (AgentIntent.STUDY_SEARCH, "study_search", ("seed_paper_or_query",), False),
            SessionMode.LITERATURE_REVIEW: (AgentIntent.LITERATURE_REVIEW, "literature_review", ("paper_set",), False),
            SessionMode.NOVELTY: (AgentIntent.NOVELTY, "novelty_generation", ("validated_gap",), False),
            SessionMode.EXPERIMENT: (AgentIntent.EXPERIMENT_DESIGN, "experiment_design", ("novelty_candidate",), False),
        }
        intent, capability, required, confirmation = mapping[mode]
        if mode is SessionMode.EXPERIMENT and re.search(r"\b(run|execute|launch)\b", message, re.I):
            intent, capability, required = AgentIntent.EXPERIMENT_EXECUTION, "experiment_execution_local", ("approved_experiment_spec",)
        return RoutingDecision(intent, capability, 1.0, f"Explicit {mode.value} activation.", required, confirmation, True)

    def _infer(self, message: str) -> RoutingDecision:
        text = message.lower()
        if re.search(r"\b(reinvent|replacement method|invent a replacement|why (?:the )?experiment failed)\b", text):
            return RoutingDecision(AgentIntent.REINVENTION, "result_conditioned_reinvention", 0.94, "The request asks for result-conditioned method replacement.", ("experiment_result",), True)
        if re.search(r"\b(run|execute|launch)\b.*\b(seed|screening|experiment|ablation)\b", text):
            return RoutingDecision(AgentIntent.EXPERIMENT_EXECUTION, "experiment_execution_local", 0.91, "The request asks to execute a scientific experiment.", ("approved_experiment_spec",), True)
        if re.search(r"\b(design|plan|cheapest|discriminating)\b.*\b(experiment|ablation|evaluation)\b", text):
            return RoutingDecision(AgentIntent.EXPERIMENT_DESIGN, "experiment_design", 0.89, "The request asks for a scientific experiment design.", ("novelty_candidate",), True)
        if re.search(r"\b(already known|already exists|prior art|novel|novelty|mechanism already)\b", text):
            return RoutingDecision(AgentIntent.NOVELTY, "novelty_generation", 0.88, "The request asks for prior-art or novelty assessment.", ("validated_gap",), True)
        if re.search(r"\b(compare|synthesize|deep review|literature review|counter-evidence)\b.*\b(papers?|studies|literature)\b", text):
            return RoutingDecision(AgentIntent.LITERATURE_REVIEW, "literature_review", 0.86, "The request asks for evidence synthesis across papers.", ("paper_set",), True)
        if re.search(r"\b(find|search|recent|discover)\b.*\b(papers?|studies|literature)\b", text):
            return RoutingDecision(AgentIntent.STUDY_SEARCH, "study_search", 0.84, "The request asks for scholarly discovery.", ("seed_paper_or_query",), True)
        if re.search(r"\b(class|function|repository|code|test|bug|api|diff|file|build|crash|traceback)\b", text):
            return RoutingDecision(AgentIntent.CODING, None, 0.82, "The request is a normal repository or coding task.")
        return RoutingDecision(AgentIntent.GENERAL, None, 0.55, "No specialized scientific intent is sufficiently clear.")
