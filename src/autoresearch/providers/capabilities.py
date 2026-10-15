"""Provider capabilities and role requirements."""

from enum import StrEnum


class Capability(StrEnum):
    TEXT = "text_generation"
    STREAMING = "streaming"
    STRUCTURED_OUTPUT = "structured_output"
    TOOL_USE = "tool_use"
    VISION = "vision"
    LONG_CONTEXT = "long_context"
    REASONING = "reasoning"
    CODE = "code_generation"
    EMBEDDINGS = "embeddings"


class ModelRole(StrEnum):
    CHAT = "chat"
    LITERATURE_EXTRACTION = "literature_extraction"
    RESEARCH_REASONING = "research_reasoning"
    SCIENTIFIC_CRITIC = "scientific_critic"
    EXPERIMENT_PLANNING = "experiment_planning"
    CODING = "coding"
    RESULT_ANALYSIS = "result_analysis"
    EMBEDDINGS = "embeddings"


ROLE_REQUIREMENTS: dict[ModelRole, frozenset[Capability]] = {
    ModelRole.CHAT: frozenset({Capability.TEXT, Capability.STREAMING}),
    ModelRole.LITERATURE_EXTRACTION: frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}),
    ModelRole.RESEARCH_REASONING: frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT, Capability.REASONING}),
    ModelRole.SCIENTIFIC_CRITIC: frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}),
    ModelRole.EXPERIMENT_PLANNING: frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}),
    ModelRole.CODING: frozenset({Capability.TEXT, Capability.CODE, Capability.TOOL_USE}),
    ModelRole.RESULT_ANALYSIS: frozenset({Capability.TEXT, Capability.STRUCTURED_OUTPUT}),
    ModelRole.EMBEDDINGS: frozenset({Capability.EMBEDDINGS}),
}
