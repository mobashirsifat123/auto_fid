"""Provider-routed structured seed-paper analysis."""

from __future__ import annotations

from dataclasses import dataclass

from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.router import ModelRouter

from .document_parser import ParsedDocument

SEED_PROMPT_VERSION = "seed-analysis-v1"
_STRING_ARRAY = {"type": "array", "items": {"type": "string"}}
SEED_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "topic": {"type": "string"},
        "research_question": {"type": "string"},
        "methods": _STRING_ARRAY,
        "datasets": _STRING_ARRAY,
        "citations": _STRING_ARRAY,
        "limitations": _STRING_ARRAY,
        "keywords": _STRING_ARRAY,
    },
    "required": ["title", "topic", "research_question", "methods", "datasets", "citations", "limitations", "keywords"],
    "additionalProperties": False,
}


@dataclass(frozen=True)
class SeedAnalysis:
    title: str
    topic: str
    research_question: str
    methods: tuple[str, ...]
    datasets: tuple[str, ...]
    citations: tuple[str, ...]
    limitations: tuple[str, ...]
    keywords: tuple[str, ...]


class SeedAnalyzer:
    def __init__(self, router: ModelRouter) -> None:
        self.router = router

    async def analyze(self, document: ParsedDocument, *, project_id: str, run_id: str) -> SeedAnalysis:
        source = document.text[:40_000] or f"Identifier: {document.doi or document.url or document.input_value}"
        payload = await self.router.structured_output(
            ModelRole.LITERATURE_EXTRACTION,
            messages=[
                {"role": "system", "content": "Extract only information supported by the supplied seed paper. Return one JSON object."},
                {"role": "user", "content": f"Prompt version: {SEED_PROMPT_VERSION}\nTitle hint: {document.title}\n\n{source}\n\nReturn: title, topic, research_question, methods[], datasets[], citations[], limitations[], keywords[]."},
            ],
            schema={"name": "seed_analysis", "schema": SEED_SCHEMA},
            project_id=project_id,
            run_id=run_id,
            operation="seed_analysis",
        )
        return SeedAnalysis(
            title=_string(payload.get("title")) or document.title,
            topic=_required(payload, "topic"),
            research_question=_required(payload, "research_question"),
            methods=_strings(payload.get("methods")), datasets=_strings(payload.get("datasets")),
            citations=_strings(payload.get("citations")), limitations=_strings(payload.get("limitations")), keywords=_strings(payload.get("keywords")),
        )


def _required(payload: dict, key: str) -> str:
    value = _string(payload.get(key))
    if not value:
        raise ValueError(f"Seed analysis field '{key}' is required.")
    return value


def _string(value: object) -> str:
    return " ".join(str(value or "").split())


def _strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for raw in value if (item := _string(raw)))
