from __future__ import annotations

import pytest

from autoresearch.domain.models import PaperRecord
from autoresearch.providers.capabilities import ModelRole
from autoresearch.providers.config import ModelAssignment
from autoresearch.providers.registry import ProviderRegistry
from autoresearch.providers.router import ModelRouter
from autoresearch.providers.testing import FakeLLMProvider
from autoresearch.review.extraction import PaperReviewExtractor


@pytest.mark.asyncio
async def test_review_keeps_only_exact_source_excerpts():
    response = {
        "problem": "Robustness", "research_question": "Does adaptation help?", "method": "Adaptive fusion", "datasets": ["Dataset-A"],
        "experimental_protocol": "Matched evaluation", "results": ["Improved robustness"], "ablations": [], "limitations": ["One dataset"], "failure_modes": [], "confidence": 0.8,
        "evidence_spans": [
            {"claim_type": "result", "statement": "Improves robustness", "excerpt": "Adaptive fusion improves robustness.", "page": 1, "section": "Results", "confidence": 0.9},
            {"claim_type": "result", "statement": "Fabricated", "excerpt": "This sentence is absent.", "page": 1, "section": "Results", "confidence": 0.9},
        ],
    }
    provider = FakeLLMProvider(responses=[response, response, response])  # one per review perspective (rigor/reproducibility/novelty)
    registry = ProviderRegistry(); registry.register_llm("fake", provider)
    router = ModelRouter(registry, {ModelRole.LITERATURE_EXTRACTION: ModelAssignment(ModelRole.LITERATURE_EXTRACTION, "fake", "fake-model")})
    paper = PaperRecord("paper", "project", "discovery", "Paper", "Adaptive fusion improves robustness.", (), 2025, "2025-01-01", None, None, None, ("test",))
    bundle = await PaperReviewExtractor(router).extract(paper, run_id="review", chunks=(), fallback_text=paper.abstract)
    assert len(bundle.claims) == 1
    claim = bundle.claims[0]
    assert claim.excerpt == "Adaptive fusion improves robustness."
    assert claim.provenance.project_id == "project"
    assert claim.provenance.run_id == "review"
    assert claim.provenance.provider == "fake"
    assert len(claim.provenance.checksum) == 64
    assert bundle.review.evidence_claim_ids == (claim.id,)
