"""Tests for exemplar-grounded manuscript calibration."""

from __future__ import annotations

from types import SimpleNamespace

from autoresearch.application.exemplars import (
    ExemplarExtractor,
    parity_check,
)


def _review(paper_id: str, confidence: float, claims: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        paper_id=paper_id,
        confidence=confidence,
        evidence_claim_ids=tuple(f"c{i}" for i in range(claims)),
    )


EXEMPLAR_TEXT = """
Abstract. We study detection. Introduction and related work follow.
Method: our approach. Experiments with 5 seeds. Table 1 and Table 2 report
mean ± std. Figure 1 shows curves. Ablation study in Section 5. Limitations:
small data. p < 0.05. Prior work [1] through [12].
Conclusion.
"""


def test_rank_top_five_by_confidence_then_claims() -> None:
    reviews = [
        _review("low", 0.2, claims=9),
        _review("high", 0.9, claims=1),
        _review("mid", 0.5, claims=5),
        _review("tie-a", 0.5, claims=4),
        _review("tie-b", 0.5, claims=4),
        _review("tie-c", 0.5, claims=4),
    ]
    ranked = ExemplarExtractor().rank(reviews, limit=5)
    assert [r.paper_id for r in ranked] == ["high", "mid", "tie-a", "tie-b", "tie-c"]


def test_feature_extraction_from_paper_text() -> None:
    feats = ExemplarExtractor().features("p1", EXEMPLAR_TEXT)
    assert "abstract" in feats.sections
    assert "ablation" in feats.sections and "limitations" in feats.sections
    assert feats.max_table_ref == 2
    assert feats.max_figure_ref == 1
    assert feats.pm_std_count >= 1
    assert feats.p_value_count >= 1
    assert feats.seed_mention_count >= 1
    assert feats.citation_estimate == 12


def test_profile_medians_and_section_frequency() -> None:
    extractor = ExemplarExtractor()
    papers = [
        extractor.features("p1", EXEMPLAR_TEXT),
        extractor.features("p2", EXEMPLAR_TEXT),
    ]
    profile = extractor.profile(papers)
    assert profile.n_papers == 2
    assert profile.section_frequency["abstract"] == 1.0
    assert profile.medians["max_table_ref"] == 2.0
    assert profile.medians["citation_estimate"] == 12.0


def test_parity_check_flags_missing_sections_and_rigor() -> None:
    extractor = ExemplarExtractor()
    profile = extractor.profile([extractor.features("p1", EXEMPLAR_TEXT)])
    weak = """
# A study

We do things. Some results exist. Conclusion.
"""
    report = parity_check(weak, profile)
    assert not report.parity
    features_flagged = {gap["feature"] for gap in report.gaps}
    assert any(f.startswith("section:") for f in features_flagged)
    assert "max_table_ref" in features_flagged
    assert "pm_std_count" in features_flagged
    assert any("never by decorative text" in g for g in report.guidance)


def test_parity_check_passes_strong_manuscript() -> None:
    extractor = ExemplarExtractor()
    profile = extractor.profile([extractor.features("p1", EXEMPLAR_TEXT)])
    strong = EXEMPLAR_TEXT.replace("[12]", "[12]")  # same rigor level
    report = parity_check(strong, profile)
    assert report.parity
    assert report.gaps == ()


def test_empty_profile_is_inconclusive_not_pass() -> None:
    report = parity_check("anything", ExemplarExtractor().profile([]))
    assert not report.parity
    assert any("no_exemplars" in note for note in report.notes)
