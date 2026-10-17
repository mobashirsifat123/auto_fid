"""Tests for the prose-style audit (AI-tell detection + exemplar bands)."""

from __future__ import annotations

from autoresearch.application.prose_style import (
    extract_prose_metrics,
    exemplar_style_band,
    prose_audit,
)

EXEMPLAR_TEXT = """
We train a detector on 12,000 images and evaluate on a held-out set of 3,000
images. The detector reaches 92.3% accuracy (Table 1), an improvement of 4.1
points over the baseline of Smith et al. (2019). We report the mean over 5
seeds. The improvement is consistent: the worst seed gains 2.2 points.
Prior work [3] studied smaller corpora. Our ablation removes the attention
module and loses 1.9 points, which suggests the gain is not only from depth
(Table 2). The method fails on blurry images; 7% of failures fall in this
class (Figure 3). We conclude that the augmentation pipeline drives most of
the improvement, though the margin on small objects remains thin (p < 0.05).
"""

AI_GARBAGE_TEXT = """
In the realm of machine learning, it is worth noting that our novel
framework plays a crucial role. Moreover, we delve into a comprehensive
framework that seamlessly leverages state-of-the-art techniques.
Furthermore, this showcases the pivotal landscape of modern detection.
Notably, our approach underscores holistic improvements. Additionally, we
showcase cutting-edge results. It should be noted that the model is
remarkably effective across diverse scenarios.
"""


def test_metrics_extract_concrete_densities() -> None:
    metrics = extract_prose_metrics(EXEMPLAR_TEXT)
    assert metrics.words > 80
    assert metrics.numeric_density > 5  # concrete numbers throughout
    assert metrics.citation_density > 1  # [3], Smith et al. (2019)
    assert metrics.avg_sentence_length > 5


def test_ai_tells_detected_in_garbage_not_in_exemplar() -> None:
    garbage = prose_audit(AI_GARBAGE_TEXT, [EXEMPLAR_TEXT])
    exemplar = prose_audit(EXEMPLAR_TEXT, [EXEMPLAR_TEXT])
    garbage_tells = {t["tell"] for t in garbage.tells}
    assert "cliche_delve" in garbage_tells
    assert "cliche_realm" in garbage_tells
    assert "self_novel" in garbage_tells
    assert garbage.verdict == "REVISE"
    assert exemplar.verdict == "PASS"
    assert any("rewrite flagged LLM-prose tells" in g for g in garbage.rewrite_guidance)


def test_band_deviations_reported() -> None:
    garbage = prose_audit(AI_GARBAGE_TEXT, [EXEMPLAR_TEXT])
    assert any(
        d["metric"] == "numeric_density" and d["manuscript"] < d["exemplar_band"][1]
        for d in garbage.band_deviations
    )
    assert any("evidence-backed" in g for g in garbage.rewrite_guidance)


def test_no_exemplars_is_inconclusive_never_pass() -> None:
    report = prose_audit("Perfectly fine text with 5 numbers and [1] cites.", [])
    assert report.verdict == "INCONCLUSIVE"
    assert any("no_exemplars" in note for note in report.notes)


def test_band_bounds_and_singleton_std() -> None:
    bands = exemplar_style_band([EXEMPLAR_TEXT])
    assert "avg_sentence_length" in bands
    band = bands["avg_sentence_length"]
    assert band.low < band.exemplar_median < band.high
    metrics = extract_prose_metrics("One sentence only.")
    assert metrics.sentence_length_std is None  # honest: undefined, not fabricated
