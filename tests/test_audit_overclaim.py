"""Hermetic unit tests for autoresearch.audit.overclaim (DESIGN §5.4).

Covers: the guarded pattern-tuple equality with application/manuscript.py,
pattern hits and misses, section scoping, the statistical-evidence gate for
"significantly outperforms", and the advisory-only entailment stub pass.
No network, no LLM (the judge is a deterministic stub).
"""

from __future__ import annotations

from autoresearch.audit.overclaim import (
    _COMPARATIVE_PATTERNS,
    _OVERCLAIM_PATTERNS,
    EntailmentJudge,
    EntailmentResult,
    EntailmentVerdict,
    OverclaimAuditor,
    TextSection,
)

# ----------------------------------------------------------------------
# Guarded duplicate (DESIGN §5.4: the tuples must never drift silently)
# ----------------------------------------------------------------------


def test_overclaim_pattern_tuple_matches_manuscript_module() -> None:
    from autoresearch.application.manuscript import (
        _OVERCLAIM_PATTERNS as manuscript_patterns,
    )

    assert _OVERCLAIM_PATTERNS == manuscript_patterns


def test_comparative_pattern_ids_unique() -> None:
    pattern_ids = [pattern.pattern_id for pattern in _COMPARATIVE_PATTERNS]
    assert len(pattern_ids) == len(set(pattern_ids))


# ----------------------------------------------------------------------
# Deterministic pattern pass
# ----------------------------------------------------------------------


def test_pattern_hits_in_scope_sections() -> None:
    sections = [
        TextSection(
            "Results", "Our method outperforms all baselines. This is state-of-the-art."
        ),
        TextSection("Discussion", "A groundbreaking result."),
        TextSection("Introduction", "We build on prior work."),
    ]
    result = OverclaimAuditor().audit(sections)
    hits = [d for d in result.details if d["kind"] == "pattern_hit"]
    assert {(h["pattern_id"], h["section"]) for h in hits} == {
        ("outperforms all", "Results"),
        ("state-of-the-art", "Results"),
        ("groundbreaking", "Discussion"),
    }
    assert all(h["in_scope"] for h in hits)
    # Results carries 2 sentences, Discussion 1 => 3 in-scope sentences,
    # all 3 carry at least one hit.
    assert result.counts["sentences_examined"] == 3
    assert result.counts["hit_sentences"] == 3
    assert result.rates["overclaim_pattern_rate"] == 1.0
    for hit in hits:
        assert hit["start"] < hit["end"]
        assert hit["matched_text"]
        assert hit["sentence"]


def test_pattern_misses_on_clean_text() -> None:
    sections = [
        TextSection("Abstract", "We measure the effect of the intervention."),
        TextSection("Results", "The mean difference was 0.3 with a wide interval."),
    ]
    result = OverclaimAuditor().audit(sections)
    assert result.counts["pattern_hits"] == 0
    assert result.counts["hit_sentences"] == 0
    assert result.rates["overclaim_pattern_rate"] == 0.0


def test_out_of_scope_hits_recorded_but_excluded_from_rate() -> None:
    sections = [
        TextSection("Introduction", "Our approach is the best model for this task."),
        TextSection("Results", "Accuracy improved by 0.02."),
    ]
    result = OverclaimAuditor().audit(sections)
    hits = [d for d in result.details if d["kind"] == "pattern_hit"]
    # "best model" hits twice: the substring pattern and SUPERLATIVE_MODEL.
    assert {(h["pattern_id"], h["section"]) for h in hits} == {
        ("best model", "Introduction"),
        ("SUPERLATIVE_MODEL", "Introduction"),
    }
    assert all(h["in_scope"] is False for h in hits)
    assert result.counts["out_of_scope_hits"] == 2
    assert result.counts["pattern_hits"] == 0
    assert result.rates["overclaim_pattern_rate"] == 0.0
    assert result.coverage["skip_reasons"] == {"section_out_of_scope": 1}


def test_significantly_outperforms_gated_on_statistical_evidence() -> None:
    sections = [
        TextSection("Results", "Our method significantly outperforms the baseline.")
    ]
    without_evidence = OverclaimAuditor(has_statistical_evidence=False).audit(sections)
    assert without_evidence.counts["pattern_hits"] == 1
    assert without_evidence.rates["overclaim_pattern_rate"] == 1.0
    with_evidence = OverclaimAuditor(has_statistical_evidence=True).audit(sections)
    assert with_evidence.counts["pattern_hits"] == 0
    assert with_evidence.rates["overclaim_pattern_rate"] == 0.0


def test_regex_superlative_patterns() -> None:
    sections = [
        TextSection(
            "Abstract", "We present the strongest method and unmatched accuracy."
        )
    ]
    result = OverclaimAuditor().audit(sections)
    pattern_ids = {
        d["pattern_id"] for d in result.details if d["kind"] == "pattern_hit"
    }
    assert pattern_ids == {"SUPERLATIVE_MODEL", "UNMATCHED_RESULTS"}


def test_markdown_headings_and_tables_excluded_from_denominator() -> None:
    sections = [
        TextSection(
            "5. Results",
            "## Results\n| accuracy | 0.9 |\nAccuracy was 0.9.",
        )
    ]
    result = OverclaimAuditor().audit(sections)
    # only the prose sentence counts
    assert result.counts["sentences_examined"] == 1


def test_not_run_without_text() -> None:
    result = OverclaimAuditor().audit([TextSection("Results", "")])
    assert result.status == "NOT_RUN"
    assert result.not_run_reason == "no_paper_text"
    assert result.rates["overclaim_pattern_rate"] is None
    assert result.llm_assisted is False


# ----------------------------------------------------------------------
# Optional entailment pass (advisory only, never sole basis)
# ----------------------------------------------------------------------


class StubJudge:
    def __init__(self, verdict: EntailmentVerdict) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, list[float]]] = []

    def __call__(
        self, claim_text: str, stated_results: list[float]
    ) -> EntailmentResult:
        self.calls.append((claim_text, list(stated_results)))
        return EntailmentResult(
            self.verdict, ("metric:accuracy=0.75",), "stub-critic-model"
        )


def _results_sections() -> list[TextSection]:
    return [
        TextSection(
            "Results",
            "Our method improves accuracy over the baseline. The table lists seeds.",
        ),
    ]


def test_entailment_stub_flags_advisory_only() -> None:
    judge = StubJudge(EntailmentVerdict.NOT_ENTAILED)
    result = OverclaimAuditor().audit(
        _results_sections(), judge=judge, stated_results=[0.75, 0.71]
    )
    assert len(judge.calls) == 1  # only the comparative sentence
    assert judge.calls[0][0].startswith("Our method improves")
    assert judge.calls[0][1] == [0.75, 0.71]
    entailment = [d for d in result.details if d["kind"] == "entailment"]
    assert len(entailment) == 1
    assert entailment[0]["advisory_only"] is True
    assert entailment[0]["sole_basis"] is False
    assert entailment[0]["verdict"] == "not_entailed"
    assert entailment[0]["model"] == "stub-critic-model"
    assert entailment[0]["evidence"] == ["metric:accuracy=0.75"]
    assert result.llm_assisted is True
    assert result.counts["entailment_calls"] == 1
    assert result.counts["entailment_flagged"] == 1
    assert result.rates["entailment_flag_rate"] == 1.0
    # LLM-only hits never touch the primary rate
    assert result.counts["pattern_hits"] == 0
    assert result.rates["overclaim_pattern_rate"] == 0.0


def test_entailment_entailed_and_unverifiable_counted_separately() -> None:
    judge = StubJudge(EntailmentVerdict.UNVERIFIABLE)
    result = OverclaimAuditor().audit(_results_sections(), judge=judge)
    assert result.counts["entailment_unverifiable"] == 1
    assert result.counts["entailment_flagged"] == 0
    assert result.rates["entailment_flag_rate"] == 0.0

    judge_ok = StubJudge(EntailmentVerdict.ENTAILED)
    result_ok = OverclaimAuditor().audit(_results_sections(), judge=judge_ok)
    assert result_ok.counts["entailment_entailed"] == 1
    assert result_ok.rates["entailment_flag_rate"] == 0.0


def test_entailment_judge_error_never_kills_audit() -> None:
    class ExplodingJudge:
        def __call__(
            self, claim_text: str, stated_results: list[float]
        ) -> EntailmentResult:
            raise RuntimeError("router down")

    result = OverclaimAuditor().audit(_results_sections(), judge=ExplodingJudge())
    assert result.status == "RUN_COMPLETE"
    assert result.counts["entailment_errors"] == 1
    assert result.llm_assisted is True
    entailment = [d for d in result.details if d["kind"] == "entailment"]
    assert entailment[0]["error"] == "RuntimeError"  # class-name-only, redacted
    assert entailment[0]["verdict"] is None
    assert result.rates["overclaim_pattern_rate"] == 0.0


def test_entailment_pass_skipped_without_judge() -> None:
    result = OverclaimAuditor().audit(_results_sections())
    assert result.llm_assisted is False
    assert result.counts["entailment_calls"] == 0
    assert result.rates["entailment_flag_rate"] is None


def test_judge_protocol_structural_check() -> None:
    judge: EntailmentJudge = StubJudge(EntailmentVerdict.ENTAILED)
    verdict = judge("claim", [0.5])
    assert verdict.verdict is EntailmentVerdict.ENTAILED


# ----------------------------------------------------------------------
# Envelope shape (integration contract for report.py)
# ----------------------------------------------------------------------


def test_result_to_section_metric_envelope() -> None:
    sections = [TextSection("Results", "Best model overall.")]
    section = OverclaimAuditor().audit(sections).to_section()
    assert set(section) == {
        "status",
        "not_run_reason",
        "llm_assisted",
        "counts",
        "rates",
        "coverage",
        "details",
    }
    assert section["status"] == "RUN_COMPLETE"
    assert section["llm_assisted"] is False
    assert all(isinstance(value, int) for value in section["counts"].values())
    for rate in section["rates"].values():
        assert rate is None or 0.0 <= rate <= 1.0


def test_rate_never_exceeds_one_with_many_hits() -> None:
    text = "It is sota. It remains sota. Still sota. Forever sota."
    result = OverclaimAuditor().audit([TextSection("Results", text)])
    assert result.counts["pattern_hits"] == 4
    assert result.rates["overclaim_pattern_rate"] == 1.0


def test_empty_sections_list_is_not_run() -> None:
    result = OverclaimAuditor().audit([])
    assert result.status == "NOT_RUN"
    assert result.not_run_reason == "no_paper_text"
