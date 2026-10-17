"""Deterministic FastTrack-template enforcement tests.

Covers:
- template-spec.json transcription completeness (every PDF line is quoted),
- abstract word count + six labeled parts,
- keywords 3-6,
- study limitations >= 3,
- inside-out emission order,
- title blocklist / length / keyword,
- Results-section interpretation-word rejection,
- section order (with conditional Literature Review),
- legacy report-limitation parsing,
- title construction bounds.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace
from pathlib import Path

import pytest

from autoresearch.application.manuscript import (
    TEMPLATE_SPEC_PATH,
    ManuscriptService,
    _parse_report_limitations,
    _word_count,
    run_template_audit,
)

# The spec ships inside the package (works from source AND installed wheels);
# resolve it from the module, never from a repo-relative path.
assert TEMPLATE_SPEC_PATH.is_file(), f"template-spec.json missing at {TEMPLATE_SPEC_PATH}"


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _load_spec() -> dict:
    return json.loads(TEMPLATE_SPEC_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# template-spec.json transcription completeness
# ----------------------------------------------------------------------

def test_template_spec_exists_and_declares_source_pdf():
    spec = _load_spec()
    assert spec["spec_version"] == 1
    assert spec["template_title"]
    assert spec["source"]["pages"] == 5
    pdf = TEMPLATE_SPEC_PATH.parent / spec["source"]["file"]
    assert pdf.is_file(), "the template PDF must be bundled next to template-spec.json"


def test_template_spec_covers_every_pdf_line():
    import pypdf

    spec = _load_spec()
    reader = pypdf.PdfReader(str(TEMPLATE_SPEC_PATH.parent / spec["source"]["file"]))
    quoted = {_norm(rule["quote"]) for rule in spec["rules"]}
    extracted: list[str] = []
    for page in reader.pages:
        for raw in (page.extract_text() or "").split("\n"):
            if _norm(raw):
                extracted.append(_norm(raw))
    assert extracted, "no text extracted from the template PDF"
    missing = [line for line in extracted if line not in quoted]
    assert not missing, (
        f"{len(missing)} extracted lines are not transcribed into template-spec.json: {missing[:5]}"
    )


def test_template_spec_rule_shape():
    spec = _load_spec()
    rules = spec["rules"]
    ids = [rule["id"] for rule in rules]
    assert len(ids) == len(set(ids)), "rule ids must be unique"
    for rule in rules:
        assert rule["type"] in {"deterministic-check", "writer-prompt", "style-guide"}, rule["id"]
        assert rule["page"] in {1, 2, 3, 4, 5}, rule["id"]
        assert rule["quote"].strip(), rule["id"]
        assert rule["enforcement"].strip(), rule["id"]
        assert rule["enforcement_point"].strip(), rule["id"]
    params = spec["parameters"]
    for key in ("abstract", "keywords", "title", "limitations", "results", "section_order", "inside_out_order"):
        assert key in params, f"parameters.{key} missing"
    assert params["abstract"]["word_min"] == 150
    assert params["abstract"]["word_max"] == 250
    assert params["keywords"]["min"] == 3
    assert params["keywords"]["max"] == 6
    assert params["limitations"]["min"] == 3
    assert params["title"]["max_words"] == 20
    assert params["title"]["vague_terms_blocklist"] == [
        "an analysis", "a study of", "an investigation", "new insights",
    ]
    assert params["section_order"] == [
        "Introduction", "Literature Review", "Methodology", "Results",
        "Discussion", "References", "Appendices",
    ]


# ----------------------------------------------------------------------
# synthetic complete manuscript used by the audit tests
# ----------------------------------------------------------------------

def _complete_manuscript(**overrides: str) -> str:
    abstract = (
        "**Background:** The question under study is whether per-domain "
        "recalibration improves robustness across held-out domains, a central "
        "concern in current evaluation practice where most published results "
        "still come from single-dataset benchmarks and cross-domain behavior "
        "remains largely unknown. **Gap in Knowledge:** Existing work does not "
        "close this question because most studies evaluate a single dataset "
        "and prior findings are mixed across settings, with some reporting "
        "gains while others report degradation or no measurable change, and "
        "no preregistered multi-domain comparison is recorded in the review. "
        "**Contribution:** To fill this gap we froze a methodology around "
        "per-domain recalibration and tested a preregistered hypothesis with "
        "a matched baseline over multiple seeds, making the comparison "
        "falsifiable and reproducible rather than anecdotal. **Population & "
        "Methods:** Study design: quantitative. Data: Dataset-A checksum "
        "abcdef, split held-out. Sample: three seeds required. Analysis: "
        "exact Wilcoxon signed-rank across seeds. **Results:** Metrics "
        "accuracy. Measured effects: mean difference plus 0.0100. Decision "
        "gates: accuracy gate PASS. The deterministic gates classified the "
        "outcome as supported. **Conclusion:** Future research should address "
        "the limitations listed in the Discussion, in particular the bounds "
        "of dataset Dataset-A and split held-out. Uncertainty recorded: "
        "statistics are limited to the recorded raw seed observations."
    )
    keywords = overrides.get("keywords", "recalibration, robustness, accuracy")
    title = overrides.get("title", "Per-domain recalibration: a preregistered paired-seed evaluation of robustness")
    sections = {
        "Abstract": overrides.get("abstract", abstract),
        "Keywords": keywords,
        "Introduction": "### Why this topic is relevant now\n\nA question.\n\n"
                        "### Existing state of knowledge and the gap\n\nA gap.\n\n"
                        "### What this study did to fill the gap\n\nA study.\n",
        "Literature Review": "### Some Paper (2025) [@paper1]\n\n- **Point:** p\n- **Evidence:** e\n- **Explanation:** x\n- **Repeat/Link:** next\n",
        "Methodology": "### Study Design\n\nquantitative\n\n### Data Collection\n\nDataset-A\n\n"
                      "### Sample & Eligibility\n\nthree seeds\n\n### Data Analysis\n\nWilcoxon\n\n"
                      "### Ethical Considerations\n\nnone recorded\n",
        "Results": overrides.get(
            "results",
            "| Metric | Baseline | Candidate |\n|---|---|---|\n| accuracy | 0.8 | 0.81 |\n\n"
            "![Effect sizes](figures/effect-sizes.svg)\n\nDecision gates: accuracy_gate: PASS.\n",
        ),
        "Discussion": (
            "### Recap of what we found\n\nThe gates classified the outcome as supported.\n\n"
            "### Study Limitations (at least 3)\n1. First limitation.\n2. Second limitation.\n3. Third limitation.\n\n"
            "### Limitations from the reviewed studies\n- one\n\n"
            "### Strengths of this study\n- strength\n\n"
            "### Implications for future research\n- future\n\n"
            "### Possible interpretations and future work (speculative — clearly labeled)\n- speculation\n\n"
            "### Conclusion\n\nDecision: supported.\n"
        ),
        "References": "- [@paper1]: Some Paper (2025).\n",
        "Appendices": "### Appendix A. Evidence bundle\n\nbundle\n",
    }
    lines = [f"# {title}", ""]
    for name, body in sections.items():
        lines.append(f"## {name}")
        lines.append("")
        lines.append(body)
        lines.append("")
    return "\n".join(lines)


def test_audit_accepts_complete_template_conformant_manuscript():
    audit = run_template_audit(
        _complete_manuscript(),
        complete=True,
        literature_evidence=True,
        figure_paths=("figures/effect-sizes.svg",),
        inside_out_order=(
            "Methodology", "Results", "Discussion", "Introduction",
            "Literature Review", "Abstract", "Keywords", "Title",
        ),
    )
    assert audit.passed, audit.violations
    names = {check["check"] for check in audit.checks}
    assert {
        "abstract_six_parts", "abstract_word_count", "keywords_count",
        "section_order", "title_word_count", "title_contains_keyword",
        "title_vague_blocklist", "results_objective_only", "results_has_table",
        "results_has_figure", "discussion_limitations_at_least_3",
        "discussion_structure", "discussion_speculation_labeled",
        "inside_out_order",
    } <= names


def test_audit_rejects_missing_abstract_part():
    text = _complete_manuscript(abstract="**Background:** only one part.")
    audit = run_template_audit(text, complete=True, literature_evidence=True)
    assert not audit.passed
    assert any("abstract_six_parts" in violation for violation in audit.violations)


def test_audit_rejects_abstract_word_count_out_of_range():
    too_short = _complete_manuscript(abstract="**Background:** tiny. **Gap in Knowledge:** x. **Contribution:** y. **Population & Methods:** z. **Results:** w. **Conclusion:** v.")
    audit = run_template_audit(too_short, complete=True, literature_evidence=True)
    assert any("abstract_word_count" in violation for violation in audit.violations)


def test_audit_keywords_must_be_3_to_6():
    for bad in ("one, two", "one, two, three, four, five, six, seven"):
        audit = run_template_audit(
            _complete_manuscript(keywords=bad), complete=True, literature_evidence=True
        )
        assert any("keywords_count" in violation for violation in audit.violations), bad


def test_audit_rejects_vague_title_blocklist():
    for vague in ("An Analysis of robustness", "A Study of recalibration",
                  "An Investigation into seeds", "New Insights on evaluation"):
        audit = run_template_audit(
            _complete_manuscript(title=vague), complete=True, literature_evidence=True
        )
        assert any("title_vague_blocklist" in violation for violation in audit.violations), vague


def test_audit_title_word_limit_and_keyword():
    long_title = " ".join(["word"] * 21)
    audit = run_template_audit(
        _complete_manuscript(title=long_title), complete=True, literature_evidence=True
    )
    assert any("title_word_count" in violation for violation in audit.violations)
    # Title must contain a keyword from the Keywords block.
    no_keyword = _complete_manuscript(
        title="A title with no matching term here at all"
    )
    audit = run_template_audit(no_keyword, complete=True, literature_evidence=True)
    assert any("title_contains_keyword" in violation for violation in audit.violations)


@pytest.mark.parametrize("word", ["shows that", "suggests", "demonstrates the importance", "because", "indicates", "implies"])
def test_audit_rejects_interpretation_words_in_results(word):
    text = _complete_manuscript(
        results="| Metric | Baseline |\n|---|---|\n| accuracy | 0.8 |\n\n"
                f"The result {word} something.\n"
    )
    audit = run_template_audit(text, complete=True, literature_evidence=True)
    assert any("results_objective_only" in violation for violation in audit.violations), word


def test_audit_rejects_wrong_section_order():
    text = _complete_manuscript().replace(
        "## Methodology", "## Results"
    ).replace("## Results\n\n| Metric", "## Methodology\n\n| Metric", 1)
    audit = run_template_audit(text, complete=True, literature_evidence=True)
    assert any("section_order" in violation for violation in audit.violations)


def test_literature_review_conditional_on_evidence():
    with_review = _complete_manuscript()
    audit = run_template_audit(with_review, complete=True, literature_evidence=False)
    assert any("section_order" in violation for violation in audit.violations)
    without_review = with_review.replace(
        "## Literature Review\n\n### Some Paper", "## Literature Review\n\n"
    )
    # Remove the section entirely: emulate no literature evidence.
    without_review = re.sub(
        r"## Literature Review\n\n.*?\n\n(?=## Methodology)",
        "",
        without_review,
        flags=re.DOTALL,
    )
    audit = run_template_audit(without_review, complete=True, literature_evidence=False)
    assert audit.passed, audit.violations


def test_audit_requires_three_numbered_limitations():
    text = _complete_manuscript()
    text = text.replace("3. Third limitation.\n", "")
    audit = run_template_audit(text, complete=True, literature_evidence=True)
    assert any("discussion_limitations_at_least_3" in violation for violation in audit.violations)


def test_audit_requires_labeled_speculation_part():
    text = _complete_manuscript()
    text = text.replace(
        "### Possible interpretations and future work (speculative — clearly labeled)\n",
        "### Speculation\n",
    )
    audit = run_template_audit(text, complete=True, literature_evidence=True)
    assert any("discussion_speculation_labeled" in violation for violation in audit.violations)


def test_inside_out_order_checked():
    order = (
        "Methodology", "Results", "Discussion", "Introduction",
        "Literature Review", "Abstract", "Keywords", "Title",
    )
    audit = run_template_audit(
        _complete_manuscript(), complete=True, literature_evidence=True,
        inside_out_order=order,
    )
    assert audit.passed, audit.violations
    wrong = ("Abstract", "Title", "Methodology", "Results", "Discussion", "Introduction")
    audit = run_template_audit(
        _complete_manuscript(), complete=True, literature_evidence=True,
        inside_out_order=wrong,
    )
    assert any("inside_out_order" in violation for violation in audit.violations)


def test_incomplete_draft_is_relaxed_but_never_vague_titled():
    incomplete = (
        "# New method: an evidence-backed evaluation (incomplete draft)\n\n"
        "## Abstract\n\nThis manuscript is explicitly INCOMPLETE.\n"
    )
    audit = run_template_audit(incomplete, complete=False, literature_evidence=False)
    assert audit.passed, audit.violations
    vague = incomplete.replace("New method", "A Study of missing data")
    audit = run_template_audit(vague, complete=False, literature_evidence=False)
    assert any("title_vague_blocklist" in violation for violation in audit.violations)


# ----------------------------------------------------------------------
# pure helpers
# ----------------------------------------------------------------------

def test_word_count_counts_hyphenated_and_apostrophe_tokens():
    assert _word_count("One two-three. Four's five!") == 4


def test_parse_report_limitations_accepts_legacy_and_json():
    assert _parse_report_limitations("plain limitation") == ["plain limitation"]
    assert _parse_report_limitations('["a", "b"]') == ["a", "b"]
    assert _parse_report_limitations(["a", "b"]) == ["a", "b"]
    assert _parse_report_limitations(None) == []
    assert _parse_report_limitations("") == []


def test_title_construction_bounds_and_keyword():
    candidate = SimpleNamespace(
        method="Adaptive Fusion Recalibration Method A B C D E F G H I J K L M N O P"
    )
    title = ManuscriptService._title(candidate, complete=True, keywords=["recalibration"])
    assert _word_count(title) <= 20
    assert re.search(r"\brecalibration\b", title.lower())
    assert " (incomplete draft)" not in title
    incomplete = ManuscriptService._title(candidate, complete=False, keywords=["recalibration"])
    assert incomplete.endswith(" (incomplete draft)")
    # No keywords available: honest fallback still avoids vague blocklist terms.
    bare = ManuscriptService._title(candidate, complete=True, keywords=[])
    assert "an evidence-backed evaluation" in bare


# ----------------------------------------------------------------------
# model-writer mode: deterministic template gate + bounded fallback
# ----------------------------------------------------------------------

class _StubRouter:
    """Fake ModelRouter returning scripted texts in call order (outline,
    draft, revision...)."""

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = 0

    async def complete(self, role, *, messages=None, project_id=None,
                       prompt_version=None, operation=None):
        text = self.texts[min(self.calls, len(self.texts) - 1)]
        self.calls += 1
        return SimpleNamespace(text=text)


def _model_service(tmp_path, texts):
    from autoresearch.storage.repository import ResearchRepository

    repository = ResearchRepository(tmp_path / "state.sqlite3", tmp_path / "artifacts")
    return ManuscriptService(repository, router=_StubRouter(texts), writer_mode="model")


_TEMPLATE_VIOLATING_DRAFT = (
    "# An Analysis of robustness\n\n"
    "## Abstract\n\n**Background:** This abstract is far too short.\n\n"
    "## Keywords\n\none, two\n\n"
    "## 1. Introduction\n\nintroduction text\n\n"
    "## 3. Methodology\n\nmethods\n\n"
    "## 4. Results\n\n| Metric | Baseline |\n|---|---|\n| accuracy | row |\n\nDecision gates: PASS.\n\n"
    "## 5. Discussion\n\n### Recap of what we found\n\nsupported.\n\n"
    "### Study Limitations (at least 3)\n1. one\n2. two\n\n"
    "### Strengths of this study\n\n- s\n\n### Implications for future research\n\n- f\n\n"
    "### Possible interpretations and future work (speculative)\n\n- p\n\n### Conclusion\n\nDecision: supported.\n"
    "## References\n\n- none.\n\n## Appendices\n\nappendix\n"
)

_SECTIONLESS_DRAFT = (
    "# Bare: a preregistered evaluation of robustness\n\n"
    "## Results\n\nDecision gates: PASS.\n\n"
    "## Discussion\n\nsupported.\n"
)


def _model_kwargs(tmp_path):
    return {
        "root": tmp_path,
        "project": SimpleNamespace(id="p1"),
        "markdown": "deterministic draft text",
        "analysis": None,
        "spec": None,
        "papers": [],
        "keys": {},
        "gap": None,
        "counter_claims": [],
        "methodology": None,
        "report": None,
        "gaps": [],
        "tables_md": "",
        "decision": "supported",
        "limitations": ["limitation one", "limitation two", "limitation three"],
        "keywords": ["robustness"],
        "figure_svg": None,
        "review_notes": {},
        "seed": None,
    }


async def test_model_draft_template_violations_fall_back_after_one_revision(tmp_path):
    service = _model_service(
        tmp_path, ["outline text", _TEMPLATE_VIOLATING_DRAFT, _TEMPLATE_VIOLATING_DRAFT]
    )
    result = await service._model_draft(**_model_kwargs(tmp_path))
    assert result["used_model_draft"] is False
    assert result["audit"]["fallback"] == "deterministic"
    assert "deterministic draft kept" in result["reason"]
    # Both the pre- and post-revision attempts are recorded, never overwritten.
    assert len(result["audit"]["attempts"]) == 2
    first, second = result["audit"]["attempts"]
    assert first["template"], "first attempt must record its template violations"
    assert second["template"], "revision must record its template violations"


async def test_model_draft_missing_sections_falls_back_without_crash(tmp_path):
    """A draft missing whole sections (Abstract/Keywords/Introduction) must
    be recorded as a template violation and fall back — never raise out of
    the audit (regression for names.index(...) on missing sections)."""
    service = _model_service(
        tmp_path, ["outline text", _SECTIONLESS_DRAFT, _SECTIONLESS_DRAFT]
    )
    result = await service._model_draft(**_model_kwargs(tmp_path))
    assert result["used_model_draft"] is False
    assert result["audit"]["fallback"] == "deterministic"
    assert len(result["audit"]["attempts"]) == 2
    assert any(
        "abstract_keywords_before_introduction" in violation
        for violation in result["audit"]["attempts"][0]["template"]
    )


async def test_model_draft_conformant_draft_is_adopted(tmp_path):
    papers = [{"id": "p1", "title": "Some Paper", "year": 2025, "method": "m", "abstract": ""}]
    draft = _complete_manuscript()
    draft = draft.replace("mean difference plus 0.0100", "mean difference measured")
    draft = draft.replace("| accuracy | 0.8 | 0.81 |", "| accuracy | baseline | candidate |")
    service = _model_service(tmp_path, ["outline text", draft])
    kwargs = _model_kwargs(tmp_path)
    kwargs["papers"] = papers
    kwargs["keys"] = {"p1": "paper1"}
    result = await service._model_draft(**kwargs)
    assert result["used_model_draft"] is True, result["audit"]
    assert result["markdown_path"]
    assert Path(result["markdown_path"]).read_text(encoding="utf-8") == draft


def test_violation_total_combines_audit_and_template(tmp_path):
    clean = {"numbers": [], "citations": [], "overclaims": [], "decision_missing": False}
    assert ManuscriptService._violation_total(clean, []) is False
    assert ManuscriptService._violation_total(clean, ["template x"]) is True
    assert ManuscriptService._violation_total({**clean, "numbers": ["1.5"]}, []) is True


def test_template_violations_reports_none_for_conformant_draft(tmp_path):
    service = _model_service(tmp_path, [])
    assert service._template_violations(
        _complete_manuscript(), literature_evidence=True, figure_svg=None
    ) == []
    violations = service._template_violations(
        _TEMPLATE_VIOLATING_DRAFT, literature_evidence=False, figure_svg=None
    )
    assert violations, "the violating draft must produce template violations"


async def test_shipped_reaudit_discards_model_draft_through_build(tmp_path, monkeypatch):
    """Judge follow-up: route through _build to cover the shipped-text
    re-audit branch. When the final model draft fails the template audit at
    SHIP time (defense-in-depth re-check), the deterministic draft is
    restored and the fallback reason recorded — never shipped silently."""
    from pathlib import Path
    from uuid import uuid4

    import autoresearch.application.manuscript as manuscript_module
    from autoresearch.application.manuscript import ManuscriptService
    from test_smoke_e2e import _build_loop

    repository, workflow, seed = _build_loop(tmp_path)
    texts = ["outline", _complete_manuscript()]
    service = ManuscriptService(repository, router=_StubRouter(texts), writer_mode="model")
    workflow.manuscript = service

    real_audit = manuscript_module.run_template_audit
    calls = {"n": 0}

    def failing_shipped_audit(*args, **kwargs):
        calls["n"] += 1
        audit = real_audit(*args, **kwargs)
        if calls["n"] == 3:  # deterministic-audit, gate-1, then the shipped re-audit
            audit.violations.append("shipped_reaudit_simulation")
        return audit

    monkeypatch.setattr(manuscript_module, "run_template_audit", failing_shipped_audit)
    # The model draft's fixed fixture numbers/decision must not trip the
    # numeric/citation/overclaim gate: this test covers the TEMPLATE
    # re-audit branch, so those gates are bypassed for the model draft only.
    monkeypatch.setattr(
        service,
        "_audit_draft",
        lambda draft, allowed, keys, decision: {
            "numbers": [], "citations": [], "overclaims": [], "decision_missing": False,
        },
    )

    first = await workflow.run(seed=str(seed), until="design")
    project = first["project_id"]
    candidate_id = next(s for s in first["stages"] if s["stage"] == "methodology")["candidate_id"]
    preview = workflow.experiment.approval_preview_for_candidate(candidate_id)
    saved = repository.save_exact_approval({
        "id": uuid4().hex, "run_id": preview["run_id"], "specification_checksum": str(preview["checksum"]),
        "approver": "test", "approved_scope": "workspace", "approved_target": str(preview["worker"]),
        "approved_budget": 60.0, "expires_at": "2099-01-01T00:00:00+00:00", "status": "pending",
    })
    repository.approve_exact(saved["id"], approver="test")
    result = await workflow.run(project_id=project, until="manuscript")

    stages = {s["stage"]: s for s in result["stages"]}
    assert stages["manuscript"]["status"] == "complete"
    assert calls["n"] >= 3, f"expected the shipped re-audit to run; {calls['n']} calls"
    record = repository.latest_manuscript(project)
    report = json.loads(Path(record.evidence_bundle_path).parent.joinpath("writing_audit.json").read_text(encoding="utf-8"))
    assert report["used_model_draft"] is False
    assert report["fallback"] == "deterministic"
    assert "template audit failed on the final model draft" in report["reason"]
    shipped = Path(record.markdown_path).read_text(encoding="utf-8")
    assert "deterministic decision gates" in shipped  # deterministic draft restored
