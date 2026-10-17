"""End-to-end offline integration test: bundle → wiring → metric sections.

Hermetic (no network): citations runs without sources (honest NOT_RUN); the
deterministic metrics run fully. The runner agent (order 11) extends this to
full report assembly; the citations-wired variant mirrors the mock-transport
pattern in test_audit_citations.py.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from autoresearch.audit import contract as cm
from autoresearch.audit import wiring

PAPER_MD = """# Results

We show that the method achieves 92.3% accuracy on the benchmark [@good2024].

The baseline achieves 85.2% accuracy on the same benchmark [@good2024].

All results are reported for the 3 seeds we ran.
"""

BIB = """@article{good2024,
  title = {Real Paper About Detection},
  author = {Smith, Jane},
  year = {2024},
  journal = {Journal of Things},
  volume = {1}
}
"""

ARTIFACT_JSON = json.dumps(
    {"accuracy": 0.923, "predictions": [1, 0, 1], "labels": [1, 0, 1]}
)
RUN_LOG = (
    "run 1: seed=42 completed\nrun 2: seed=43 FAILED\nerror: divergence on seed 43\n"
)


@pytest.fixture()
def bundle(tmp_path: Path) -> cm.PaperArtifactBundle:
    (tmp_path / "paper.md").write_text(PAPER_MD)
    (tmp_path / "references.bib").write_text(BIB)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "metrics.json").write_text(ARTIFACT_JSON)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "run.log").write_text(RUN_LOG)
    adapter = cm.GenericDirectoryAdapter()
    return adapter.collect(tmp_path)


def test_bundle_roles_assigned(bundle: cm.PaperArtifactBundle) -> None:
    roles = {f.path: f.role for f in bundle.files}
    assert roles["paper.md"] == cm.FileRole.PAPER_SOURCE
    assert roles["references.bib"] == cm.FileRole.BIBLIOGRAPHY
    assert roles["artifacts/metrics.json"] == cm.FileRole.ARTIFACT
    assert roles["logs/run.log"] == cm.FileRole.RUN_LOG
    assert any(entry.key == "good2024" for entry in bundle.bibliography)
    assert any(section.section == "Results" for section in bundle.paper_text)


def test_wiring_runs_all_metrics_offline(bundle: cm.PaperArtifactBundle) -> None:
    deps = wiring.MetricRunnerDeps(
        run_log_summary=cm.RunLogSummary(
            source_files=("logs/run.log",),
            runs_total=2,
            runs_failed=1,
            stages_executed=("train",),
            error_events=("error: divergence on seed 43",),
        ),
        stated_results=(0.923,),
    )
    outcome = wiring.audit_bundle(bundle, deps)
    metrics = {section.metric: section for section in outcome.metric_sections}
    assert set(metrics) == {
        "citations",
        "claims",
        "numbers",
        "overclaim",
        "disclosure",
        "reproducibility",
    }

    # Citations: no sources injected → harness-side NOT_RUN (excluded, never a pass)
    assert metrics["citations"].status == "NOT_RUN"
    assert metrics["citations"].not_run_reason == "no_citation_sources"
    assert metrics["citations"].rates == {
        "citation_trust_rate": None,
        "unverifiable_rate": None,
        "misattribution_rate": None,
    }

    # Claims: two claims — 92.3% artifact-traced (percent normalization);
    # 85.2% unsupported in artifacts → falls to citation_backed (bib key).
    assert metrics["claims"].status == "RUN_COMPLETE"
    assert metrics["claims"].counts["claims_examined"] == 2
    assert metrics["claims"].counts["artifact_traced"] == 1
    assert metrics["claims"].counts["citation_backed"] == 1
    assert metrics["claims"].rates["strong_support_rate"] == 0.5

    # Numbers: 5 tokens total (92.3, 85.2, 3 seeds, 2×2024 citation years).
    # Honest strictness: 92.3 traces via percent normalization, 3 is structural;
    # 85.2 (baseline) and the 2024s stay UNTRACED — citation years may not be
    # laundered into sources while citation resolution is NOT_RUN, and the
    # baseline number has no artifact. (1+1)/5 = 0.4.
    assert metrics["numbers"].status == "RUN_COMPLETE"
    assert metrics["numbers"].counts["numeric_tokens_examined"] == 5
    assert metrics["numbers"].counts["traced_artifact"] == 1
    assert metrics["numbers"].counts["untraced"] == 3
    assert metrics["numbers"].rates["numeric_provenance_rate"] == 0.4

    # Overclaim: no SOTA/groundbreaking language → zero-hit rate, RUN_COMPLETE
    assert metrics["overclaim"].status == "RUN_COMPLETE"
    assert metrics["overclaim"].rates["overclaim_pattern_rate"] == 0.0

    # Disclosure: "All results are reported" vs runs_failed=1 → honesty < 1
    assert metrics["disclosure"].status == "RUN_COMPLETE"
    assert metrics["disclosure"].rates["disclosure_honesty_rate"] == 0.0
    assert any(d.get("kind") == "verdict" for d in metrics["disclosure"].details)

    # Reproducibility: no adapter commands → honest NOT_RUN (never a pass)
    assert metrics["reproducibility"].status == "NOT_RUN"
    assert metrics["reproducibility"].not_run_reason == "NOT_RUNNABLE/no_commands"


def test_wiring_disclosure_without_truth_is_not_run(
    bundle: cm.PaperArtifactBundle,
) -> None:
    outcome = wiring.audit_bundle(bundle, wiring.MetricRunnerDeps())
    metrics = {section.metric: section for section in outcome.metric_sections}
    assert metrics["disclosure"].status == "NOT_RUN"
    assert metrics["disclosure"].not_run_reason == "NO_GROUND_TRUTH"


def test_wiring_adapter_supplies_ground_truth(bundle: cm.PaperArtifactBundle) -> None:
    class SummaryAdapter(cm.GenericDirectoryAdapter):
        def run_log_summary(self, bundle):
            return cm.RunLogSummary(runs_total=1, runs_failed=0)

    outcome = wiring.audit_bundle(
        bundle, wiring.MetricRunnerDeps(), adapter=SummaryAdapter()
    )
    metrics = {section.metric: section for section in outcome.metric_sections}
    assert metrics["disclosure"].status == "RUN_COMPLETE"
    assert metrics["disclosure"].rates["disclosure_honesty_rate"] == 1.0


def test_reproducibility_reruns_scored_fail_closed(
    bundle: cm.PaperArtifactBundle,
) -> None:
    class ReproAdapter(cm.GenericDirectoryAdapter):
        def reproduction_commands(self, bundle):
            return (
                cm.ReproductionCommand(
                    command=("python3", "-c", "print('{\"accuracy\": 0.923}')"),
                    source_file="README.md",
                    description="clean rerun emitting a measurement doc",
                ),
                cm.ReproductionCommand(
                    command=("python3", "-c", "raise SystemExit(3)"),
                    source_file="README.md",
                    description="failing rerun",
                ),
            )

    outcome = wiring.audit_bundle(
        bundle, wiring.MetricRunnerDeps(), adapter=ReproAdapter()
    )
    metrics = {section.metric: section for section in outcome.metric_sections}
    repro = metrics["reproducibility"]
    assert repro.status == "RUN_COMPLETE"
    assert repro.counts["commands_expected"] == 2
    assert repro.counts["rerun_match"] == 1
    assert repro.rates["reproduction_rate"] == 0.5
    # Containment level is always disclosed (soft spot #3)
    containment = [d for d in repro.details if d.get("kind") == "containment_level"]
    assert containment and containment[0]["value"] == "env_scrub_policy_not_os_sandbox"
