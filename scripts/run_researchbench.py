#!/usr/bin/env python3
"""Run the offline ResearchBench release gate without network or paid providers."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evals" / "benchmarks" / "researchbench-v0.1.json"
sys.path.insert(0, str(ROOT / "src"))

from autoresearch.providers.redaction import redact_text
from autoresearch.versioning import PRODUCT_VERSION


def _pytest_counts(output: str) -> tuple[int, int, int]:
    passed = sum(int(value) for value in re.findall(r"(\d+) passed", output))
    failed = sum(int(value) for value in re.findall(r"(\d+) failed", output))
    skipped = sum(int(value) for value in re.findall(r"(\d+) skipped", output))
    return passed, failed, skipped


def run_case(node: str, python: str) -> dict[str, object]:
    started = time.perf_counter()
    result = subprocess.run(
        [python, "-m", "pytest", "-q", node],
        cwd=ROOT,
        env={**os.environ, "RUN_LIVE_AUTORESEARCH_TESTS": "0"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    passed, failed, skipped = _pytest_counts(result.stdout)
    status = "pass" if result.returncode == 0 and passed > 0 and failed == 0 else "fail"
    return {
        "status": status,
        "test": node,
        "passed": passed,
        "failed": failed,
        "skipped": skipped,
        "latency_seconds": round(elapsed, 4),
        "cost_usd": 0.0,
        "provider": "deterministic-fixture",
        "model": "fake-v0",
        "prompt_version": "researchbench-v0.1",
        "error": None if status == "pass" else redact_text(result.stdout[-1200:]),
    }


def run_cli(cli_root: Path) -> dict[str, object]:
    started = time.perf_counter()
    result = subprocess.run(
        ["npx", "--yes", "bun@1.3.14", "test", "--cwd", "packages/autoresearch-cli"],
        cwd=cli_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    elapsed = time.perf_counter() - started
    passed = sum(int(value) for value in re.findall(r"\((\d+) pass", result.stdout))
    failed = sum(int(value) for value in re.findall(r"\((\d+) fail", result.stdout))
    status = "pass" if result.returncode == 0 and failed == 0 else "fail"
    return {
        "status": status,
        "test": "packages/autoresearch-cli test",
        "passed": passed,
        "failed": failed,
        "skipped": 0,
        "latency_seconds": round(elapsed, 4),
        "cost_usd": 0.0,
        "provider": "deterministic-fixture",
        "model": "fake-v0",
        "prompt_version": "tui-qa-v0.1",
        "error": None if status == "pass" else redact_text(result.stdout[-1200:]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "researchbench_latest.json")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--cli-root", type=Path, default=ROOT.parent / "autoresearch-cli")
    parser.add_argument("--skip-cli", action="store_true")
    parser.add_argument("--strict", action="store_true", help="Return non-zero when a release gate is red.")
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    cases: list[dict[str, object]] = []
    started = time.perf_counter()
    for suite in manifest["suites"]:
        for case in suite["cases"]:
            result = run_case(str(case["test"]), args.python)
            cases.append({"suite": suite["name"], **case, "result": result})
    if not args.skip_cli and args.cli_root.is_dir():
        cases.append({"suite": "Generic-TUI", "id": "tui-deterministic-slice", "description": "Conversation-first terminal vertical slice", "result": run_cli(args.cli_root)})

    passed = sum(item["result"]["status"] == "pass" for item in cases)
    failed = len(cases) - passed
    latencies = [float(item["result"]["latency_seconds"]) for item in cases]
    total_latency = time.perf_counter() - started
    task_success_rate = passed / len(cases) if cases else 0.0
    gates = manifest["release_gates"]
    unsafe_failures = sum(item["result"]["status"] == "fail" and item["suite"] in {"ResearchBench-Execution", "ResearchBench-Reviewer"} for item in cases)
    gate_results = {
        "required_case_success_rate": {"value": task_success_rate, "threshold": gates["required_case_success_rate"], "passed": task_success_rate >= float(gates["required_case_success_rate"])},
        "unsafe_case_failures": {"value": unsafe_failures, "threshold": gates["max_unsafe_case_failures"], "passed": unsafe_failures <= int(gates["max_unsafe_case_failures"])},
        "cost_usd": {"value": 0.0, "threshold": gates["max_cost_usd"], "passed": True},
        "latency_seconds": {"value": round(total_latency, 4), "threshold": None, "passed": failed == 0},
    }
    report = {
        "benchmark": manifest["benchmark"],
        "benchmark_version": manifest["version"],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "product_version": PRODUCT_VERSION,
        "fixture_policy": manifest["fixture_policy"],
        "cases": cases,
        "summary": {"total": len(cases), "passed": passed, "failed": failed, "task_success_rate": task_success_rate, "latency_seconds": round(total_latency, 4), "median_case_latency_seconds": round(statistics.median(latencies), 4) if latencies else 0.0, "cost_usd": 0.0},
        "metrics": {
            "task_success_rate": task_success_rate,
            "tool_selection_accuracy": None,
            "routing_accuracy": None,
            "evidence_precision": None,
            "evidence_recall": None,
            "citation_correctness": None,
            "unsupported_claim_rate": None,
            "known_method_false_novelty_rate": None,
            "experiment_spec_validity": None,
            "reproducibility_rate": None,
            "human_intervention_count": 0,
            "token_usage": 0,
            "cost_usd": 0.0,
            "latency_seconds": round(total_latency, 4),
            "first_event_latency": None,
            "unmeasured_metric_reason": "The v0.1 offline gate measures deterministic pass/fail, latency, and zero fixture cost. Quality metrics requiring per-claim labels or human adjudication remain null rather than fabricated.",
        },
        "release_gates": gate_results,
    }
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.expanduser().resolve()), "summary": report["summary"], "release_gates": gate_results}, indent=2))
    return 0 if all(item["passed"] for item in gate_results.values()) or not args.strict else 1


if __name__ == "__main__":
    raise SystemExit(main())
