#!/usr/bin/env python3
"""Run the opt-in live-provider smoke gate and report skipped work honestly."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoresearch.providers.redaction import redact_text
from autoresearch.versioning import PRODUCT_VERSION


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "live_provider_matrix.json")
    args = parser.parse_args()
    enabled = os.getenv("RUN_LIVE_AUTORESEARCH_TESTS") == "1"
    matrix = [
        {"name": "provider-health-and-routing", "scope": "external LLM and embedding provider configuration"},
        {"name": "study-search", "scope": "20-paper discovery with configured scholarly sources"},
        {"name": "deep-review", "scope": "hierarchical evidence-backed review and top-five report"},
    ]
    results: list[dict[str, object]] = []
    for item in matrix:
        if not enabled:
            results.append({**item, "status": "skipped", "reason": "RUN_LIVE_AUTORESEARCH_TESTS is not 1; no credentials or network were used.", "cost_usd": 0.0})
            continue
        started = time.perf_counter()
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/test_live_workflow.py"],
            cwd=ROOT,
            env=dict(os.environ),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        results.append({
            **item,
            "status": "passed" if result.returncode == 0 else "failed",
            "latency_seconds": round(time.perf_counter() - started, 4),
            "cost_usd": float(os.getenv("AUTORESEARCH_LIVE_ESTIMATED_COST_USD", "0")),
            "failure_summary": None if result.returncode == 0 else redact_text(result.stdout[-1500:]),
            "provider_configured_by": "backend environment only",
        })
    report = {
        "product_version": PRODUCT_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "enabled": enabled,
        "credentials_exposed_to_tui": False,
        "configuration": {
            "provider_config": os.getenv("AUTORESEARCH_PROVIDER_CONFIG", "<default or absent>"),
            "seed": os.getenv("AUTORESEARCH_LIVE_SEED", "<default seed>"),
            "max_cost_usd": os.getenv("AUTORESEARCH_LIVE_MAX_COST_USD", "2.00"),
            "max_runtime_seconds": os.getenv("AUTORESEARCH_LIVE_MAX_RUNTIME_SECONDS", "1800"),
        },
        "results": results,
        "policy": "Skipped is not passed. No live result is inferred when credentials are absent.",
    }
    args.output.expanduser().parent.mkdir(parents=True, exist_ok=True)
    args.output.expanduser().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output.expanduser().resolve()), "enabled": enabled, "statuses": [item["status"] for item in results]}, indent=2))
    return 0 if (not enabled or all(item["status"] == "passed" for item in results)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
