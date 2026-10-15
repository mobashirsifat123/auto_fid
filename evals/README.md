# AutoResearch evaluation foundation

This directory contains offline, deterministic release gates. It is deliberately
separate from the production pipeline: fixtures use fake providers and local
papers, and the evaluator records `cost_usd: 0` for those runs. That is a test
fact, not evidence about a live provider or scientific quality.

Run the offline gate from the backend repository:

```bash
.venv/bin/python scripts/run_researchbench.py --strict --output docs/researchbench_latest.json
```

The versioned manifest is `evals/benchmarks/researchbench-v0.1.json`. Every case
records its source test, benchmark version, and fixture/provider identity. The
runner stores only redacted failure summaries, never captured environment values
or full test output.

The live matrix is opt-in and honest about missing credentials:

```bash
.venv/bin/python scripts/run_live_matrix.py --output docs/live_provider_matrix.json
RUN_LIVE_AUTORESEARCH_TESTS=1 AUTORESEARCH_PROVIDER_CONFIG=/secure/provider.toml \
  .venv/bin/python scripts/run_live_matrix.py --output docs/live_provider_matrix.json
```

No live result is inferred from an offline fixture. Novelty, experiment, and
remote-worker capability decisions remain backend-owned; reinvention and remote
execution are not enabled by this evaluation package.
