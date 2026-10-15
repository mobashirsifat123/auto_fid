# Phase 2 completion report — evaluation foundation

Status: **PASS for offline release gates; live provider gate not run**

Implementation commit: backend `8c78ce7` (`feat: add deterministic ResearchBench release gates`).

## ResearchBench

The versioned manifest at
`/Users/mobashirsifat/autoresearch/evals/benchmarks/researchbench-v0.1.json`
defines seven suites: Search, Review, Novelty, ExperimentDesign, Execution,
Reproducibility, and Reviewer. It maps each case to a real deterministic test
node and labels the fake-provider policy explicitly.

The measured report is
`/Users/mobashirsifat/autoresearch/docs/researchbench_latest.json`:

| Metric | Measured result |
|---|---:|
| Cases | 22 |
| Passed / failed | 22 / 0 |
| Task success rate | 1.0000 |
| Median case latency | 1.0568 s |
| Total runtime | 34.5759 s |
| Fixture provider cost | $0.00 |
| Unsafe execution/safety failures | 0 |

The runner records provider, model, prompt-version, latency, token/cost policy,
and benchmark version per case. Claim-level evidence precision, recall,
citation correctness, unsupported-claim rate, and first-event latency are
`null` in v0.1 because there is no adjudicated label set or live event timing;
this is an explicit measurement gap, not a fabricated zero.

## Live provider matrix

`/Users/mobashirsifat/autoresearch/docs/live_provider_matrix.json` contains
three scopes (provider routing, Study Search, Deep Review). They are all
`skipped` in this run because no credentials or live-test opt-in were supplied.
No live success is claimed. To run it with a bounded budget:

```bash
RUN_LIVE_AUTORESEARCH_TESTS=1 \
AUTORESEARCH_PROVIDER_CONFIG=/secure/provider.toml \
AUTORESEARCH_LIVE_MAX_COST_USD=2.00 \
AUTORESEARCH_LIVE_MAX_RUNTIME_SECONDS=1800 \
.venv/bin/python scripts/run_live_matrix.py --output docs/live_provider_matrix.json
```

Live reports preserve backend-only credential ownership and redact failure
summaries. Invalid keys, timeout, rate-limit, malformed output, outage, and
budget exhaustion behavior remain covered by provider contract tests; live
network exercise is opt-in rather than required for normal CI.

## Phase capability matrix

| Capability | State |
|---|---|
| Study Search | enabled and benchmarked offline |
| Deep Literature Review | enabled and benchmarked offline |
| Novelty | existing backend behavior benchmarked; no new product scope added |
| Experiment design | existing approval-gated behavior benchmarked |
| Remote workers | disabled |
| Reinvention | disabled with explicit backend reason |

Exit criteria pass for offline ResearchBench and automated release gates. The
live-provider criterion is intentionally **not applicable/passable** until a
user supplies credentials and a budget; its absence is documented as a blocker.
