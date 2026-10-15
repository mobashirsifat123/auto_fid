# AutoResearch program progress

| Phase | Status | Evidence | Next action |
|---|---|---|---|
| 0 — Preserve and stabilize | Complete | Baseline commits, tags, bundles, clean worktrees, full suites pass | Implement Phase 1 |
| 1 — Release platform | Complete | `docs/phase_1_completion_report.md`, local v0.1.0-alpha.1 host artifact, CLI/backend gates | Preserve the signed-release blocker and run cross-host builds when toolchains exist |
| 2 — Evaluation foundation | Complete with live matrix skipped | `docs/researchbench_latest.json`, `docs/live_provider_matrix.json`, `evals/benchmarks/researchbench-v0.1.json` | Run the opt-in matrix only with valid credentials and budget |
| 3 — Extensibility | Deferred | Out of immediate scope | Requires Phase 2 evidence |
| 4 — Remote workers | Deferred | No authenticated worker | Keep remote execution disabled |
| 5 — Observability | Deferred | Basic usage/events exist | Revisit after evaluations |
| 6 — Team/cloud readiness | Deferred | Local SQLite only | Revisit after local release gates |
| 7 — Additional clients | Deferred | Terminal and headless first | No web client |
| 8 — Reinvention/publication | Deferred | Reinvention disabled | Requires validated execution lineage |

## Phase 0 capability snapshot

The backend and terminal client expose a default conversational agent, durable
sessions/runs/events, provider-first research workflows, scoped permissions,
and approval-gated local experiment execution. Remote workers and Reinvention
remain disabled with explicit capability reasons.

## Phase 1/2 measured snapshot

- ResearchBench v0.1: 22/22 deterministic cases passed, 0.0 USD fixture cost,
  34.5759 seconds total runtime, 1.0 task success rate.
- CLI deterministic slice: 16 passed, 0 failed; typecheck, build, and OpenAPI
  freshness passed.
- Live provider matrix: three scopes explicitly skipped because
  `RUN_LIVE_AUTORESEARCH_TESTS=1` and credentials were not supplied.
- The offline report leaves claim-level quality metrics null when no adjudicated
  labels exist; it does not turn fixture pass rates into scientific claims.
