# AutoResearch QA completion report

Date: 2026-08-02

## Outcome

The backend and terminal client passed the deterministic QA scope after fixing
eight P1/P2 reliability and product issues, including the large-event SSE
truncation issue. No P0 defect remains and no unresolved P1 remains. One P2
startup baseline (QA-009) remains documented; it is attributable to the cold
`npx`/Bun/OpenTUI development launch and had no low-risk source-only fix in this
cycle. The unavailable reinvention capability is exposed through its exact API
route with an honest 409 response.

## Validated product behavior

- Python backend starts with a configured provider router or a usable empty
  provider state.
- Projects, sessions, runs, checkpoints, artifacts, usage, permissions, and
  approvals remain durable.
- Discovery and review mutations are idempotent and return immediately with a
  `run_id`.
- Run and agent SSE streams validate cursors, reconnect from the last ID, skip
  stale client events, surface malformed payload warnings, and drain large
  histories without dropping events.
- Resume continues Discovery/Review from persisted checkpoints.
- Cancellation prevents late permission approval from executing a mutation.
- The TUI distinguishes cancel-active-work (`Ctrl+C`) from exit (`Escape`),
  renders empty/disabled states, and remains usable after resize.
- Provider keys stay backend-only; the client reads only optional backend auth
  configuration and never receives provider credentials.
- The generated TypeScript contract is checked against backend OpenAPI.

## Automated results

Final counts are recorded after the final rerun below. The expected report
format is:

| Suite | Result |
|---|---|
| Backend pytest | 61 passed, 1 skipped, 1 warning |
| Backend compileall | pass |
| Backend pip check | pass |
| CLI install frozen | pass |
| CLI typecheck | pass |
| CLI tests | 16 passed, 0 failed, 51 expect() calls |
| CLI contract freshness | pass |
| CLI production build | pass |
| TUI deterministic E2E | 7 passed, 0 failed, 26 expect() calls |

The focused backend integration/security group also passed 51 tests with one
expected skipped test and one dependency deprecation warning. The real cold CLI
launch reached its first output in 7.47 seconds and the branded screen in 8.04
seconds; this includes the `npx`/Bun/OpenTUI startup path and is recorded as a
baseline rather than hidden.

## External integrations and credentials

No live external provider was invoked during this uncredentialed QA cycle.
Semantic Scholar, OpenAlex, Crossref, arXiv, OpenReview, Unpaywall, external
LLM, and external embedding execution therefore have no fabricated success or
failure counts here. The adapter contracts, retry/rate-limit logic, provider
routing, structured outputs, usage accounting, and prior live-integration
artifacts remain covered by their existing deterministic tests and reports.

## Reproduction commands

See [qa_test_plan.md](qa_test_plan.md) for the exact backend, CLI, OpenAPI, PTY,
and deterministic E2E commands. See [qa_defect_ledger.md](qa_defect_ledger.md)
for each defect's reproduction and fix.

## Limitations

The browser frontend and SSH/GPU workers remain intentionally absent. Novelty,
experiment, and reinvention availability is governed by backend capability
state; this QA cycle did not turn disabled pipelines into simulated success.
Git status, history, branches, tags, commits, remotes, staging, and push actions
were intentionally not inspected or changed.
