# AutoResearch QA defect ledger

QA cycle: 2026-08-02. Scope: the Python backend at
`/Users/mobashirsifat/autoresearch` and the terminal client at
`/Users/mobashirsifat/autoresearch-cli`. The original OpenCode and qwen-ui
repositories were out of scope. Git inspection and mutation were intentionally
skipped at the user's request.

## Summary

| ID | Priority | Area | Defect | Status | Regression validation |
|---|---:|---|---|---|---|
| QA-001 | P1 | Startup/empty state | A missing provider TOML made API startup fail instead of exposing a usable empty installation. | Resolved | `test_api_keeps_empty_provider_installation_usable`; PTY launch and `/v1/health` |
| QA-002 | P1 | Run SSE | A malformed `Last-Event-ID` raised an unstructured conversion error. | Resolved | `test_run_sse_rejects_malformed_last_event_id` |
| QA-003 | P1 | Permissions/cancellation | A permission approved after session cancellation could execute a pending write. | Resolved | `test_cancelled_session_cannot_execute_late_permission` |
| QA-004 | P1 | TUI controls | Ctrl+C exited the TUI even while an agent turn or pipeline run was active. | Resolved | `AutoResearchApp` keyboard path; terminal smoke and app renderer tests |
| QA-005 | P1 | Client SSE | Invalid JSON or stale event IDs could terminate/replay a client stream. | Resolved | `survives malformed and out-of-order run events without replaying them` |
| QA-006 | P2 | TUI resize | Long capability reasons overflowed narrow terminals; a nested sidebar scrollbox corrupted redraw output. | Resolved | `renders online workspace and survives terminal resize` |
| QA-007 | P2 | Generated contract | The client had no stale OpenAPI/schema check. | Resolved | `bun run ... check-contract` |
| QA-008 | P1 | Run SSE scale | A completed run with more than one 200-event database page returned only the first page. | Resolved | `test_run_sse_drains_large_completed_event_history`; 1,000-event replay probe |
| QA-009 | P2 | Cold CLI startup | The documented `npx` development launch reached branding in 8.04 s on this machine. | Observed; no low-risk product fix identified | PTY startup measurement; production bundle still passes |

No P0 defect was found. No unresolved P1 defect remains in the tested scope.

## Evidence map

| ID | Expected behavior | Actual behavior before fix | Files changed | Regression test/evidence |
|---|---|---|---|---|
| QA-001 | Missing providers leave the local API usable and explain model unavailability. | Startup failed and the client reported offline. | `src/autoresearch/providers/factory.py`, `tests/test_api.py` | `test_api_keeps_empty_provider_installation_usable`; live empty-state curl/PTTY |
| QA-002 | Invalid cursors return structured HTTP 400. | Cursor conversion raised an unstructured error. | `src/autoresearch/api/app.py`, `tests/test_api.py` | `test_run_sse_rejects_malformed_last_event_id` |
| QA-003 | Cancellation wins over a late permission approval. | Approved pending write executed after cancellation. | `src/autoresearch/agent/runtime.py`, `src/autoresearch/agent/store.py`, `tests/test_agent_runtime.py` | `test_cancelled_session_cannot_execute_late_permission` |
| QA-004 | Ctrl+C cancels active work; Escape exits. | Ctrl+C exited even while work was active. | `packages/autoresearch-cli/src/app.tsx` | OpenTUI renderer smoke and live PTY launch |
| QA-005 | Invalid/stale client events do not crash or replay. | Malformed JSON could terminate parsing and old IDs could replay. | `packages/autoresearch-cli/src/api.ts`, `packages/autoresearch-cli/src/render.ts`, `packages/autoresearch-cli/test/api.test.ts` | malformed/out-of-order SSE test |
| QA-006 | Narrow resize remains readable and stable. | Reasons overflowed and nested scrolling corrupted redraw. | `packages/autoresearch-cli/src/app.tsx`, `packages/autoresearch-cli/test/app.test.tsx` | resize harness at 100×30 and 72×24 |
| QA-007 | Generated schema drift fails deterministically. | No freshness check existed. | `packages/autoresearch-cli/scripts/check-contract.ts`, `packages/autoresearch-cli/package.json` | `check-contract` reports current |
| QA-008 | Completed SSE replays all durable history. | Only the first 200-event page was returned. | `src/autoresearch/api/app.py`, `tests/test_api.py` | `test_run_sse_drains_large_completed_event_history`; 1,000-event probe |
| QA-009 | Startup shows branded UI promptly. | Cold `npx`/Bun/OpenTUI launch produced first output at 7.47 s and branding at 8.04 s. | No product source change; measured toolchain/runtime path | Performance report; retained as a P2 baseline for a future packaged-launch optimization |

## Reproduction and fixes

### QA-001 — missing provider configuration

Reproduction was a backend launch with `AUTORESEARCH_PROVIDER_CONFIG` unset and
no default `.autoresearch/config.toml`. The container raised
`Provider configuration not found`, and the TUI reported the backend offline.
The provider factory now returns an empty `ModelRouter` for a missing provider
document. Health, projects, sessions, tools, durable inspection, and honest
capability errors remain available; model-backed work reports the missing role
instead of crashing startup.

### QA-002 — malformed run cursor

`GET /v1/runs/{run_id}/events` with `Last-Event-ID: not-a-cursor` previously
failed during `int(...)`. The route now returns HTTP 400 with
`detail.code=invalid_event_cursor`, matching the agent SSE route.

### QA-003 — late permission after cancellation

The runtime previously checked the permission decision but not the session
cancellation flag immediately before tool execution. The runtime now marks the
tool call and turn cancelled and emits `agent.turn_canceled`; the tool is not
executed. Cancelled tool calls are terminal in the durable store.

### QA-004 — cancellation versus exit

The TUI now sends an agent-session cancellation request when an agent turn is
active, sends a run cancellation request when a pipeline is active, and exits
only when no active work exists. Escape remains an unconditional exit control.

### QA-005 — malformed or replayed client events

The client parser now ignores event IDs at or below its cursor, reports invalid
payloads as a visible `client.event_error`, and continues reading later valid
events. The renderer displays that warning without treating it as research
evidence.

### QA-006 — narrow terminal rendering

Capability labels are compact below 100 columns, reasons are moved behind
`/tools`, and the sidebar uses bounded rows rather than a nested scrollbox.
Resize is subscribed through the renderer's resize event and cleaned up on
unmount.

### QA-007 — contract drift

`packages/autoresearch-cli/scripts/check-contract.ts` generates OpenAPI and
TypeScript into a temporary directory and compares the result with the checked
in generated declaration. The check is now part of the CLI package scripts.

### QA-008 — paged run history

The backend SSE route now drains every database page. A full page continues
immediately; a terminal run closes only after a short final page or an empty
terminal read. This preserves all durable events without adding a delay between
pages.

## Security findings

No exploitable shell-injection, path traversal, secret-leak, or artifact-path
defect was found. The added QA security group covers absolute/parent paths,
symlink escape, shell/control-character injection, and header/URL/environment
redaction. The backend has no `shell=True`, `os.system`, `eval`, `exec`, or
`Popen` usage in the reviewed product code; subprocesses use typed allowlisted
argv vectors and a restricted environment.
