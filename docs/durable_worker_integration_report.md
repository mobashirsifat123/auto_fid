# Durable worker integration report

## Implemented

- Additive SQLite V6 migration for worker jobs, exact approval status/action records, lease metadata, process metadata, heartbeat metadata, and enriched state transitions.
- `LocalTypedWorker` with capability reporting, typed allowlisted actions, workspace containment, minimal environment, bounded output, process-group cancellation, idempotent checksum submission, heartbeat updates, durable events, and stale detection.
- Exact SHA-256 approval binding over the complete `ExperimentSpec`, typed action plan, and target worker. API approval accepts checksum, scope, target, budget, approver, and expiry.
- Experiment execution mode now requires an approved exact checksum and a configured local worker. Planning remains non-executing.
- Generated TypeScript schema refreshed and CLI approval/event rendering updated.
- Strict statistical analysis rejects a successful process with missing structured seed metrics.

## Verification

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q` | 49 passed, 1 skipped |
| `.venv/bin/python -m compileall -q src` | passed |
| `.venv/bin/python -m pip check` | passed |
| `bun typecheck` | passed |
| `bun test` | 13 passed |
| `bun build` | passed |
| `tests/test_local_typed_worker.py` | 2 passed |

The offline worker fixture verifies allowlisted script execution, checksum rejection, path rejection, duplicate submission, and durable job persistence. Full remote/GPU execution and opt-in live provider validation were not run.

## Known limitations

The current worker has workspace-level containment rather than an OS sandbox, no remote adapter, and no automatic daemonized reconciler. Validation-gate orchestration, raw metric artifact persistence, and multi-seed report production still require a follow-up implementation; the strict analyzer now fails closed when those artifacts are absent. Reinvention remains disabled.
