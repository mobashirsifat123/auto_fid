# Novelty and experiment integration report

Date: 2026-08-01

## Git

- Backend: `7b2ff33` → `68b84d9` on `codex/first-milestone`; safety tag `pre-sota-novelty-experiments`.
- TUI: `af23e0ffdc` → `d54cd84a04` on `dev`.
- Temporary feature branches were fast-forward merged and deleted. No worktree was created.
- Backend has no configured remote. TUI remotes both target the read-only OpenCode upstream, so nothing was pushed.

## Delivered

- Section/page-aware retrieval from verified evidence claims for prior-art comparison.
- Bounded partial-overlap candidate revision and mandatory reassessment.
- Incremental-candidate retention, hard scientific gates, optional portfolio tiers, and immutable novelty claim ledgers.
- Additive schema V5 with claim ledgers, optional portfolio membership, approval/state tables, and experiment-policy fields.
- Declarative experiment policy with typed Python/pytest actions, path/symlink escape rejection, canonical approval checksums, `shell=False` process construction, bounded logs and timeouts.
- API/service run-ID consistency fix; experiment modes in OpenAPI and generated TypeScript; planning never triggers execution.
- Deterministic paired statistics; single-seed observations return `INSUFFICIENT_EVIDENCE`.
- CLI parsing and generated-contract support for `plan`, `validate`, `execute`, `inspect`, and `report`; reinvention remains disabled.

## Validation

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q` | 47 passed, 1 skipped, 1 existing Starlette warning |
| `.venv/bin/python -m compileall -q src` | passed |
| `.venv/bin/python -m pip check` | no broken requirements |
| `bun install --frozen-lockfile` | passed, no changes |
| CLI typecheck | passed |
| CLI tests | 13 passed, 0 failed |
| CLI production build | passed |

The skipped test is opt-in live integration. Paid-provider and remote-worker tests were not run.

## Security audit

No `shell=True`, `os.system`, `eval`, or Python `exec` match exists. The only subprocess match is `asyncio.create_subprocess_exec(*argv, ...)`, after typed policy resolution. No provider SDK import appears in novelty or experiment modules. `FakeLLMProvider` appears only under `providers/testing` and is not registered as a production provider.

## Honest limitations

Experiment execution remains unavailable until a policy-controlled worker is configured; capability metadata says `conditional` with workspace-only isolation. The V5 approval/state tables and typed executor establish the boundary, but worker heartbeat, stale-job recovery, container/network enforcement, full validation-gate execution, multi-seed statistical tests beyond paired summaries, and a remote GPU demonstration are not implemented in this change. Novelty retrieval uses the project evidence corpus; it does not prove exhaustive global novelty. No claim of state-of-the-art performance is made.
