# Phase 0 completion report — preserve and stabilize

## Starting state

Backend started at `97a7d56` on `codex/first-milestone`; the CLI started at
`cafc803c82` on `dev`. Both repositories contained the coherent, uncommitted
unified-agent and QA-hardening implementation.

## Preservation actions

- Reviewed all changed and untracked source, API, schema, test, and
  documentation paths.
- Committed the backend as `9675cde` — `feat: finalize QA-hardened unified
  agent runtime`.
- Committed the CLI as `d499ea3916` — `feat: finalize conversation-first
  terminal QA polish`.
- Created annotated tags `phase-0-preserved-backend` and
  `phase-0-preserved-cli`.
- Created local bundle backups:
  - `/Users/mobashirsifat/.local/state/autoresearch/backups/autoresearch-phase0-9675cde.bundle`
  - `/Users/mobashirsifat/.local/state/autoresearch/backups/autoresearch-cli-phase0-d499ea3916.bundle`

No remote was pushed. The backend has no configured remote; the CLI remotes are
OpenCode upstream references and are not treated as a user-controlled AutoResearch
release destination.

## Validation

| Command | Result |
|---|---|
| `.venv/bin/python -m pytest -q` | 61 passed, 1 skipped, 1 known deprecation warning |
| `.venv/bin/python -m compileall -q src` | Passed |
| `.venv/bin/python -m pip check` | Passed |
| CLI frozen install | Passed; no changes |
| CLI typecheck | Passed |
| CLI tests | 16 passed, 0 failed, 51 expectations |
| CLI OpenAPI freshness | Passed |
| CLI production build | Passed; 56 modules bundled |
| Security scans | No secret signatures; only intended shell-less subprocess calls |

## Security audit

The agent/worker execution matches are `asyncio.create_subprocess_exec` calls
fed from allowlisted or typed action vectors. There are no `shell=True`,
`os.system`, `eval`, `exec`, or `Popen` matches. The terminal client has no
browser storage or unsafe HTML sink. Provider status remains secret-free.

## Exit criteria

All Phase 0 gates are green: full baseline passes, code is separately committed,
tags and local bundles exist, both repositories are clean, and no extra worktree
exists. Phase 1 is permitted.

## Rollback

Restore from the Phase 0 tags or clone from the exact local bundles above.
No destructive reset command is required.
