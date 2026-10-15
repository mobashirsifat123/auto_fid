# AutoResearch Phase 0–2 final program report

Date: 2026-08-02

## Delivered scope

The product is now installable as a local Python backend plus packaged AutoResearch
TUI. `autoresearch` owns configuration, loopback daemon startup, version/API
compatibility, and TUI launch. Research logic, provider routing, evidence,
budgets, durable artifacts, permissions, and experiment decisions remain in the
Python backend. The terminal never receives a provider key.

Phase 2 adds an offline, deterministic ResearchBench release gate and an
opt-in live-provider matrix. Fixture passes are not represented as live-provider
or scientific-quality success.

## Repositories and commits

| Repository | Branch | Relevant commits |
|---|---|---|
| backend | `codex/first-milestone` | Phase 0 `9675cde`, `c708ffd`; Phase 1 `0eccf9a`; Phase 2 `8c78ce7` |
| terminal client | `dev` | Phase 0 `d499ea3916`; Phase 1 `c5ab38b79e` |

The original OpenCode repository was not modified. No worktree was created.

## Validation

| Gate | Result |
|---|---|
| backend full test suite | 67 passed, 1 skipped |
| Python compile / dependency check | passed / no broken requirements |
| terminal typecheck | passed |
| terminal tests | 16 passed, 0 failed |
| terminal production build | passed, 57 modules |
| generated OpenAPI contract freshness | passed |
| ResearchBench v0.1 | 22 passed, 0 failed; 1.0 success rate; 0.0 USD fixture cost |
| live-provider matrix | 3 scopes skipped honestly; no opt-in credentials supplied |
| host release artifact | built, checksummed, unsigned by design |
| clean virtual-environment installer | passed |

The only observed test-suite warning is the existing Starlette/httpx deprecation
notice. It does not change pass status and remains recorded in the risk register.

## Launch, build, and recovery

```bash
cd /Users/mobashirsifat/autoresearch
.venv/bin/python scripts/build_release.py
/bin/bash dist/release/autoresearch-0.1.0-alpha.1-darwin-arm64/install.sh \
  dist/release/autoresearch-0.1.0-alpha.1-darwin-arm64
autoresearch
```

For developer validation:

```bash
autoresearch doctor --json
autoresearch status --json
autoresearch run --prompt "Summarize this workspace" --jsonl
.venv/bin/python scripts/run_researchbench.py --strict
.venv/bin/python scripts/run_live_matrix.py
```

Verified updates use `autoresearch update --source RELEASE_DIR --activate`; the
prior TUI binary is retained and `autoresearch rollback` restores it. The local
artifact is intentionally unsigned until a signing identity and publishing
destination are provided.

## Current limitations

- Only macOS ARM64 was built and installer-smoked in this program run.
- No real provider credentials or cost budget were supplied, so the live matrix
  is skipped rather than passed.
- ResearchBench has deterministic workflow/safety coverage but no adjudicated
  corpus for claim-level evidence precision/recall or novelty false-positive
  rates; those metrics remain null.
- Remote workers and Result-Conditioned Reinvention remain disabled.
