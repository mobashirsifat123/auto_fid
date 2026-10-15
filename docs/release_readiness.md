# Release readiness audit (2026-10-14)

Prepared for the first public GitHub release. **No external publication
action has been taken** — pushing, tagging, and publishing require explicit
owner authorization.

## Tracked release surface

The tracked tree (~183 files at final staging) consists of: `src/` (backend,
including the canonical agent skills under `src/autoresearch/skills/`),
`tests/`, `docs/`, `evals/` (including a redistributable fixture catalog),
`scripts/` (release/install tooling), `pyproject.toml`, `README.md`,
`.github/workflows/ci.yml`, `.autoresearch/config.example.toml`
(placeholders only), `frontend/README.md`.

## Private / personal content

- **No credentials or secrets found** in tracked files (pattern scan for key
  material, bearer tokens, and `api_key = "..."` literals).
- Personal home paths were removed from all *active* runtime and
  documentation files (`src/autoresearch/skills/*`, `docs/user_guide.md`,
  `docs/security.md`, `docs/claude_code.md`, `docs/agent_runtime.md`,
  `docs/experiment_worker_runbook.md`, `docs/architecture.md`).
- Historical engineering reports under `docs/` (phase reports, QA ledgers,
  migration reports) intentionally retain original paths as historical
  evidence, per repository policy.
- Untracked local research content (private datasets, `research/`,
  `reproduction/`, `artifacts/datasets/`, `paper/`, worktrees, `.venv`,
  local databases) is excluded from the release by `.gitignore`; nothing was
  deleted.
- `docs/research_track.md` (previously the root README) documents the private
  research project this tool was built for. **Owner decision:** keep it in the
  public repo, or drop it from the release if the repository should present
  the tool only.
- `.gitmodules` exists locally but no submodule gitlinks are tracked; it is
  not part of the release.

## Third-party content

- No third-party source code is tracked (submodules are untracked local
  checkouts).
- Dependencies are declared in `pyproject.toml` (httpx, pypdf, typer,
  fastapi, uvicorn; dev: pytest, pytest-asyncio) under their own licenses.
  The public test suite needs no optional scientific packages; CV-track tests
  requiring torch/scipy belong to the untracked private research workspace.

## License decision (owner input required)

Resolved: MIT (`LICENSE`), copyright mobashirsifat123. Prior note kept for history: the repository originally shipped without a license.
applies (all rights reserved; others may not legally redistribute or modify).
Recommended: choose an OSI license the owner is comfortable defending (e.g.
MIT or Apache-2.0) and add a `LICENSE` file plus a `license` field in
`pyproject.toml`. This tool intentionally does not copy code from
HKUDS/AI-Researcher (no license upstream); only mechanisms/prompt-design ideas
were adapted, with provenance comments at each site.

## Outstanding owner decisions

1. ~~Select and add a license~~ — done (MIT).
2. Keep or drop `docs/research_track.md` (private research narrative) from
   the public repo.
3. Repository name/branding (`provider-first-autoresearch` package name on
   PyPI vs GitHub repo name) if packages will ever be published.
4. Whether CI should also cover Python 3.13/3.14 (currently 3.11/3.12;
   locally verified on 3.14).

## Verification snapshot

See the release report for: fresh-environment install, offline first-run
example, full test-suite status, deterministic workflow demonstration, and
independent review disposition.
