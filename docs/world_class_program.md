# AutoResearch release and evaluation program

## Scope

This program turns the local AutoResearch prototype into a reproducible,
installable, measurable local-first product without moving scientific decisions
out of the Python backend or weakening approval, provider, or workspace
boundaries.

The active scope is deliberately limited to Phases 0–2:

1. Preserve the QA-hardened baseline.
2. Add a local release platform: launcher, daemon lifecycle, diagnostics,
   semantic compatibility, packaging, and headless execution.
3. Add an offline evaluation foundation and release gates.

Phases 3–8 are not started by this program document. In particular, plugins,
MCP, remote workers, tenancy/cloud, web clients, and Reinvention require their
own later security and capability gates.

## Architecture decisions

- Python remains the source of truth for sessions, pipelines, ModelRouter,
  evidence, persistence, permissions, usage, and experiments.
- The TypeScript/OpenTUI package remains a client. It receives no provider key
  and continues using generated OpenAPI contracts.
- A small Python product launcher owns daemon startup and process lifecycle.
  It invokes only the packaged TUI executable (or an explicitly configured
  installed binary); it does not cold-start a development client or embed
  research logic.
- Product, API, CLI, database, and worker protocol versions are explicit.
  Compatibility is negotiated through `/v1/health` before the TUI starts.
- Release artifacts are checksum-verified locally. Signing and publishing stay
  explicitly unsigned/unpublished until credentials and destinations exist.
- Offline deterministic evaluation is the release gate. Live provider checks
  are opt-in and skip clearly when no credentials or budget configuration exist.

## Phase gates

| Phase | Objective | Gate |
|---|---|---|
| 0 | Preserve baseline | Clean commits, tags, backups, complete baseline suite |
| 1 | Release platform | One-command local launch, doctor, setup, headless JSON, artifact checks, CI |
| 2 | Evaluation foundation | Offline ResearchBench, versioned results, threshold gates, opt-in live matrix |
| 3+ | Deferred | Separate approval after Phase 2 evidence |

## Capability matrix

| Capability | State after Phase 0 | Phase 1/2 requirement |
|---|---|---|
| Conversational agent | Enabled with a configured chat role | Launcher and offline evaluation coverage |
| Repository/file tools | Enabled, contained | Permission and tool-choice evaluations |
| Terminal tools | Policy-controlled and approval-gated | Preserve `shell=False` and evaluate cancellation |
| Study Search | Enabled when provider roles exist | Offline workflow fixture and optional live smoke |
| Deep Literature Review | Enabled when provider roles exist | Evidence/review benchmark |
| Novelty Generation | Enabled when provider roles exist | Known-method and wording benchmark |
| Experiment Design | Enabled | Spec/approval evaluation |
| Local Experiment Execution | Conditional and exact-spec approval-gated | Deterministic execution/reproducibility gate |
| Remote GPU Execution | Disabled | Deferred to authenticated worker phase |
| Result-Conditioned Reinvention | Disabled | Deferred; never simulated |

## Rollback

- Backend baseline tag: `phase-0-preserved-backend`.
- CLI baseline tag: `phase-0-preserved-cli`.
- Local bundle backups are stored outside either worktree under
  `/Users/mobashirsifat/.local/state/autoresearch/backups/`.
- Future release updates must retain the previous installation and activate a
  staged version only after health and compatibility checks pass.
