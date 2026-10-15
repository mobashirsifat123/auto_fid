# Phase 1 completion report — release platform

Status: **PASS on the current macOS host**

Implementation commits: backend `0eccf9a` (`feat: add local release platform`)
and terminal client `c5ab38b79e` (`feat: package AutoResearch terminal release entry`).

## Implementation

The backend now owns product configuration, loopback daemon lifecycle, API
compatibility negotiation, safe diagnostics, checksum-manifest verification, and
staged updates. The root `autoresearch` command launches the packaged TUI when
called without a subcommand. `run --json` and `run --jsonl` provide headless
machine-readable operation output.

The terminal package exposes the AutoResearch version handshake and a compiled
release entry point. It remains a client/controller; scientific services and
provider credentials stay in Python.

## Measured validation

| Gate | Result |
|---|---|
| release-layer tests | 7 passed |
| CLI command exposure | passed |
| CLI typecheck | passed |
| CLI deterministic tests | 16 passed, 0 failed |
| CLI production bundle | passed, 57 modules |
| compiled TUI `--version` | passed; product 0.1.0-alpha.1, API 2.0.0 |
| backend wheel build | passed; `provider_first_autoresearch-0.1.0a1-py3-none-any.whl` |
| release manifest | passed; 6 files, unsigned and explicitly labelled |
| OpenAPI freshness | passed |
| clean virtual-environment installer smoke | passed; installed wheel, configured TUI, `version` and `doctor --json` succeeded |
| installed headless JSONL run | passed; daemon started automatically, emitted durable events, and stopped cleanly |

Host artifact:

`/Users/mobashirsifat/autoresearch/dist/release/autoresearch-0.1.0-alpha.1-darwin-arm64`

Manifest SHA-256:

`fb3ea6f09ee61b4c60b3f17fefc46981e57ea49f60acd062b75dfbad422365b0`

## Capability matrix

| Capability | Phase 1 state |
|---|---|
| local Python daemon | ready, loopback-only |
| packaged TUI | ready on the build host |
| Study Search | enabled in backend |
| Deep Literature Review | enabled in backend |
| Novelty | existing backend capability preserved; not expanded here |
| Experiment | existing approval-gated local behavior preserved |
| Reinvention | disabled with explicit backend reason |
| remote workers | disabled |

## Open limitations

Cross-platform binaries must be built on their target host/toolchain. The local
artifact is unsigned, and signing/publishing credentials were not provided.
The CLI CI contract-freshness step requires a backend checkout; the local
combined command is the authoritative cross-repository check.
