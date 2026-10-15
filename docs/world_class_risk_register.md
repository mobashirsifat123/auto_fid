# AutoResearch program risk register

| ID | Risk | Status | Mitigation / blocker |
|---|---|---|---|
| R-001 | No writable backend remote was configured. | Open | Local Git bundle backup created; user must provide a writable repository URL to publish. |
| R-002 | CLI `origin` points to OpenCode upstream and is not treated as writable for AutoResearch releases. | Open | Local Git bundle backup created; do not push upstream. |
| R-003 | No code-signing identity or release publishing credentials were supplied. | Open | Produce checksummed unsigned local artifacts; document signing as blocked. |
| R-004 | No provider credentials/budget were supplied for a live smoke matrix. | Open | Offline fixtures gate normal releases; opt-in live matrix skips with an explicit reason. |
| R-005 | No authenticated remote GPU worker exists. | Controlled | Keep remote execution disabled; do not create an SSH fallback. |
| R-006 | Cold development TUI startup uses `npx` and is slow. | Mitigated in Phase 1 | Package a standalone Bun TUI and prefer it from the launcher. |
| R-007 | Local artifact/database paths can contain private research data. | Controlled | Diagnostics redact secrets and omit file contents by default; backups remain local. |
| R-008 | Existing FastAPI test client emits a Starlette/httpx deprecation warning. | Open | Test suite passes; dependency upgrade is tracked separately from release work. |
| R-009 | ResearchBench v0.1 has no adjudicated claim-level labels for evidence precision/recall or novelty false-positive rates. | Open | Preserve these metrics as null; add licensed labeled fixtures and human review before setting quantitative quality thresholds. |
| R-010 | The v0.1.0-alpha.1 artifact was built and installer-smoked only on macOS ARM64. | Open | Build and validate macOS x64, Linux x64/ARM64, and Windows x64 on their native CI runners before broad release. |
