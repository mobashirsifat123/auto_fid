# Novelty and experiment implementation-gap audit

Audit date: 2026-08-01. Baseline: backend `7b2ff33`, CLI `af23e0ffdc`.

## Baseline verification

- Backend: 41 passed, 1 skipped, 1 Starlette deprecation warning; compileall and pip check passed.
- CLI: frozen install passed; typecheck passed; 13 tests passed; production build passed.
- Backend schema is version 4. No AutoResearch worktree was present or created.

## Findings

| Area | Already correct | Gap at baseline |
|---|---|---|
| Architecture | FastAPI routes use application services; all model calls use `ModelRouter`; typed models and SQLite persistence exist | Novelty and experiment orchestration were straight-through milestone implementations |
| Novelty | Six uncertainty-aware prior-art statuses and candidate provenance exist | Retrieval used an empty corpus; incremental candidates were discarded; arbitrary novelty cutoff; no revision/reassessment; exactly three portfolio slots forced |
| Critique | Separate critic model role and ten dimensions exist | No explicit rubric/evidence/confidence; numerical score substituted for scientific gates |
| Experiment plan | Typed `ExperimentSpec`, environment manifest, validation/result/report records exist | Missing metric direction/minimum effect, worker policy, exact approval binding, and repository inspection |
| Execution | Job/result persistence exists | Validation gates were simulated; callback accepted arbitrary command text; default output fabricated success metrics; no exact-spec approval gate |
| Statistics | Statistical artifact type exists | Baselines, effect, confidence interval, and cost were hard-coded; single-seed result could be reported as supported |
| API/TUI | `/invent`, `/experiment`, SSE, idempotency, approvals, cancellation, generated contracts exist | Lifecycle resources and experiment modes were incomplete; capability detail overstated isolation/execution |
| Security | No production provider SDK imports in novelty/experiment; fake providers isolated under `providers/testing` | Execution policy and path/secret controls needed adversarial tests |

This audit is the implementation baseline. Passing fixture tests alone is not evidence that a method is novel, statistically supported, or securely sandboxed.
