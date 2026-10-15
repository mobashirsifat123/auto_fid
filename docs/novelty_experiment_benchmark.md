# Fixture benchmark: novelty and experiment controls

Date: 2026-08-01. These deterministic fixtures validate engineering behavior, not scientific novelty or state-of-the-art performance.

| Measure | Expected fixture result | Limitation |
|---|---|---|
| Candidate count/diversity | Five strategy-conditioned candidates | Semantic diversity with live models is opt-in |
| Known method policy | Never promoted | Depends on retrieval coverage |
| Incremental policy | Retained when scientific gates pass | Significance still requires expert review |
| Portfolio filling | One to three candidates; empty tiers allowed | Existing V4 rows retain three columns |
| Claim wording | Prohibits “proven novel”, “definitively novel”, “first-ever” | Policy cannot prove exhaustive prior-art coverage |
| Unsafe action rejection | Traversal, absolute paths and symlink escapes rejected | OS-level sandbox unavailable locally |
| Single-seed statistics | `INSUFFICIENT_EVIDENCE` | No significance test is attempted |
| Duplicate execution | Durable API idempotency retained | Remote-worker heartbeat is not enabled without a worker |

Standard tests use fake providers and harmless local fixtures. Live novelty and remote execution remain opt-in via `RUN_LIVE_NOVELTY_TESTS=1` and `RUN_REMOTE_EXPERIMENT_TESTS=1`.
