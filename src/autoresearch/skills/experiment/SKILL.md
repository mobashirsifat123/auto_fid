---
name: experiment
description: Run the AutoResearch experiment lifecycle — design a falsifiable experiment from a novelty candidate, validate the environment, require explicit user approval of the exact action plan, then (only with approval) execute it through the policy-controlled worker, and write a scientific report with Bayesian analysis. Requires a candidate_id from the novelty skill. Use when the user wants to design, approve, and run a controlled experiment for a proposed method.
metadata:
  opencode/autoinvoke: "false"  # hidden from model auto-selection; explicit invocation only
  opencode/slash: "true"        # expose as an explicit /<name> command in the OpenCode TUI
---

# Experiment (AutoResearch backend)

Thin bridge to the AutoResearch scientific backend. Planning never
executes code. Execution is gated behind an exact-specification approval
that must be explicitly confirmed by the user. Never skip the approval
step under any circumstances.

## When to use

- The full cycle is being driven by `autoresearch skill workflow` — the
  workflow's design stage consumes the frozen methodology automatically
  (the spec carries `gap_id`, `methodology_freeze_id`, and pre-registered
  `interpretation_rules`). Use this skill directly only for manual control
  of one candidate's lifecycle.
- A `novelty` skill run produced candidate IDs (`low_risk_candidate_id`
  is the recommended default).
- The user wants to design, validate, approve, and/or run a falsifiable
  experiment for a proposed method.
- The user asks to "design an experiment", "test this method", "run it",
  or "get experiment results".

## When NOT to use

- There is no novelty candidate yet: run the `novelty` skill first.
- The user asks to "just run" code without reviewing the exact spec —
  **never skip the approval step**.
- The worker is unavailable (exit code `6`) — report the limitation
  rather than faking results.

## Prerequisites

- A `candidate_id` from the novelty skill output (use `low_risk_candidate_id`
  by default unless the user specifies otherwise).
- The AutoResearch backend running with a configured workspace.

## Full Lifecycle

The experiment skill has 5 ordered steps. Always follow them in order.

### Step 1 — Plan (no code runs)

Produces one concise experiment specification for the smallest decisive test,
then runs one independent adversarial check. That check feeds the final
falsification condition, confounders, and interpretation rules. Power estimates
require a standardized effect-size assumption; missing variance is explicit.

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill experiment plan \
  --candidate-id "<candidate_id>"
```

**Output:** `spec` (hypothesis, falsification_condition, metrics, seeds,
statistical_plan, ablations) + `approval` (worker, actions, checksum, budget).
**Show this to the user verbatim before any approval.**

### Step 1b — Implement (research code generation; no execution)

If no `scripts/run_experiment.py` exists in the planned repository, the
backend generates the implementation from the frozen methodology through the
model router and statically verifies it — deterministically:

- syntax + stdlib-only imports (worker policy),
- exact `IDENTITY` block, declared seeds, required artifact types,
- measurement-contract emission (`json.dumps` document with `seeds`),
- **mechanism coverage**: methodology components mapped to named code
  symbols; coverage < 75% stays `MECHANISM_PARTIAL`, never rounded up.

Bounded repair loop (≤3 attempts) feeds verification failures back. The
workflow's design stage runs this automatically; manual:

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill experiment implement \
  --candidate-id "<candidate_id>"
```

Fidelity is labelled (`STATIC_VERIFIED` here; `SMOKE`/`EXPLORATORY`/
`CONFIRMATORY` attach at execution). A failed verification is recorded as an
`implementation_failure` finding — never masked by silently degrading.

### Step 2 — Validate (no code runs)
Checks that the execution environment (Python version, dependencies, datasets)
is ready. Does not execute experiment code.
Reuses the saved specification and environment snapshot; it never replans or
changes the approved checksum. Request a new plan explicitly when inputs change.

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill experiment validate \
  --candidate-id "<candidate_id>"
```

### Step 3 — Approve (show user, then require explicit consent)

First show the exact action plan preview **without** `--accept`:

```bash
# Preview only — writes nothing
"${AUTORESEARCH_BIN:-autoresearch}" skill experiment approve \
  --candidate-id "<candidate_id>"
```

**STOP. Show the user the exact `actions`, `checksum`, `worker`, and `budget`
from the preview. Ask for explicit confirmation before proceeding.**

Only after the user explicitly confirms, record the approval:

```bash
# Records approval — only after explicit user confirmation
"${AUTORESEARCH_BIN:-autoresearch}" skill experiment approve \
  --candidate-id "<candidate_id>" --accept
```

### Step 4 — Execute (requires recorded approval)

Runs the approved exact-spec plan through the policy-controlled worker.
Only valid after Step 3 `--accept` has been recorded.

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill experiment execute \
  --candidate-id "<candidate_id>"
```

**Output:** run status and worker `jobs`.

### Step 5 — Report

Writes the scientific report: frequentist + Bayesian analysis, validity
assessment, and final decision.

Analysis runs automatically after an approved execution completes: the
worker's measured artifacts are ingested, identity-validated, and analyzed
deterministically before this step. If integrity validation rejected the
measurements (e.g. FIXTURE-labelled artifacts, missing identity, incomplete
seeds), this command refuses with `insufficient_evidence` and the recorded
reason — never a fabricated decision.

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill experiment report \
  --candidate-id "<candidate_id>"
```

### Optional — Inspect or Cancel

```bash
# Inspect the environment manifest
"${AUTORESEARCH_BIN:-autoresearch}" skill experiment inspect \
  --candidate-id "<candidate_id>"

# Cancel a running experiment
"${AUTORESEARCH_BIN:-autoresearch}" skill experiment cancel \
  --run-id "<run_id>"
```

Exit codes: `0` success · `1` internal · `2` usage · `3` approval required ·
`4` not found · `5` insufficient evidence · `6` worker unavailable.

## Progress reporting

While the skill runs, the CLI writes one JSON line per completed step to
**stderr** (stdout stays the single final result):

```json
{"progress": {"stage": "review.papers", "percent": 40, "completed": 8,
  "total": 20, "detail": "8/20 papers reviewed",
  "remaining": ["novelty", "experiment", "manuscript"]}}
```

Surface every event to the user as: `stage: N% (k/total) — remaining: ...`.
Percentages are completed steps / total steps, NEVER time estimates. Do not
invent a percentage before a progress line arrives; report stalls honestly.


- The 5 steps are the progress bar. Announce `[k/5] <step>` before running
  each step's command (e.g. `[3/5] approve`) and report the outcome before
  moving on.
- While `execute` (step 4) is in flight — and whenever the user asks — poll
  `"${AUTORESEARCH_BIN:-autoresearch}" skill status` (read-only; narrow with
  `--project "<project_id>"`) instead of guessing from elapsed time.
- The user can invoke `/status` in any session at any time; it never
  interferes with a running step.

## Interpreting the JSON results

### `plan` output

- `spec.hypothesis` — the testable hypothesis.
- `spec.falsification_condition` — what would refute the hypothesis.
- `spec.metrics` — what is measured.
- `spec.random_seeds` — reproducibility seeds (from power analysis).
- `spec.statistical_plan` — frequentist + Bayesian approach.
- `approval.worker` — the worker type (e.g., `local`).
- `approval.actions` — the exact actions to be executed.
- `approval.checksum` — the tamper-detection checksum.
- `approval.budget` — the compute budget.

### `report` output

- `decision` — one of: `supported`, `partially_supported`, `no_effect`,
  `implementation_failure`, `optimization_failure`, `data_limitation`,
  `evaluation_artifact`, `cost_prohibitive`, `domain_specific`, `mechanism_contradicted`.
- `decision_evidence` — the specific evidence for the decision.
- `limitations` — at least 3 study-limitation rows (measurement provenance,
  dataset/split/seed bounds, unrun ablations/robustness). The manuscript
  Discussion lists them; observations and interpretations stay in separate
  fields and are never merged.
- `exact_commands` — the executor's recorded argv per job. The manuscript
  Appendices list these verbatim; when none were recorded, that absence is
  reported honestly, never reconstructed.

Raw seed observations (checksummed) feed the manuscript Appendices
("Raw seed metrics") through the evidence bundle.

### What to tell the user after each step

- **After plan**: Present the hypothesis, falsification condition, and exact
  action plan. Ask the user to review and confirm before approving.
- **After approve (preview)**: Show the exact actions, checksum, worker, and
  budget. Ask explicitly: *"Do you confirm this exact action plan?"*
- **After execute**: Report the run status and any failed jobs.
- **After report**: State the decision clearly and explain the evidence.
  Surface Bayesian probabilities if available.

## Quality rules

- Planning has no default runtime limit. Let the design and adversarial check
  finish; do not impose token, reasoning, or runtime caps unless the user asks.
  `--max-runtime-seconds N` is opt-in. Never duplicate an active run. Execution
  still requires approval of the exact specification and compute budget.
- **NEVER run `execute` without a recorded `--accept` approval and without
  the user's explicit confirmation of the exact action plan.**
- **NEVER pass the checksum from one candidate's spec to a different candidate's
  execution** — the backend rejects mismatches.
- If the worker is unavailable (exit code `6`), report that execution is
  disabled rather than faking results.
- If execute exits with `3` (approval required), the spec changed since
  the last approval — re-run the approve step before executing.
- If `decision` is `implementation_failure` or `evaluation_artifact`, do NOT
  claim the experiment succeeded — report the failure honestly.
- Surface `decision_evidence` to the user verbatim; do not paraphrase the
  scientific conclusions.
- The report must carry at least 3 study-limitation rows so the manuscript
  Discussion can list them (template requirement); every row is an
  objective fact about the run, never an invented weakness.
