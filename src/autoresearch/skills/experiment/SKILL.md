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

Translates the novelty candidate into a complete, reproducible experiment
specification with adversarial falsification analysis and power analysis.

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill experiment plan \
  --candidate-id "<candidate_id>"
```

**Output:** `spec` (hypothesis, falsification_condition, metrics, seeds,
statistical_plan, ablations) + `approval` (worker, actions, checksum, budget).
**Show this to the user verbatim before any approval.**

### Step 2 — Validate (no code runs)

Checks that the execution environment (Python version, dependencies, datasets)
is ready. Does not execute experiment code.

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

**Boundary (current release):** the artifact-ingestion and statistical-analysis
step is not yet wired into the CLI, so a completed execution is reported as
`executed_analysis_pending` and this command refuses with
`insufficient_evidence` until an analyzed report exists. Report that state to
the user honestly; never infer a decision from an exit code or fabricate
metrics.

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

### What to tell the user after each step

- **After plan**: Present the hypothesis, falsification condition, and exact
  action plan. Ask the user to review and confirm before approving.
- **After approve (preview)**: Show the exact actions, checksum, worker, and
  budget. Ask explicitly: *"Do you confirm this exact action plan?"*
- **After execute**: Report the run status and any failed jobs.
- **After report**: State the decision clearly and explain the evidence.
  Surface Bayesian probabilities if available.

## Quality rules

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
