---
name: workflow
description: Drive the full AutoResearch cycle in one durable, resumable command — seed → literature review → research question → hypotheses → versioned methodology → experimental design → approval-gated execution. Use when the user wants the whole pipeline, or to resume a project to a later stage. Individual skills remain available for single-stage control.
metadata:
  opencode/autoinvoke: "false"  # hidden from model auto-selection; explicit invocation only
  opencode/slash: "true"        # expose as an explicit /<name> command in the OpenCode TUI
---

# Workflow (AutoResearch backend)

Thin bridge to the AutoResearch scientific backend. It reuses completed
stages, so re-invoking with `--project` resumes instead of repeating.

## When to use

- The user wants the complete cycle ("take this seed through to an experiment
  design"), or wants to resume/extend an existing project to a later stage.
- The user asks what stage a project is at (the result lists per-stage
  dispositions with durable identities).

## When NOT to use

- The user wants a single stage under manual control — use the `review`,
  `novelty`, or `experiment` skills.
- No seed input (PDF path, DOI, HTTPS URL, or a local catalog) and no existing
  project ID.

## Prerequisites

- The `autoresearch` CLI installed and on `PATH` (override with
  `AUTORESEARCH_BIN`).
- Provider configuration for model-backed stages (see
  `.autoresearch/config.example.toml`).
- Execution requires an exact-specification approval; without one the run
  stops at `awaiting_approval` and prints the exact package and missing
  requirement. Never approve on the user's behalf without showing the spec.

## Usage

```bash
# New project from a seed paper, stop after review + question:
"${AUTORESEARCH_BIN:-autoresearch}" skill workflow \
  --seed-paper "<pdf|doi|url>" --until question

# Resume to a frozen methodology and experiment design:
"${AUTORESEARCH_BIN:-autoresearch}" skill workflow \
  --project "<project_id>" --until design

# Continue to execution (approval-gated) and inspect the package:
"${AUTORESEARCH_BIN:-autoresearch}" skill workflow \
  --project "<project_id>" --until experiment
```

Stop stages: `review`, `question`, `hypotheses`, `methodology`, `design`,
`experiment`.

## Interpreting the JSON result

- `stages[]` lists per-stage dispositions: `completed`, `reused`,
  `regenerated`, `spec_conflict`, `replanned`, `awaiting_approval`, `executed_analysis_pending`,
  `worker_unavailable`, `execution_failed`, `not_justified`, `unavailable`,
  `stopped`, `budget_exhausted`, …
- Durable identities: `gap_id` (the research question), `portfolio_id`,
  `methodology_freeze_id` + `version`, `spec_id`, and the approval `checksum`.
- `executed_analysis_pending` is honest: a completed approved action is not a
  scientific result until artifacts are ingested and analyzed. Report that
  state; never infer a decision from an exit code.

## Quality rules

- Report the stage dispositions verbatim; do not paraphrase "awaiting" states
  as successes.
- Methodology freezing records what will be tested; it never certifies novelty.
- Do not fabricate results, citations, or decisions. All claims come from the
  backend's retained artifacts.
