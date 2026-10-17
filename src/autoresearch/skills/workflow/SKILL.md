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
`experiment`, `manuscript`. **Credentials:** inside an agent session the
skills automatically follow the session's own AI model and API key
(auto-detected from the OpenCode credential store; no config file needed).
With an explicit config, resolution order is `api_key_env` → OpenCode
credential store → for Anthropic-protocol providers the session's
`ANTHROPIC_API_KEY` / connector settings. The design stage also
generates + statically verifies the experiment implementation when none
exists (bounded repair loop; failures recorded). The manuscript stage
requires an analyzed experiment report; without one it reports `blocked`
with the exact missing prerequisite instead of fabricating content. Every
analyzed outcome is persisted to the shared Findings Memory, which
conditions later novelty runs (failures and refutations first).

The manuscript stage enforces the FastTrack Academic Research Paper
Template (bundled `template-spec.json`): six-part 150-250-word abstract,
3-6 evidence-derived keywords, three-part Introduction, conditional
Literature Review, five-field cookbook Methodology, objective Results with
tables + deterministic figure, Discussion with ≥ 3 limitations, References,
Appendices, and the Inside-Out writing order — audited deterministically
and fail-closed (`template_audit.json`). The upstream stages supply the
evidence those rules consume: review (quotable gap + project keywords +
per-paper study design/datasets/limitations), novelty (the five Methodology
fields or honest "not specified at freeze time" markers), experiment
(≥ 3 limitation rows + raw seed metrics + exact commands).

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


- Stages run in order — review → question → hypotheses → methodology →
  design → experiment. Announce the target stage before invoking the
  command, and report the per-stage dispositions when it returns.
- Stages can take minutes. When the host supports background execution, run
  the CLI in the background and poll
  `"${AUTORESEARCH_BIN:-autoresearch}" skill status --project "<project_id>"`
  every ~30–60 s, echoing one line per poll, e.g.
  `[stage 2/6] review — run <run_id> running`. Stop when the newest run
  reaches a terminal status.
- The user can invoke `/status` in any session at any time for the same
  read-only snapshot while this workflow runs.

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

## Where state lives

Run the first invocation from inside the project's folder: the project's
state is stored in `<project-dir>/.autoresearch/` and the project remembers
it. Resume with `--project <id>` from any directory — the skills anchor to
the project's own database and never mix projects or write into the source
checkout.
