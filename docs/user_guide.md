# AutoResearch user guide

Two interfaces exist: the `autoresearch` CLI (works everywhere, no companion
needed) and an optional terminal client developed in a separate repository.
Provider credentials belong only in the backend environment.

**CLI only (recommended start):** see the
[workflow section](#one-command-research-cycle-workflow) below —
`autoresearch workflow` covers the full cycle without the terminal client.

**With the optional terminal client** (separate repository, not required):

```bash
cd /path/to/autoresearch
export AUTORESEARCH_AGENT_ROOTS=/path/to/autoresearch:<companion-terminal-client-dir>
autoresearch-api
cd <companion-terminal-client-dir>
npx --yes bun@1.3.14 run --cwd packages/autoresearch-cli dev
```

The first screen is a conversational `Default Agent`, not a pipeline picker.
Ask about code, files, tests, architecture, or errors without a slash command.
Read-only repository tools are selected by the backend model; file edits and
short terminal commands show a scoped permission request before running.

Useful controls:

* `/tools` shows the source-of-truth capability registry.
* `/mode` shows the current persistent mode; `/use default|study-search|review|novelty|experiment` changes it.
* `/auto on` and `/auto off` control conservative capability suggestions.
* `/discover`, `/review`, `/invent`, and `/experiment` remain typed shortcuts.
* `/approve permission:ID` approves a generic tool; `/approve suggestion:ID` activates a suggested pipeline.
* `/runs`, `/artifacts`, `/usage`, `/provider`, `/cancel`, and `/help` inspect durable state.

`Ctrl+C` cancels the active agent turn or pipeline run and keeps the session
open. `Escape` exits the terminal client. If no provider configuration exists,
the backend remains usable for projects, sessions, tools, and inspection while
model-backed turns report the missing provider role explicitly.

Use `/invent --gap-id GAP_ID` after an evidence-backed review. Inspect prior-art status, critique and the persisted claim ledger before selecting a candidate. Use `/experiment plan CANDIDATE_ID` first. `validate`, `inspect`, `execute`, and `report` are explicit modes; execution must not be interpreted as available unless backend capability metadata reports a configured policy-controlled worker and a matching approval.

`/reinvent` remains disabled. A potentially-novel label means only that no duplicate was found within recorded retrieval scope; expert verification remains required.

## One-command research cycle (workflow)

`autoresearch workflow` (or the skill-bridge equivalent `autoresearch skill
workflow`) advances one project through the whole evidence-first cycle with
durable, resumable state — no manual copying of IDs between stages:

```
seed/DOI/URL → discovery+review → question (gap) → hypotheses (portfolio)
  → methodology (versioned freeze) → design (spec linked to gap+freeze)
  → execution (only under exact-specification approval)
```

```bash
# New project, stop after the literature review and question:
autoresearch workflow --seed ~/papers/seed.pdf --until question

# Same project, continue to a frozen methodology and design:
autoresearch workflow --project <project_id> --until design

# Continue to execution: emits the exact package + missing requirement
# (checksum, actions, budget) unless an approval already covers the spec:
autoresearch workflow --project <project_id> --until experiment

# Bounded adaptive revision after an authorized experiment (opt-in):
autoresearch workflow --project <project_id> --until experiment --max-revisions 1
```

Behavior guarantees:

* **Resume without repeats** — completed stages are reused (`"status":
  "reused"`); each invocation prints per-stage dispositions and identities
  (`gap_id`, `portfolio_id`, `methodology_freeze_id`, `spec_id`, checksums).
* **Versioned methodology** — freezing writes a checksummed
  `methodology-v{N}.json` artifact (hypothesis, predicted observations,
  competing explanations, falsification conditions, closest prior methods,
  algorithm outline, procedure, resource demands, mandatory uncertainty
  statement). Freezing never certifies novelty.
* **Designs consume the freeze** — the spec records `gap_id` and
  `methodology_freeze_id` and carries pre-registered `interpretation_rules`
  for positive/negative/null/failed/unavailable outcomes.
* **Execution stays gated** — without an approved checksum the run stops at
  `awaiting_approval` and creates no jobs; a completed approved action is
  reported honestly as `executed_analysis_pending` until artifacts are
  ingested and analyzed. Duplicate dispatch of the same approved spec
  returns the same job.
* **Bounded revision** — with `--max-revisions N`, a revision-worthy report
  decision (refuted / no_effect / insufficient_evidence / …) triggers at most
  N reinvention cycles, each producing a NEW candidate, freeze version, and
  spec; prior attempts are retained. Protected evaluation is never fed back
  into candidate development.

Individual skills (`review`, `novelty`, `experiment`, `reinvention`) remain
usable standalone and operate on the same durable state.
