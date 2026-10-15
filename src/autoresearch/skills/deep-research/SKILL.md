---
name: deep-research
description: Run the AutoResearch deep-research pipeline — discover related papers, run an evidence-backed review, and surface one cited research gap. Use when the user wants a literature review, prior-art survey, or research-gap analysis on a topic, paper, or question.
metadata:
  opencode/autoinvoke: "false"  # hidden from model auto-selection; explicit invocation only
  opencode/slash: "true"        # expose as an explicit /<name> command in the OpenCode TUI
---

# Deep Research (AutoResearch backend)

Thin bridge to the AutoResearch scientific backend. Never run a conversational
agent for this work — call the stable skill CLI and interpret its JSON.

## When to use

- The user asks for a literature review, related-work survey, or research-gap
  analysis on a paper, DOI, URL, PDF, or research question.
- The user wants evidence-backed claims with citations, not a generative summary.

## When NOT to use

- The user just wants a short web search — that is normal web research, not
  the backend pipeline.
- No seed paper and no catalog file exists and the user cannot provide one:
  the pipeline needs a starting corpus.

## Usage

Provide exactly one of `--seed-paper`, `--from-catalog`, or `--project`:

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill deep-research \
  --seed-paper "<pdf path | doi | https url>" \
  [--limit 20] [--top-k 5]
```

Exit codes: `0` success · `1` internal · `2` usage · `3` approval · `4` not found · `5` insufficient evidence.

## Interpreting the JSON result

- `project_id` — use it for later `novelty` and `experiment` steps.
- `review_run_id` — the review run, resumable via `autoresearch review --run`.
- `top_paper_ids` — the ranked evidence-backed papers.
- `gap` — the single highest-confidence research gap with `id`, `statement`,
  `supporting_paper_ids`, `supporting_claim_ids`, and `counter_evidence_claim_ids`.
  This `gap.id` is what `novelty` consumes.
- `json_report` / `markdown_report` — durable cited reports.

## Quality rules

- Quote only claims that appear in the report's evidence; never invent citations.
- If the user asks for novelty next, call `autoresearch skill novelty --project <project_id>`.
- If the user asks to design an experiment, first get a novelty candidate, then
  call `autoresearch skill experiment plan --candidate-id <id>`.
- If the backend reports a degraded provider (`autoresearch skill doctor`),
  tell the user what role is unconfigured rather than proceeding with fabrications.
