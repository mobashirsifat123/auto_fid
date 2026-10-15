---
name: review
description: Run the AutoResearch evidence-backed review pipeline on a paper database. Performs deep literature review, ranks papers by relevance, extracts evidence-backed claims, and surfaces the highest-confidence research gap. Use when the user wants to study their paper database, run a literature review, or surface research gaps from existing or newly discovered papers. Outputs project_id and gap_id consumed by the novelty skill.
metadata:
  opencode/autoinvoke: "false"  # hidden from model auto-selection; explicit invocation only
  opencode/slash: "true"        # expose as an explicit /<name> command in the OpenCode TUI
---

# Review (AutoResearch backend)

Thin bridge to the AutoResearch scientific backend. Never run a
conversational agent for this work — call the stable skill CLI and
interpret its JSON output. All evidence claims are verbatim-verified
against source text.

## When to use

- The user wants the **whole cycle in one command** (review → question →
  hypotheses → versioned methodology → design → approved execution):
  run `autoresearch skill workflow --seed-paper <pdf|doi|url> --until <stage>`
  instead of chaining the skills by hand. It reuses completed stages, so
  `autoresearch skill workflow --project <id> --until experiment` resumes
  exactly where the previous invocation stopped. Stop stages: `review`,
  `question`, `hypotheses`, `methodology`, `design`, `experiment`.
- The user wants to study an existing paper database (project already in the DB).
- The user provides a seed paper (PDF path, DOI, or HTTPS URL) and wants related
  literature discovered, reviewed, and a research gap surfaced.
- The user explicitly asks for a literature review, deep study, evidence analysis,
  or research gap report.

## When NOT to use

- The user wants quick web search results — that is normal research, not this skill.
- There is no seed paper and no existing project yet — ask the user to provide one.
- The user only wants novelty assessment or experiment design — use the `novelty`
  or `experiment` skills instead (but check if a review has already been run first).

## Prerequisites

- The `autoresearch` CLI must be installed and on `PATH`
  (`pip install .` from the repository root; override with the
  `AUTORESEARCH_BIN` environment variable if you use a specific venv).
- Provider configuration with API-key environment variables (see
  `.autoresearch/config.example.toml`); roles required depend on the skill.
- Run `autoresearch skill doctor` if providers appear unconfigured.

## Usage

### Step 0 — Always check existing projects first

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill status
```

If projects exist with a `latest_gap`, offer the user Mode A before starting
a new discovery run.

### Mode A — Existing project (papers already in the database)

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill review \
  --project "<project_id>" \
  [--top-k 5] [--hierarchical]
```

### Mode B — New seed paper (discover + review + gap in one shot)

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill deep-research \
  --seed-paper "<pdf path | doi | https url>" \
  [--limit 20] [--top-k 5] [--since-year 2021] [--hierarchical]
```

### Mode C — Local JSON catalog of papers

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill deep-research \
  --from-catalog "<path to catalog JSON>" \
  [--top-k 5]
```

### Choosing the right mode

1. Run `skill status` first — if projects exist, offer Mode A.
2. If the user provides a seed paper/DOI/URL, use Mode B.
3. If the user provides a catalog JSON, use Mode C.
4. If none of the above, ask the user to provide a seed paper or catalog.

Exit codes: `0` success · `1` internal · `2` usage · `4` not found · `5` insufficient evidence.

## Interpreting the JSON result

After a successful run, the JSON contains:

- `project_id` — **persist this**; required by the `novelty` skill.
- `review_run_id` — the specific review run.
- `reviewed` — number of papers with verifiable evidence extracted.
- `failed_reviews` — papers that had no verbatim-verifiable quotes (not an error).
- `top_paper_ids` — ordered list of top-ranked paper IDs.
- `gap` — the highest-confidence research gap:
  - `gap.id` — **persist this**; it is the `--gap-id` for the `novelty` skill.
  - `gap.statement` — the precise gap statement to show the user.
  - `gap.classification` — `evidence_supported` or `hypothesis`.
  - `gap.confidence` — 0–1 score.
  - `gap.supporting_paper_ids`, `gap.supporting_claim_ids` — evidence provenance.
  - `gap.counter_evidence_claim_ids` — opposing evidence (may be empty).
- `json_report` / `markdown_report` — absolute paths to durable cited reports.

### What to tell the user after a successful review

1. State the research gap clearly: *"The highest-confidence gap is: [gap.statement]"*
2. State the confidence and classification (`evidence_supported` vs `hypothesis`).
3. Mention how many papers were reviewed and how many are in the top ranking.
4. If counter-evidence exists, surface it — do not hide it.
5. Tell the user their `project_id` and `gap.id` for the next step.
6. Suggest running the `novelty` skill next:
   `autoresearch skill novelty --project <project_id> --gap-id <gap.id>`

## Quality rules

- Quote only claims that appear in the report evidence; **never invent citations**.
- If `gap.classification` is `hypothesis`, say so — do not present it as a proven gap.
- If `failed_reviews` significantly exceeds `reviewed`, warn the user (PDFs may be
  image-only or paywalled, causing no verbatim excerpts to be found).
- If a provider role is unconfigured (check `skill doctor`), tell the user which
  role is missing rather than fabricating output.
- The `project_id` and `gap.id` are the **required inputs** for the `novelty` skill.
  Always surface them clearly after a successful run.
