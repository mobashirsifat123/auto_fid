---
name: novelty
description: Run the AutoResearch novelty pipeline — generate candidate methods, check them against prior art, and adversarially critique each one using an ensemble of reviewer perspectives. Requires a completed review run (project_id and gap_id). Use when the user wants to invent novel approaches to a research gap, assess method novelty, or produce a portfolio of risk-tiered candidate methods. Outputs candidate_ids consumed by the experiment skill.
metadata:
  opencode/autoinvoke: "false"  # hidden from model auto-selection; explicit invocation only
  opencode/slash: "true"        # expose as an explicit /<name> command in the OpenCode TUI
---

# Novelty (AutoResearch backend)

Thin bridge to the AutoResearch scientific backend. Never run a
conversational agent for this work — call the stable skill CLI and
interpret its JSON. The backend is deliberately skeptical; it does NOT
claim novelty — it surfaces evidence of overlap and preserves ambiguity.

## When to use

- The full cycle is being driven by `autoresearch skill workflow` — you do not
  need this skill separately; the workflow freezes a versioned methodology
  (`methodology_freeze_id`, artifact `methodology-v{N}.json`) and records it on
  the experiment spec.
- A completed `review` skill run produced a `project_id` and a `gap.id`.
- The user asks to invent approaches to the research gap, assess novelty,
  or produce candidate methods for experimentation.
- The user says "what could be novel?", "invent approaches", or "what methods
  address the gap?".

## When NOT to use

- No review has been run yet: run the `review` skill first.
- The user wants creative brainstorming without evidence checks — the backend
  will report prior-art overlap, not generate unchecked ideas.
- The user already has a novelty candidate and wants to design an experiment —
  use the `experiment` skill directly.

## Prerequisites

- A completed `review` run with a validated research gap.
- The `project_id` from the review output.
- Optionally the `gap.id` (defaults to the project's highest-confidence gap).

## Pipeline (what the backend runs for you)

1. **Candidate generation** — 5 candidates from 5 distinct strategies:
   - limitation root cause elimination
   - cross-domain mechanism transfer
   - contradiction resolution
   - theoretical boundary extension
   - computational bottleneck bypass

2. **Prior-art checking** — each candidate is compared against:
   - The project's verified evidence corpus (internal SQLite claims)
   - External search results from scholarly APIs

3. **Adversarial ensemble critique** — 3 reviewer perspectives (area chair,
   skeptic, enthusiast) score each surviving candidate on 10 dimensions:
   novelty, significance, falsifiability, feasibility, and more.

4. **Iterative refinement** — candidates with `partial_overlap` are revised
   up to 2 rounds to reduce documented overlap.

5. **Diversity-aware portfolio selection** — the best 3 candidates are
   selected as low-risk, balanced, and high-risk tiers.

## Usage

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill novelty \
  --project "<project_id>" \
  [--gap-id "<gap_id>"]
```

`--gap-id` is optional; omitting it uses the project's highest-confidence gap.

Exit codes: `0` success · `1` internal · `2` usage · `4` not found · `5` insufficient evidence.

## Interpreting the JSON result

- `portfolio_id` — the durable novelty portfolio (for reference only).
- `gap_id` — the gap this portfolio addresses.
- `candidate_ids` — all candidate IDs in the portfolio.
- `low_risk_candidate_id` — highest feasibility + compute efficiency.
  **Use this for the experiment skill by default.**
- `balanced_candidate_id` — strong publication potential + feasibility,
  methodologically diverse from the low-risk candidate.
- `high_risk_candidate_id` — highest information gain, highest novelty score,
  explicitly higher execution risk.

### What to tell the user after novelty runs

1. Summarize the portfolio: *"3 candidates were selected across risk tiers."*
2. For each tier, state the method name and overlap status.
3. **Never present any candidate as "novel"** — use the backend's language:
   - `potentially_novel` → "appears distinct under retrieved evidence"
   - `partial_overlap` → "partial overlap with [papers]; refined but not fully resolved"
   - `incremental` → "incremental improvement over existing methods"
   - `expert_review_required` → "requires expert verification before claiming novelty"
4. Surface the top objections and failure scenarios from the critique.
5. Recommend the `low_risk_candidate_id` for the experiment skill:
   *"To design an experiment, use candidate ID: <low_risk_candidate_id>"*

## Quality rules

- **Never present a candidate as definitively novel** — the backend explicitly
  prohibits the wording "proven novel", "definitively novel", "first-ever".
- If `low_risk_candidate_id` is null (all candidates failed gates), report
  that the pipeline produced no candidate meeting the scientific requirements.
- If the `scientific_critic` role is unconfigured (check `skill doctor`),
  critic scores will be absent — say so instead of summarizing empty critiques.
- Surface counter-evidence and overlap honestly; the user needs accurate
  information to decide whether to proceed to the experiment skill.
- The `low_risk_candidate_id` (or whichever the user chooses) is the
  **required input** for the `experiment` skill. Always surface it clearly.
