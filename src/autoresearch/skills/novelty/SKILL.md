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
   skeptic, enthusiast) run concurrently, using the retrieved prior-art
   assessment, and score each surviving candidate on 10 dimensions:
   novelty, significance, falsifiability, feasibility, and more.

4. **Iterative refinement** — candidates with `partial_overlap` are revised
   at most once, then reassessed. Unresolved overlap stays explicit.

5. **Diversity-aware portfolio selection** — up to 3 candidates that pass
   the scientific gates are selected as low-risk, balanced, and high-risk tiers.

6. **Methodology freeze** — the selected candidate is frozen into a
   versioned methodology (`methodology-v{N}.json`) that fills the manuscript
   template's Methodology fields: `study_design`, `data_collection`,
   `sample_and_eligibility`, `data_analysis`, `ethical_considerations`
   (plus hypothesis, mechanism, falsification conditions, algorithm
   outline, procedure, resource demands, uncertainty). A field the model
   does not supply is stored verbatim as `"not specified at freeze time"`
   — the manuscript's Methodology renders that marker and lists the field
   as a study limitation; it never invents the content.

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
- `candidates` — method, closest-method difference, overlap status, unresolved
  overlap, falsification condition, minimum experiment, objections, and failure
  scenarios. Use these fields to explain the result without another model run.
- `low_risk_candidate_id` — highest feasibility + compute efficiency.
  **Use this for the experiment skill by default.**
- `balanced_candidate_id` — strong publication potential + feasibility,
  methodologically diverse from the low-risk candidate.
- `high_risk_candidate_id` — highest information gain, highest novelty score,
  explicitly higher execution risk.

### What to tell the user after novelty runs

1. Report the actual number selected; fewer than three tiers can pass the gates.
2. For each tier, state the method name and overlap status.
3. **Never present any candidate as "novel"** — use the backend's language:
   - `potentially_novel` → "appears distinct under retrieved evidence"
   - `partial_overlap` → "partial overlap with [papers]; refined but not fully resolved"
   - `incremental` → "incremental improvement over existing methods"
   - `expert_review_required` → "requires expert verification before claiming novelty"
4. Surface the top objections and failure scenarios from the critique.
5. Recommend the `low_risk_candidate_id` for the experiment skill:
   *"To design an experiment, use candidate ID: <low_risk_candidate_id>"*

## Live progress (percent + remaining)

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

## Quality rules

- The stage has no default runtime limit. Let generation, retrieval, critique,
  and reassessment finish. Do not impose token, reasoning, or runtime caps unless
  the user requests them; `--max-runtime-seconds N` is opt-in. Never launch a
  duplicate while a run is active. Record candidate failures; only candidates
  passing the scientific gates qualify for selection.
- **Never present a candidate as definitively novel** — the backend explicitly
  prohibits the wording "proven novel", "definitively novel", "first-ever".
- The freeze records *what will be tested*; it never certifies novelty or
  scientific truth. Template Methodology fields that are not recorded are
  stored as `"not specified at freeze time"` and reported as limitations —
  never invented.
- If `low_risk_candidate_id` is null (all candidates failed gates), report
  that the pipeline produced no candidate meeting the scientific requirements.
- If the `scientific_critic` role is unconfigured (check `skill doctor`),
  critic scores will be absent — say so instead of summarizing empty critiques.
- Surface counter-evidence and overlap honestly; the user needs accurate
  information to decide whether to proceed to the experiment skill.
- The `low_risk_candidate_id` (or whichever the user chooses) is the
  **required input** for the `experiment` skill. Always surface it clearly.

## Findings Memory (feedback loop)

Before generating, the backend retrieves prior findings for this question —
**including failures and refutations** — and conditions generation on them
(do not re-propose a refuted mechanism without an explicit mechanism change).
Cheap deterministic rejections run before any critic call:
- `unfalsifiable` — no falsification condition or no minimal experiment;
- `refuted_finding:<id>` — lexical match to a refuted method for this question.
Rejections are recorded (`novelty.candidate_rejected_cheap` events), never silent.
