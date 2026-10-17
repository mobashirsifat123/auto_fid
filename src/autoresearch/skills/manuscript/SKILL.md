---
name: manuscript
description: Stage 4 of the AutoResearch cycle — generate a complete, evidence-grounded scientific manuscript (Markdown + LaTeX + BibTeX + tables + manifest) from the exact validated review→novelty→experiment lineage of a project. Deterministic generation; never fabricates results, citations, or authors. Reports precise evidence gaps when the chain is incomplete.
metadata:
  opencode/autoinvoke: "false"  # hidden from model auto-selection; explicit invocation only
  opencode/slash: "true"        # expose as an explicit /<name> command in the OpenCode TUI
---

# Manuscript (AutoResearch backend)

Thin bridge to the AutoResearch manuscript stage. Generation is
deterministic: every sentence is assembled from stored, checksummed evidence(review gap + verified claims, frozen methodology, experiment spec,
statistical analysis, report). No model call invents prose, numbers,
citations, authors, or results. This skill never submits or publishes
anything — it only writes editable files locally.

## Mandatory template (FastTrack Academic Research Paper Template)

The FastTrack template (5 pages, bundled as
`FastTrack-Academic-paper-template.pdf` next to this SKILL.md) is
**mandatory, not advisory**. Every instruction line of the template is
transcribed into `template-spec.json` (rule id, source page, type,
enforcement point) and enforced at its enforcement point. The manuscript
stage fails closed on any deterministic-check violation: a complete
manuscript that does not satisfy the template is never shipped.

Headline rules (enforced in `application/manuscript.py`, audited and
persisted as `template_audit.json` beside the manuscript):

1. **Abstract** — 150-250 words with all six labeled parts: Background,
   Gap in Knowledge, Contribution, Population & Methods, Results, Conclusion.
2. **Keywords** — 3-6 terms, derived ONLY from the Findings Memory keywords
   and paper metadata; never invented.
3. **Inside-Out Writing Method** — Methodology → Results → Discussion are
   built before Introduction → Abstract → Keywords → Title; the emission
   order is recorded in `template_audit.json` and checked.
4. **Introduction** — exactly three labeled parts: *Why this topic is
   relevant now* / *Existing state of knowledge and the gap* (built from
   the review's recorded critiques and counter-evidence) / *What this study
   did to fill the gap*. PEER-structured paragraphs.
5. **Literature Review** — included only when literature evidence exists;
   PEER-structured entries (Point, Evidence, Explanation, Repeat/Link) with
   per-study limitations surfaced from the evidence matrix.
6. **Methodology** — the template's five labeled fields (Study Design, Data
   Collection, Sample & Eligibility, Data Analysis, Ethical Considerations)
   rendered from the methodology freeze, plus a cookbook-style frozen
   procedure. A field not recorded at freeze time is written verbatim as
   "not specified at freeze time" and reported as a limitation — never
   invented.
7. **Results** — objective data only: the deterministic tables, a
   deterministic figure (SVG + TikZ) generated from validated data when
   measurements exist, decision gates, and exact statistical tests.
   Interpretation words ("shows that", "suggests", "demonstrates the
   importance", "because", "indicates", "implies") are rejected by a
   deterministic check.
8. **Discussion** — Recap of what we found; **Study Limitations (at least
   3)**; limitations from reviewed studies; Strengths; Implications for
   future research; speculative claims allowed ONLY inside the clearly
   labeled "Possible interpretations and future work" part.
9. **Title** — specific, ≤ 20 words, contains a keyword; vague-term
   blocklist rejected ("An Analysis", "A Study of", "An Investigation",
   "New Insights").
10. **References** — every in-text citation appears in References and vice
    versa; BibTeX built only from stored paper metadata.
11. **Appendices** — evidence bundle (ids + checksums), raw seed metrics,
    and exact recorded commands, generated from the bundle; never empty
    filler.

In the `model` writer mode the model draft must pass the SAME template
audit as the deterministic draft; one bounded revision, then fallback to
the deterministic draft. See `template-spec.json` for the full rule
inventory with enforcement points.

## When to use

- The experiment stage produced an analyzed report and the user wants the
  manuscript ("write the paper", "draft the manuscript", "generate the
  paper from the results").
- The user asks what a complete write-up would need (an incomplete draft +
  evidence-gap report is a valid, honest outcome).
- The full cycle is driven by `autoresearch skill workflow --until
  manuscript` (this stage runs automatically as stage 7).

## When NOT to use

- No project exists yet — run `review` first.
- The user wants to publish or submit somewhere — out of scope by design.

## Prerequisites

- A project with a completed review, frozen methodology, executed
  experiment, validated statistical analysis, and deterministic report.
  Missing pieces produce an INCOMPLETE draft plus a precise gap list, never
  fabricated content.

## Usage

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill manuscript \
  --project "<project_id>"

# Or through the full workflow:
"${AUTORESEARCH_BIN:-autoresearch}" skill workflow \
  --project "<project_id>" --until manuscript
```

## Interpreting the JSON result

- `status` — `complete` (full evidence chain present) or `incomplete`.
- `evidence_gaps` — exact missing prerequisites (e.g. `"experiment: no
  validated statistical analysis (measurement ingestion/analysis pending)"`).
- `decision` — the experiment report's decision, quoted verbatim
  (`supported`, `refuted`, …). Negative and null results are valid.
- `keywords` — the 3-6 terms used in the manuscript Keywords block
  (Findings Memory + paper metadata only).
- `template_audit` — `passed`, `violations`, `checks`, `inside_out_order`,
  and the persisted `template_audit.json` path. A complete manuscript that
  fails the template audit is never produced (fail-closed).
- `lineage` — the exact gap/portfolio/freeze/spec/analysis/report chain this
  manuscript was built from. Incompatible lineages are never mixed.
- `artifacts` — paths to `markdown`, `latex`, `bibtex`,
  `evidence_bundle.json` (all lineage ids + checksums + keywords +
  limitations + exact commands + figures), `manifest.json` (file checksums),
  `template_audit.json`, `figures/` (deterministic SVG when measurements
  exist), and `pdf` (only when a compiler was available AND compilation
  verified; otherwise `null`).

## What to tell the user

- Report `status` first; if `incomplete`, list `evidence_gaps` verbatim and
  state that no results were fabricated.
- Report the decision verbatim with its evidence — including negative/null
  outcomes, which are valid scientific results.
- Point to the editable Markdown/LaTeX/BibTeX files for author revision.
- If `pdf` is null, say no verified LaTeX compiler was available.

## Exit codes

`0` success (including an honestly incomplete draft) · `1` internal ·
`2` usage · `4` project not found.

## Writer modes

- `deterministic` (default): field-assembled manuscript, zero model calls.
- `model` (`[research] manuscript_writer = "model"` or
  `AUTORESEARCH_MANUSCRIPT_WRITER=model`): model-assisted scientific
  argument (contribution outline → section drafting from a pre-formatted
  evidence pack) gated by a **deterministic audit**: every number must trace
  to the allowed evidence set (sign- and rounding-tolerant), every citation
  key must exist, overclaim phrases are rejected, the decision word must
  appear verbatim, and the draft must pass the full FastTrack template
  audit (six-part abstract, 3-6 keywords, section order, ≥ 3 limitations,
  objective Results, vague-title blocklist). One bounded revision; surviving
  violations fall back to the deterministic draft. The audit is persisted
  (`writing_audit.json`) and the deterministic companion
  (`manuscript_deterministic.md`) is kept for diffing.

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

- NEVER present an incomplete manuscript as complete; the gap report is
  part of the deliverable.
- The FastTrack template is mandatory: `template-spec.json` transcribes
  every instruction line and the deterministic audit enforces its
  machine-checkable rules fail-closed. Incomplete drafts are exempt from
  checks that would require fabricated content (abstract length, keyword
  count, section order), but never from the vague-title blocklist.
- NEVER paraphrase the decision or invent numerical values; numbers in the
  manuscript come from the deterministic statistical analysis only.
- Every citation key is verified against the generated BibTeX, which is
  built exclusively from stored paper metadata — invented references are
  structurally impossible.
- Keywords come only from the Findings Memory and paper metadata; a
  missing methodology field is written "not specified at freeze time" and
  listed as a limitation — never invented.
- Re-running the skill regenerates text from unchanged durable evidence; it
  never re-trains, re-runs experiments, or duplicates model calls.
- Fixture-labeled measurements cannot pass the upstream evidence gates, so
  they cannot reach this stage as results.
