---
name: reinvention
description: Run the AutoResearch reinvention pipeline — perform result-conditioned scientific iteration on completed experiment reports. Diagnoses failure/success root causes, determines strategic actions (refine, pivot, rebuild, abort), formulates next-generation hypotheses, and synthesizes Cycle N+1 candidates. Requires a candidate_id from a completed experiment run. Use when the user wants to iterate on an experiment, diagnose why a method succeeded or failed, or advance research to the next cycle.
metadata:
  opencode/autoinvoke: "false"  # hidden from model auto-selection; explicit invocation only
  opencode/slash: "true"        # expose as an explicit /<name> command in the OpenCode TUI
---

# Reinvention (AutoResearch backend)

Thin bridge to the AutoResearch scientific backend for result-conditioned research iteration. Never run a conversational agent loop — call the stable skill CLI and interpret its JSON.

Reinvention takes empirical evidence from a completed `experiment` report (metrics, Bayesian evidence, validity audits, ablations) and determines the strategic direction for the next research cycle.

## When to use

- An `experiment` skill run has completed with an `ExperimentReport` (decision: `supported`, `partially_supported`, `no_effect`, `mechanism_contradicted`, etc.).
- The user wants to iterate on empirical findings, diagnose why an experiment failed or succeeded, or advance the research direction to Cycle N+1.
- The user asks "what should we try next?", "how do we fix this failed experiment?", "reinvent this method", or "iterate on these results".

## When NOT to use

- No experiment has been executed yet: run `experiment` skill first.
- The user wants to start a fresh literature review from scratch: use `review` skill.

## Prerequisites

- A completed `experiment` run with a valid `candidate_id` and generated report.

## Reinvention Actions (what the backend determines)

1. **`refine`** (for `supported` / `partially_supported`): Extrapolates empirical gains, addresses remaining limitations and ablations, and raises performance/efficiency targets.
2. **`pivot`** (for `no_effect` / `mechanism_contradicted`): Diagnoses why the causal mechanism failed under testing and proposes a modified mechanism to resolve the root cause.
3. **`rebuild`** (for `implementation_failure` / `optimization_failure`): Fixes hyperparameter instability, gradient issues, or protocol defects.
4. **`abort`** (for `cost_prohibitive` / `data_limitation`): Formally marks the direction as unviable under current resource bounds with a documented negative result.

## Usage

```bash
"${AUTORESEARCH_BIN:-autoresearch}" skill reinvention --candidate-id <candidate_id>
```

## Result format

```json
{
  "ok": true,
  "operation": "reinvention",
  "project_id": "string",
  "run_id": "string",
  "reinvention_id": "string",
  "action": "refine|pivot|rebuild|abort",
  "root_cause_diagnosis": "string",
  "empirical_lessons": ["string"],
  "next_generation_hypothesis": "string",
  "next_candidate_id": "string (optional)",
  "next_candidate_method": "string (optional)"
}
```

## What to report to the user

- **Strategic Action**: Highlight whether the system chose to `REFINE`, `PIVOT`, `REBUILD`, or `ABORT`.
- **Root Cause Analysis**: Summarize the scientific diagnosis of why the parent experiment succeeded or failed.
- **Empirical Lessons**: List key empirical takeaways extracted from statistical/bayesian analyses.
- **Next-Gen Hypothesis & Candidate**: Present the next-generation hypothesis and `next_candidate_id` ready for Cycle N+1 experiment planning!
