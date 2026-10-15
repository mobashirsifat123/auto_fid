# Scientific Integrity and Provenance Hardening

## Outcome

AutoResearch now fails closed for experimental claims. A process exit code,
cached number, fixture, or model prediction cannot become a measured result or
`SUPPORTED` decision without a validated scientific evidence bundle.

## Forensic failure and root cause

The previous experiment service associated successful worker completion with
hard-coded SGKH/F8 metrics, three synthetic seeds, fixture ablations, and a
`SUPPORTED` string. The worker could execute `pytest .`, which validates code
but is not scientific execution. Report generation also allowed a model to
choose the conclusion. Those paths have been removed.

## Provenance and artifact contract

`experiments.integrity` introduces typed classifications: `MEASURED`,
`DERIVED`, `EXTRACTED_FROM_SOURCE`, `DATABASE_FACT`,
`CACHED_COMPATIBLE_RUN`, `MODEL_ESTIMATE`, `MODEL_HYPOTHESIS`, `FIXTURE`, and
`MISSING`. `PredictedMetric` is structurally `MODEL_HYPOTHESIS`; it cannot be
an observed metric.

Every analysis requires validated `implementation_diff`, `training_log`,
`checkpoint`, `seed_metrics`, `parameter_profile`, `flop_profile`,
`latency_profile`, and `evaluation_report` artifacts. It also validates the
identity tuple: candidate, method, repository commit, implementation/config
hashes, dataset/split checksums, seed, and metric version. Any mismatch returns
`CACHE_INCOMPATIBLE`; fixtures return `FIXTURE_PROHIBITED`.

## Deterministic analysis and reporting

Analysis consumes raw paired seed observations only, checks the declared seed
count, runs an exact two-sided signed-rank calculation, and records raw inputs,
statistic, p-value, and gates. The report writer is deterministic: model prose
cannot override `SUPPORTED`, `REFUTED`, `PARTIAL_SUPPORT`, or
`INSUFFICIENT_EVIDENCE`. Numeric ablation effects are `ABLATION_NOT_RUN` unless
measured descendant runs exist.

The audited +1.83% mAP case is explicitly tested with a +5% gate and produces
`REFUTED`, never `SUPPORTED`.

## Review and novelty safeguards

Review artifacts now expose papers ingested/parsed, full-text versus
abstract-only coverage, extracted and verified evidence, and failures. Scores
are labeled LLM evidence-support scores, not statistical confidence.

Novelty expected effects are stored as `MODEL_HYPOTHESIS`. Candidate-specific
external searches record queries, sources, retrieved papers, and timestamps.
When external retrieval is unavailable, the result is an
`EXPERT_REVIEW_REQUIRED` corpus-bounded assessment: “not found in the reviewed
corpus” is not a broader novelty claim.

## Audit and validation

Use `autoresearch audit run RUN_ID --json` to inspect claims, values,
provenance, source artifacts/runs, and verification state. The canonical
Claude Code skills remain project-managed; duplicate user-home skill copies
were moved aside to avoid divergent behavior.

Validation on this checkout:

| Command | Result |
| --- | --- |
| `.venv/bin/python -m compileall -q src` | passed |
| `.venv/bin/python -m pytest -q` | 85 passed, 1 skipped |
| `.venv/bin/python -m pip check` | no broken requirements |
| Integrity-focused tests | 7 passed (within full suite) |

## Remaining limitation

The local worker deliberately does not yet ingest a real measurement-artifact
bundle from an experiment implementation. It may validate and execute an
approved command, but will report `INSUFFICIENT_EVIDENCE` until such a bundle
is supplied. This is intentional scientific fail-closed behavior, not a
successful experimental result. No live provider or GPU run was claimed here.
