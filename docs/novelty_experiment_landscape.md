# Autonomous research and experiment-agent landscape

Retrieved 2026-08-01 from primary project repositories and papers. This is an architecture study, not a benchmark claim.

| System | Relevant architecture and strengths | Weaknesses / constraints | AutoResearch decision |
|---|---|---|---|
| [The AI Scientist](https://github.com/SakanaAI/AI-Scientist) | Idea generation, literature search, code iteration, experiments, manuscript and review in template repositories | Upstream explicitly warns that it executes LLM-written code and recommends containerization/network restriction; its source-code license includes conditions beyond a permissive OSS license | Reuse no code. Adopt staged experiments, explicit safety policy, independent critique, and provenance; reject direct model-generated execution |
| [Robin](https://github.com/Future-House/robin) | Multi-agent scientific discovery with specialized literature and experimental roles; public notebook records parameters used in the associated work | Domain- and demonstration-specific; does not by itself establish general novelty or reproducibility | Adopt role separation and immutable evidence handoffs; retain AutoResearch’s provider/router and typed service boundaries |
| [OpenHands](https://github.com/OpenHands/OpenHands) | Mature event-driven agent runtime with container-backed execution options, resumable sessions, and operational UX | Primarily a software-engineering agent; repository contains mixed licensing for enterprise areas | Reuse no code. Apply typed runtime actions, isolation capability disclosure, cancellation, and replay principles |

## Defensible design implications

1. Candidate generation and novelty assessment use different model roles; no generating model self-certifies novelty.
2. Retrieval records preserve section-level excerpts, counter-evidence, and search-scope uncertainty.
3. Scientific gates are explicit and scores only rank candidates that pass them.
4. Experiment plans are declarative; execution resolves only typed allowlisted actions with `shell=False`.
5. Approval is bound to a canonical checksum and invalidated by any material specification change.
6. Statistics are deterministic, seed-aware, and report insufficient evidence instead of significance for one seed.

No third-party code was copied during this study.
