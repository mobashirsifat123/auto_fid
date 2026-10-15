# Backend Responsibility Audit

This document classifies the modules in the `autoresearch` Python backend.

## Preserve (Scientific Service Responsibilities)
These modules contain the actual research logic and will remain as the core API service:
* `autoresearch.novelty.*` - Novelty candidate generation, assessment, and prior-art checking.
* `autoresearch.review.*` - Deep literature review, extracting contradictions and limitations.
* `autoresearch.experiments.*` - Translating novelty into ExperimentSpecs, causal mechanisms, budget planning.
* `autoresearch.workers.*` - Approval-bound isolated scientific worker execution.
* `autoresearch.papers.*` - Study search, metadata fetching, semantic ranking.
* `autoresearch.storage.*` - Persistence of research artifacts, database layer.
* `autoresearch.domain.models` - Core models (PipelineKind, RunStatus, ArtifactProvenance).

## Stop Using for Normal OpenCode Behavior (Deprecated)
These modules are remnants of the incorrect custom conversational architecture and should no longer be used as generic chat runtimes by the CLI:
* `autoresearch.agent.session` - Duplicate of OpenCode's native session state.
* `autoresearch.agent.tools` - Duplicate generic tools (file read/write, bash execution) already provided by OpenCode.
* `autoresearch.agent.permissions` - Duplicate TUI prompt/approval system for arbitrary file edits.
* `autoresearch.api.events` / `autoresearch.api.agent` - The custom Server-Sent Events stream powering the parallel TUI interface in the legacy `autoresearch-cli` package.

The Python backend will expose a small research-specific API namespace:
```text
/v1/projects/{id}/discover
/v1/projects/{id}/review
/v1/projects/{id}/invent
/v1/projects/{id}/experiment
```
Normal coding queries will be handled purely by native OpenCode providers and tools in the client.
