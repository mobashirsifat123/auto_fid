# Reuse and migration report

## Source boundary

The audit was performed against `/Users/mobashirsifat/qwen-ui`. The source
repository was read only during this migration. No database, model service,
frontend bundle, benchmark output, generated asset, or machine-specific
deployment script was copied.

The complete per-file decision record is [reuse_manifest.csv](reuse_manifest.csv).

## Exact reused provider abstractions

The following classes/functions were ported with package-name or focused
production changes:

- `Capability`, `ModelRole`, and `ROLE_REQUIREMENTS`;
- `ProviderConfig`, `ModelAssignment`, and `reject_secret_fields`;
- `ErrorKind`, `ProviderError`, and `kind_for_status`;
- `with_retry`, `redact`, `redact_text`, and safe cost estimation;
- `ProviderRegistry` and `ModelRouter`;
- `CompletionResult`, `HealthResult`, `ModelInfo`, `ToolCallResult`, and provider protocols;
- `HTTPProviderBase`, `OpenAICompatibleProvider`, `AnthropicProvider`, and `GeminiProvider`;
- `EmbeddingProvider` and `ExternalEmbeddingProvider`;
- deterministic `FakeLLMProvider` and `FakeEmbeddingProvider` for tests only;
- `ProviderBudgetGuard` policy concepts.

Focused changes are documented in the manifest: router-only embeddings,
rate limiting, application provenance, the new repository usage sink, and a
clean TOML composition root.

## Extracted ideas, not copied implementations

- `PaperIngestionPipeline`: page-aware PDF parsing, scientific chunking, and
  checksums were split into `ingestion/document_parser.py`, `chunking.py`, and
  `extraction.py`. Original schema calls and direct embedding dependencies were
  removed.
- `HybridRetriever`: transparent similarity components were reimplemented in
  `review/similarity.py`; original FTS/retrieval-trace coupling was not copied.
- `ClaimExtractor` / `EvidenceGraph`: strict evidence validation informed
  `PaperReviewExtractor`, but the original graph schema was not copied.
- `GapDiscoverer`: evidence-linked gap concepts were retained with stricter
  unknown-ID validation and mandatory hypothesis labeling.
- `ResearchService`: only the discover/review orchestration interface was
  reimplemented. Experiment, manuscript, workspace, UI, and legacy memory
  responsibilities were rejected.
- `ResearchStore`: the migration discipline and parameterized SQLite approach
  were retained in a new focused schema; none of the old tables/data were copied.

## Rejected files and reasons

| Source | Reason |
|---|---|
| `src/ui/` | Legacy Gradio UI and generated static assets |
| existing `frontend/` | Current UI is outside milestone one and contains product-specific assets |
| local model scripts/config | Qwen/vLLM/Ollama/local embedding and machine-specific infrastructure are prohibited |
| benchmark and report outputs | Generated data is not production source |
| memory/runtime databases | User/demo state must never migrate into a clean repository |
| hybrid/local model clients | New architecture uses external providers through `ModelRouter` only |
| experiment/reinvention implementations | Explicitly outside this milestone |

## Dependency changes

The runtime requires only `httpx`, `pypdf`, and `typer`. Provider vendor SDKs are
not required because adapters use documented HTTP contracts. Tests add `pytest`
and `pytest-asyncio`. There are no local inference, GPU-runtime, Gradio, FastAPI,
React, benchmark, or model-weight dependencies in milestone one.
