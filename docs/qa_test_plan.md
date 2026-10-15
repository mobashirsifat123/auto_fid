# AutoResearch QA test plan

## Scope and quality gates

This plan validates the runnable product, not future novelty, experiment
execution, reinvention, browser frontend, or SSH workers. It covers API
contracts, durable runs, SSE, reconnect/resume, cancellation, permissions,
provider routing boundaries, secret handling, terminal behavior, and the
deterministic discovery/review vertical slice.

The pass gates are:

1. backend unit/API/security tests pass;
2. Python compile and dependency checks pass;
3. CLI typecheck, tests, generated-contract check, and production build pass;
4. fake-backend end-to-end discovery/review/reconnect/artifact/usage flow passes;
5. real local PTY launch reaches the backend and reports empty provider state
   honestly;
6. no secrets appear in API responses, client requests, logs, or reports;
7. P0/P1 defects are zero and resolved low-risk P2 defects are documented.

## Exact deterministic commands

Backend:

```bash
cd /Users/mobashirsifat/autoresearch
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q src
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q tests/qa
```

API contract:

```bash
cd /Users/mobashirsifat/autoresearch
.venv/bin/python -m autoresearch.api.openapi > /tmp/autoresearch-openapi.json
```

CLI:

```bash
cd /Users/mobashirsifat/autoresearch-cli
npx --yes bun@1.3.14 install --frozen-lockfile
npx --yes bun@1.3.14 run --cwd packages/autoresearch-cli typecheck
npx --yes bun@1.3.14 run --cwd packages/autoresearch-cli test
npx --yes bun@1.3.14 run --cwd packages/autoresearch-cli check-contract
npx --yes bun@1.3.14 run --cwd packages/autoresearch-cli build
```

Deterministic TUI vertical slice:

```bash
cd /Users/mobashirsifat/autoresearch-cli
npx --yes bun@1.3.14 test test/e2e.test.ts test/app.test.tsx test/api.test.ts
```

## Local PTY configuration

Provider keys are not required for this QA path. The backend may be launched in
an intentionally empty state:

```bash
cd /Users/mobashirsifat/autoresearch
AUTORESEARCH_DATA_ROOT=/tmp/autoresearch-qa-data \
AUTORESEARCH_API_PORT=8765 \
AUTORESEARCH_AGENT_ROOTS=/Users/mobashirsifat/autoresearch:/Users/mobashirsifat/autoresearch-cli \
.venv/bin/autoresearch-api

cd /Users/mobashirsifat/autoresearch-cli
AUTORESEARCH_BACKEND_URL=http://127.0.0.1:8765 \
npx --yes bun@1.3.14 run --cwd packages/autoresearch-cli dev
```

Expected empty-state behavior: `/v1/health` is ready, `/v1/providers/status`
returns empty providers/roles, the TUI shows `backend online`, and model-backed
operations fail truthfully with a missing-role/provider message. A live external
provider run requires an operator-provided secret-free TOML file whose keys are
environment-variable names only; this QA cycle did not invent or use paid keys.

## Test matrix

| Area | Evidence | Expected result |
|---|---|---|
| API health/capabilities | API tests, OpenAPI export | Ready and required routes present |
| Auth/redaction | API/provider tests, `tests/qa` | Unauthorized access rejected; credentials absent/redacted |
| Idempotency | project/run/cancel/approval/agent tests | Retry returns original durable object and no duplicate action |
| SSE ordering | backend and CLI API tests | Ordered IDs, cursor validation, malformed-event warning, full-page drain |
| Resume | discovery/review resume tests | Same run resumes from checkpoints without duplicate durable work |
| Cancellation | run and agent cancellation tests | Terminal status is cancelled; late approval cannot execute |
| Permissions | agent runtime tests | Mutations stop at approval and execute only after valid approval |
| Source adapters | existing provider/source contract tests | HTTP contracts and failure isolation are deterministic; no live count claimed |
| TUI layout | OpenTUI renderer test | Online, empty, disabled, and narrow resize states render |
| Discovery/review | fake backend E2E | 20-paper discovery, review, top-five/gap artifact path, usage, reconnect |
| Artifact handling | API/client artifact tests | Lazy bounded reads expose explicit expansion marker |
| Performance | local health and 1,000-event probes | Measured and recorded; no silent event truncation |

## Live-integration boundary

This is a QA/product-polish cycle. It does not claim live Semantic Scholar,
OpenAlex, Crossref, arXiv, OpenReview, Unpaywall, LLM, or embedding success
without credentials and a controlled seed. Existing adapter contract tests and
the prior live-integration report remain the evidence for those integrations;
this cycle records external-provider execution as not run rather than fabricating
counts.
