# Unified AutoResearch agent runtime report

## Scope completed

AutoResearch now opens as a normal conversational terminal agent. The Python
backend remains the only owner of provider routing, scientific pipeline logic,
budgets, persistence, permissions, artifacts, and durable events. The sibling
terminal client is a controller and renderer; it does not contain provider
credentials or research decision logic.

The implemented vertical slice covers:

* durable agent sessions, messages, turns, tool calls, observations,
  permissions, suggestions, artifact references, events, and idempotency;
* bounded default chat with deterministic intent routing and backend model
  selection;
* contained repository inspection, code search, Git inspection, permission-
  gated file writes, and a short allowlisted terminal command tool;
* explicit Study Search, Literature Review, Novelty, and Experiment modes;
* authoritative capability status with disabled reasons for remote execution
  and Result-Conditioned Reinvention;
* reconnectable SSE with `Last-Event-ID`, durable turn admission,
  cancellation, scoped approval, and retry-safe mutations;
* generated TypeScript contracts from the FastAPI OpenAPI document;
* terminal rendering for typed agent and research events, artifacts, usage,
  routing, capability states, errors, and recovery notices.

## Exact launch configuration

Provider credentials are configured only in the backend environment. The CLI
receives only the backend URL and, optionally, the name of a backend-auth token
environment variable.

```bash
cd /Users/mobashirsifat/autoresearch
. .venv/bin/activate
export AUTORESEARCH_AGENT_ROOTS=/Users/mobashirsifat/autoresearch:/Users/mobashirsifat/autoresearch-cli
autoresearch-api
```

In a second terminal:

```bash
cd /Users/mobashirsifat/autoresearch-cli
npx --yes bun@1.3.14 run --cwd packages/autoresearch-cli dev
```

Optional client configuration is stored at
`~/.config/autoresearch/config.toml` and supports only backend URL, backend
auth-token variable name, default project, theme, refresh interval, artifact
open command, and confirmation policy. Provider-key environment-variable names
are rejected.

## Backend contract

The agent routes are:

```text
GET    /v1/agent/sessions
POST   /v1/agent/sessions
GET    /v1/agent/sessions/{session_id}
POST   /v1/agent/sessions/{session_id}/turns
GET    /v1/agent/sessions/{session_id}/events
POST   /v1/agent/sessions/{session_id}/mode
POST   /v1/agent/sessions/{session_id}/cancel
POST   /v1/agent/suggestions/{suggestion_id}
POST   /v1/agent/permissions/{permission_id}
```

`POST /turns` persists and returns a `turn_id` before work starts. The worker
emits typed durable events including routing, reasoning, tool proposal,
permission request, tool progress, response deltas, artifact references,
warnings, failures, and terminal completion. SSE replay starts after the
client's last event ID and does not create another turn.

Every mutating agent route accepts `Idempotency-Key`. Repeated requests return
the persisted response and do not duplicate the session, turn, event, approval,
or cancellation operation.

## Capability policy

| Capability | State | Activation |
|---|---|---|
| Default Agent | enabled when a suitable backend role is configured | normal text |
| Repository tools | enabled | selected by the backend model |
| Study Search | enabled | `/use study-search`, `/discover`, or `@discover` |
| Literature Review | enabled | `/use review`, `/review`, or `@review` |
| Novelty Generation | enabled | explicit `/use novelty`, `/invent`, or `@invent` |
| Experiment Design | enabled | explicit `/use experiment`, `/experiment`, or `@experiment` |
| Local Experiment Execution | conditional and approval-gated | backend worker and policy required |
| Remote GPU Execution | disabled | no remote worker implementation |
| Result-Conditioned Reinvention | disabled | Python pipeline not implemented |

The client renders the backend's state and reason. It does not infer or fake
unfinished capability availability.

## Security and isolation

Agent roots are explicit and path-contained. File and repository tools reject
absolute paths, traversal, symlink escapes, oversized reads, and oversized
writes. Terminal execution uses `create_subprocess_exec` with an argument
vector, a minimal environment, bounded runtime, bounded output, and an
allowlist for test/build commands. No shell string is accepted.

Provider credentials are never read by the CLI and are not passed to generic
terminal tools. Durable agent messages, event payloads, tool arguments,
permissions, observations, errors, API provider status, and CLI-visible errors
pass through redaction. Backend auth, when enabled, is bearer-token based and
the token value is not returned by provider or session endpoints.

## Verification

Backend validation completed:

```text
.venv/bin/python -m pytest -q
53 passed, 1 skipped
.venv/bin/python -m compileall -q src
pass
.venv/bin/python -m pip check
No broken requirements found.
```

CLI validation completed from `/Users/mobashirsifat/autoresearch-cli`:

```text
npx --yes bun@1.3.14 install --frozen-lockfile
pass; no dependency changes
npx --yes bun@1.3.14 run --cwd packages/autoresearch-cli typecheck
pass
npx --yes bun@1.3.14 run --cwd packages/autoresearch-cli test
15 passed, 0 failed, 50 expect() calls
npx --yes bun@1.3.14 run --cwd packages/autoresearch-cli build
pass; 56 modules bundled
```

The deterministic CLI harness covers project/run/artifact/usage/capability
flows, reconnect, terminal resize, typed event rendering, command parsing,
generated API methods, backend errors, and provider-key exclusion. Backend
fixtures cover conversational turns, scoped write approval and resume,
session mode, SSE event persistence, idempotent turns, unavailable
Reinvention, cancellation/approval mutations, provider usage metadata, and
durable secret redaction.

## Deliberate limitations

Novelty Generation and Experiment Design are contract-backed and routed to the
existing Python services, but they are not presented as proven novelty or as
unapproved execution. Remote workers and Result-Conditioned Reinvention remain
disabled with explicit capability reasons. A recovered permission request has
redacted durable arguments; exact arguments are retained only for the active
in-process call and must be re-proposed if the process has to recover it.

The requested Git audit, branch/tag operations, and commit metadata were
omitted from this implementation because the user explicitly requested that
Git-related work be skipped. No Git state is asserted by this report.

## Primary implementation files

* `/Users/mobashirsifat/autoresearch/src/autoresearch/agent/runtime.py`
* `/Users/mobashirsifat/autoresearch/src/autoresearch/agent/store.py`
* `/Users/mobashirsifat/autoresearch/src/autoresearch/agent/tools.py`
* `/Users/mobashirsifat/autoresearch/src/autoresearch/agent/capabilities.py`
* `/Users/mobashirsifat/autoresearch/src/autoresearch/agent/routing.py`
* `/Users/mobashirsifat/autoresearch/src/autoresearch/api/app.py`
* `/Users/mobashirsifat/autoresearch/src/autoresearch/storage/repository.py`
* `/Users/mobashirsifat/autoresearch-cli/packages/autoresearch-cli/src/app.tsx`
* `/Users/mobashirsifat/autoresearch-cli/packages/autoresearch-cli/src/api.ts`
* `/Users/mobashirsifat/autoresearch-cli/packages/autoresearch-cli/src/commands.ts`
* `/Users/mobashirsifat/autoresearch-cli/packages/autoresearch-cli/src/generated/schema.d.ts`
