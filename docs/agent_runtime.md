# Unified AutoResearch agent runtime

AutoResearch opens in a normal conversational agent session. The default mode
does not activate a scientific pipeline. The Python backend owns the session,
turn, routing, tool, permission, budget, provider, artifact, and event state;
the terminal client is only a controller and renderer.

## Startup

The backend must be started with the provider configuration and the directories
that the agent may inspect:

```bash
cd /path/to/autoresearch
export AUTORESEARCH_AGENT_ROOTS=/path/to/autoresearch:<companion-terminal-client-dir>
autoresearch-api
```

The CLI reads only its backend URL and the name of an optional backend-auth
token environment variable. Provider keys are never read by the CLI.

## Session and turn model

`agent_sessions`, `agent_messages`, `agent_turns`, `agent_tool_calls`,
`agent_permissions`, `agent_artifact_refs`, `agent_capability_suggestions`,
`agent_events`, and `agent_idempotency` are durable SQLite records. A turn is
admitted before its asynchronous worker starts. The bounded loop allows at
most four model calls, three tool iterations, the configured time budget, and
the configured token/cost budget. Terminal states are `completed`, `failed`,
`cancelled`, `awaiting_permission`, and `awaiting_confirmation`.

The event stream is replayable with `Last-Event-ID`. A client can reconnect to
the same session without creating another turn. Pipeline run IDs and artifact
IDs are linked back into the session rather than copied into chat text.

## Modes and routing

* Default Agent: ordinary repository, file, Git-inspection, and short command assistance.
* Study Search: explicit `/use study-search`, `/discover`, or `@discover`.
* Literature Review: explicit `/use review`, `/review`, or `@review`.
* Novelty: explicit `/use novelty`, `/invent`, or `@invent`.
* Experiment: explicit `/use experiment`, `/experiment`, or `@experiment`.
* Reinvention: visible but disabled because no backend pipeline exists.

`/auto on` enables conservative deterministic suggestions. Suggestions for
novelty and experiment work always require confirmation. A declined suggestion
returns to ordinary chat. A one-turn `@...` override does not change the
persistent session mode.

## Tool policy

Read-only repository tools are contained by the session repository path.
`write_file` and `terminal_run` create scoped permission requests. Generic
terminal execution passes an argument vector to `create_subprocess_exec`,
never a shell string, and permits only short development/test commands. It
does not receive provider credentials. Scientific execution remains a separate
checksum-bound `LocalTypedWorker` path and cannot be reached by normal chat.

## Capability states

Capability status comes from `GET /v1/capabilities`. Default chat and the
repository tools are enabled when a suitable ModelRouter role is configured.
Study Search, Literature Review, Novelty, and Experiment Design are enabled by
the current backend. Local Experiment Execution is conditional and approval
gated. Remote GPU Execution and Result-Conditioned Reinvention are disabled
with explicit reasons.

## Security and recovery

Session events, tool arguments, observations, errors, and user-visible
messages are redacted before durable event/history persistence. Backend auth is
optional for local development and uses a bearer token environment variable
when enabled. Session and mutation routes accept `Idempotency-Key`. A pending
permission can be approved or denied through the API; approval resumes the
durable turn. Cancellation marks the session and cancels the local task/run.
