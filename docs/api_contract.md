# AutoResearch API contract

The FastAPI OpenAPI document is generated with:

```bash
.venv/bin/python -m autoresearch.api.openapi > /tmp/autoresearch-openapi.json
npx --yes openapi-typescript /tmp/autoresearch-openapi.json -o ../autoresearch-cli/packages/autoresearch-cli/src/generated/schema.d.ts
```

`POST /v1/projects/{project_id}/invent` accepts an optional validated `gap_id` and returns a durable run immediately. `POST /v1/projects/{project_id}/experiment` accepts `candidate_id` and a mode (`plan`, `validate`, `execute`, `inspect`, or `report`); execution capability remains conditional. Mutations honor `Idempotency-Key`. Events replay through `/v1/runs/{run_id}/events` using `Last-Event-ID`.

`POST /v1/projects/{project_id}/reinvent` is an explicit capability gate. It
returns HTTP 409 with `capability_unavailable` until the backend pipeline is
implemented; it never fabricates a run or result. Run SSE drains all durable
event pages and only closes after the terminal history is exhausted.

The capability endpoint is authoritative: discovery, review, novelty, and experiment design are enabled; experiment execution reports its approval/worker/isolation constraints; reinvention is disabled.

## Conversational agent routes

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

`POST /turns` returns a durable `turn_id` immediately. A turn uses the backend
`ModelRouter`, can perform at most four provider calls and three generic tool
iterations, and emits typed events such as `agent.tool_proposed`,
`agent.permission_required`, `agent.response_delta`, and
`agent.turn_completed`. Agent SSE accepts both `last_event_id` and
`Last-Event-ID` and replays durable events.

`/v1/capabilities` now contains both the legacy `pipelines` compatibility view
and the authoritative `capabilities`/`tools` registry. The client disables
capabilities based on the returned state/reason; it does not infer scientific
availability locally.

All mutating agent routes accept `Idempotency-Key`. Provider credentials never
appear in request or response schemas. Session/history/event persistence uses
the backend redaction boundary.
