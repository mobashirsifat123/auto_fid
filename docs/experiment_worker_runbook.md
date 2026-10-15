# AutoResearch experiment worker runbook

The local worker is an approval-gated, typed-action executor. It is workspace-contained and filters its environment; it is not an OS sandbox and does not provide remote GPU execution.

## Start

```bash
cd /path/to/autoresearch
source .venv/bin/activate
export AUTORESEARCH_DATA_ROOT=$PWD/.autoresearch/data
export AUTORESEARCH_PROVIDER_CONFIG=$PWD/docs/live_config.example.toml
uvicorn autoresearch.api.app:create_app --factory --host 127.0.0.1 --port 8000
```

The local worker is composed into the API application and uses the workspace at `$AUTORESEARCH_DATA_ROOT/workspace`. The CLI is a separate process:

```bash
cd <companion-terminal-client-dir>
npx --yes bun@1.3.14 run --cwd packages/autoresearch-cli dev
```

## Approval lifecycle

1. Run `/experiment plan CANDIDATE_ID`.
2. Inspect `GET /v1/runs/RUN_ID`; the response includes the exact action plan and approval preview when the plan is available.
3. Approve the displayed checksum with `/approve RUN_ID SPEC_CHECKSUM` or `POST /v1/runs/RUN_ID/approve`.
4. Run `/experiment execute CANDIDATE_ID`. Execution is rejected if the checksum, worker, action plan, or expiry differs.
5. Follow events with the run view or SSE. `/cancel RUN_ID` is idempotent and terminates the process group.

## Recovery and limits

Worker jobs persist state, attempt, lease, heartbeat, process identifiers, log offset, and checksum in SQLite. Stale `LEASED`/`RUNNING` jobs are identified by `recover_stale`; completed jobs are deduplicated by specification checksum. Current local execution has one worker and one concurrent job, a 3600-second maximum runtime, bounded stdout/stderr, and no inherited provider/cloud credentials.

Metric analysis requires structured seed observations. Missing, malformed, non-finite, or absent metrics must remain a validation/evaluation failure; no metric defaults are generated.

## Storage

The database is `$AUTORESEARCH_DATABASE_PATH` or `$AUTORESEARCH_DATA_ROOT/autoresearch.sqlite3`. Reports and API artifacts are under `$AUTORESEARCH_ARTIFACT_ROOT`; scholarly PDFs are under `$AUTORESEARCH_DOWNLOAD_ROOT`. Back up the SQLite file and artifact tree together.

## Troubleshooting

- `approval_required` means planning completed but no exact approval exists.
- `checksum_mismatch` means the specification or typed action plan changed; create a new approval.
- `worker_unavailable` means the application container could not configure the local worker.
- `metric_schema_invalid` means the required structured metric artifact was not produced.
- `stale` means the heartbeat expired; inspect the persisted job and transitions before retrying.

Remote SSH, Slurm, Kubernetes, GPU execution, and Result-Conditioned Reinvention remain disabled.
