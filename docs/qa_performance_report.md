# AutoResearch QA performance report

Measured 2026-08-02 on the local macOS development environment. These are
responsive-local measurements, not production capacity claims. No external
provider request was made and no paid credential was used.

## Measurements

| Probe | Result | Method |
|---|---:|---|
| API app creation | 39.70 ms | `create_app(Settings(...))` with missing provider config |
| API TestClient startup | 114.47 ms | startup through `TestClient` context with missing provider config |
| API health request | 0.994–5.938 ms | five live `curl` requests against the final local server; final TestClient probe 2.23 ms |
| CLI cold first output | 7,472.56 ms | real `npx --yes bun@1.3.14 ... dev` process, backend unavailable |
| CLI cold branded screen | 8,039.93 ms | same real process, first captured `AUTORESEARCH` frame |
| CLI production bundle | 188 ms bundler work | Final Bun build; command also includes external `npx` startup |
| Session creation | 14.50 ms | local TestClient, empty provider installation |
| Capability listing (`/tools` backend source) | 1.91 ms | `/v1/capabilities` |
| Run listing | 2.00 ms | `/v1/runs?project_id=...` |
| Discovery admission | 7.85 ms | `POST /discover`, durable `run_id` returned before provider failure |
| First durable event read | 25.83 ms | `run.created` present in SSE response |
| Artifact listing | 3.24 ms | `GET /v1/runs/{run_id}/artifacts` |
| Reconnect read | 4.43 ms | `Last-Event-ID: 1` on the same run |
| 1,000-event SSE replay | 1,000 events, 2,039.33 ms before page-loop optimization | Initial scale probe exposed QA-008 |
| 1,000-event SSE replay after fix | 18.10 ms; 1,000 events returned | Full pages drain without the 400 ms inter-page sleep |

The first 1,000-event result is retained as defect evidence. The final suite
must rerun the probe after QA-008 and record the post-fix value here.

## Memory and rendering safeguards

- API SSE reads bounded database pages of 200 events and drains pages by cursor.
- CLI artifact reads are offset/size bounded and mark truncation explicitly.
- The TUI renders a bounded capability list and compact labels on narrow
  terminals; it does not load PDFs into the client.
- Client event history is incrementally projected from SSE events rather than
  requesting a full provider or PDF payload.
- Backend source/download concurrency and provider rate limits remain the
  controlling scale mechanisms; this QA cycle does not override them.

## Limits and follow-up

No claim is made for 100-paper live throughput, provider latency, GPU execution,
or remote workers in this report. A representative paid-provider benchmark
requires an explicit budget, provider keys, a fixed seed, and a capture of
request/token/cost records. That benchmark belongs to live integration/scale
validation, not an uncredentialed QA run.
