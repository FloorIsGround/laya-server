# Laya Internal Inference API

This project keeps one Laya checkpoint resident and exposes its typed
System-One decisions over HTTP. The service defaults to the English
`convaiinnovations/laya` checkpoint on CPU so it does not compete with Ollama
for GPU memory.

## Endpoints

- `GET /health` — process liveness; no authentication required
- `GET /ready` — model readiness; no authentication required
- `GET /v1/model-info` — loaded checkpoint and runtime information
- `GET /v1/worker-status` — authenticated gateway worker status and load counters
- `POST /v1/systemone` — Laya-compatible typed inference
- `POST /v1/evaluate` — compatibility alias for `/v1/systemone`
- `GET /docs` — interactive OpenAPI documentation

All `/v1` endpoints require `Authorization: Bearer ...` when `LAYA_API_KEY` is
set. They also accept the same token in the `X-API-Key` header.

In the `/docs` web UI, select **Authorize** near the top of the page and enter
the token under either `BearerAuth` or `ApiKeyAuth`. Enter only the token for
`BearerAuth`; Swagger adds the `Bearer` prefix automatically. Authorization is
retained across page reloads in that browser.

## Request example

```json
{
  "state": {
    "subject": "Duplicate charge",
    "body": "Please refund the second charge or we will cancel."
  },
  "questions": {
    "department": {
      "type": "choice",
      "instructions": "Which team should handle this request?",
      "criteria": {
        "billing": "invoices, payments, refunds",
        "technical": "bugs, outages, integrations",
        "sales": "pricing and new purchases",
        "other": "everything else"
      }
    },
    "churn_risk": {
      "type": "noul",
      "instructions": "Does the customer threaten to cancel?"
    }
  }
}
```

## Development

```bash
uv sync --group dev
uv run pytest
LAYA_DEVICE=cpu uv run uvicorn server:app --host 127.0.0.1 --port 8000
```

The model is loaded during application startup. Do not enable Uvicorn reload in
production because every reload reconstructs the model.

Run the external HTTP load test after the service is ready:

```bash
uv run python scripts/benchmark_api.py --requests 40 --concurrency 8
```

## Configuration

The service reads these environment variables:

| Variable | Default | Purpose |
|---|---:|---|
| `LAYA_MODEL` | `convaiinnovations/laya` | Hub ID or local checkpoint path |
| `LAYA_SUBFOLDER` | unset | Optional bundled checkpoint subfolder |
| `LAYA_DEVICE` | `cpu` | PyTorch device |
| `LAYA_API_KEY` | required | Bearer token protecting `/v1`; startup fails when omitted |
| `LAYA_CONCURRENCY` | `1` | Simultaneous model calls per process |
| `LAYA_QUEUE_CAPACITY` | `64` | Waiting requests per process |
| `LAYA_QUEUE_TIMEOUT_SECONDS` | `30` | Maximum queue wait |
| `LAYA_TORCH_THREADS` | `12` | CPU threads used by this process |
| `LAYA_MAX_CONTEXT` | `512` | Per-question encoder window (256-8192, capped by encoder capacity) |
| `LAYA_WORKER_ID` | `laya-worker` | Stable deployment identity reported to gateways |
| `LAYA_REQUIRE_CONFIGURED_DEVICE` | `true` | Return unready if the live device drifts from `LAYA_DEVICE` |

## Gateway worker protocol

Worker protocol version 1 adds request-correlation headers to every response:
`X-Request-ID`, `X-Laya-Worker-ID`, `X-Laya-Worker-Instance`, and
`X-Laya-Protocol-Version`. A valid printable `X-Request-ID` supplied by the
gateway is echoed; otherwise the worker generates a UUID.

`/v1/worker-status` reports configured capacity, live in-flight/executing/queued
counts, cumulative completion/failure/rejection counters, actual model device,
max context, question/option head limit, tokenizer fingerprint,
and a process-instance UUID that changes after restart. Counters are process
local and reset with that instance UUID.

The context override is applied after the checkpoint loads; it does not change
the question/option head budget or calibration. A longer window costs more GPU
memory and time, especially with multiple questions. Benchmark and evaluate
long-document accuracy before changing production configuration.

For GPU deployments, set `LAYA_REQUIRE_CONFIGURED_DEVICE=true`. Laya can fall
back to CPU after a CUDA memory failure; strict readiness makes that degraded
worker return 503 instead of continuing to receive gateway traffic.

For CPU serving, change process count and PyTorch thread count together. For
example, two Uvicorn workers should start near six Torch threads each to avoid
oversubscribing the 12 physical CPU cores. Benchmark before increasing either.

### Example CPU baseline

Measured on the 12-core Ryzen 9 7900X with the two-question fixture in
`tests/fixtures/request.json` and eight concurrent clients:

| Processes | Torch threads/process | Requests/second | P50 latency | P95 latency |
|---:|---:|---:|---:|---:|
| 1 | 12 | 7.185 | 1,110 ms | 1,118 ms |
| 2 | 6 | 6.876 | 1,126 ms | 1,189 ms |
| 4 | 3 | 3.732 | 1,419 ms | 3,389 ms |

One process is the measured production default. A single warm request took
143 ms; the higher concurrent latency above includes queueing behind the one
resident model.

## systemd user service

The unit in `deploy/laya-api.service` binds to port 8000 with one process and 12
Torch threads. Install and start it with:

```bash
systemctl --user link "$HOME/laya-server/deploy/laya-api.service"
systemctl --user daemon-reload
systemctl --user enable --now laya-api.service
```

To keep the user service running after logout and across reboots, an
administrator must enable lingering once:

```bash
sudo loginctl enable-linger "$USER"
```

## Calling the service

```bash
curl --fail-with-body \
  -H "Authorization: Bearer $LAYA_API_KEY" \
  -H "Content-Type: application/json" \
  --data @request.json \
  http://127.0.0.1:8000/v1/systemone
```
