# 09 - API summary (SIH26078)

Base URL: `http://localhost:8000`. Interactive docs at `/docs`; the OpenAPI
schema at `/openapi.json`.

## Start here

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/v1/system/status` | **The single best request.** Identity, phase, environment, readiness, data source and its timestamp, per-component availability, stored counts, and the prototype disclaimer. |
| GET | `/health` | Liveness. Never touches the filesystem or a model. |
| GET | `/ready` | Readiness: `READY` / `DEGRADED` / `NOT_READY`, with components and notes. |

## Data

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/api/v1/datasets` | NetCDF files under `DATA_DIR` |
| GET | `/api/v1/datasets/{path}/summary` | Validated metadata; rejects path traversal |
| POST | `/api/v1/datasets/validate` | Dry-run the data contract |
| GET | `/api/v1/data/sources` | Registered ingestion sources and their requirements |
| GET | `/api/v1/data/datasets` | Ingested datasets with manifests and checksums |
| POST | `/api/v1/data/validate` | Quality-check a file without writing |

## Events and tracking

| Method | Path |
| --- | --- |
| GET | `/api/v1/events` |
| GET | `/api/v1/events/{event_id}` |
| GET | `/api/v1/tracks` |
| GET | `/api/v1/tracks/{track_id}` |
| GET | `/api/v1/tracks/{track_id}/trajectory` |

## Models (both optional)

| Method | Path | When no checkpoint exists |
| --- | --- | --- |
| GET | `/api/v1/gnn/status` | 200, `available: false`, with a reason |
| GET | `/api/v1/gnn/model` | Architecture and feature contract, still served |
| GET | `/api/v1/gnn/predict` | 503 with actionable guidance |
| GET | `/api/v1/downscaling/status` | 200, `available: false`, with a reason |
| GET | `/api/v1/downscaling/model` | Contract still served |
| POST | `/api/v1/downscaling/run` | 503 with actionable guidance |

A missing model is a **documented state, not an error**. The status endpoints
answer 200 with `available: false`; the endpoints that genuinely cannot work
return 503 with a message that says what to do.

## Alerts

| Method | Path |
| --- | --- |
| GET | `/api/v1/alerts` |
| GET | `/api/v1/alerts/{alert_id}` |
| POST | `/api/v1/alerts/generate` (503 if `ALERT_GENERATION_ENABLED=false`) |
| POST | `/api/v1/alerts/{id}/acknowledge` |
| POST | `/api/v1/alerts/{id}/resolve` |

Severities are this project's own categories, not official IMD or WMO warnings.

## Orchestration

| Method | Path |
| --- | --- |
| GET | `/api/v1/pipeline/status` |
| GET | `/api/v1/pipeline/runs` |
| GET | `/api/v1/pipeline/runs/{run_id}` |
| POST | `/api/v1/pipeline/run` (`?dry_run=true` plans without writing) |
| GET | `/api/v1/ingestion/status` |

## Evaluation and validation

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/api/v1/evaluation/status` | Whether anything has been evaluated |
| GET | `/api/v1/evaluation/runs` | Stored evaluation records with provenance |
| GET | `/api/v1/validation/status` | Phase 7 leakage-free validation state |
| GET | `/api/v1/validation/report` | Full report |

## Operations (Phase 10)

| Method | Path | Notes |
| --- | --- | --- |
| GET | `/api/v1/operations/status` | Same verdict as `/ready`, versioned |
| GET | `/api/v1/operations/metrics` | System counters and histograms. **No model-quality metrics by design.** |
| GET | `/api/v1/operations/integrity` | SHA-256 over checkpoints and manifests |

## Conventions

**Status codes.** `404` the id is unknown. `422` the request or the data is
invalid — retrying will not help. `503` a provider or optional dependency is
unavailable — retrying might. The distinction is the useful part.

**Error envelope.** Every error uses one shape:

```json
{
  "error": "not_found",
  "message": "human-readable detail",
  "request_id": "req-8f3a...",
  "details": {}
}
```

The status code is preserved exactly and the original `detail` becomes
`message`, so no pre-Phase-10 assertion was weakened. A 500 never exposes a
stack trace, an exception repr or an internal path.

**Correlation.** Every response carries `X-Request-ID`. A client-supplied id is
honoured only if it looks like an id (bounded, printable, no control
characters) — reflecting arbitrary text into a header is an injection vector.

**Unavailable is null.** Where a quantity is unknown it is `null` with a
reason, never `0`. `/api/v1/system/status` is the clearest example: with no
evaluation on record, `evaluation.available` is `false` and `counts.evaluation_runs`
is `null`.
