# 03 - System architecture (SIH26078)

## End-to-end flow

```text
┌─────────────────────────────────────────────────────────────────────┐
│ Weather Data                                                        │
│   • synthetic fixture  (generated locally, deterministic, seeded)   │
│   • ERA5 reanalysis     (historical archive, NOT live)              │
│   • local NetCDF file   (read-only)                                 │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Ingestion & Quality                                                 │
│   provider abstraction · retry with backoff · freshness verdict     │
│   variable whitelist · SHA-256 manifest                             │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Preprocessing                                                       │
│   contract validation → bounded gap fill → derived wind variables   │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Anomaly Detection                                                   │
│   climatology (month / day-of-year) · score · threshold ladder       │
│   → connected-component cell grouping → events                      │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Event Tracking                                                      │
│   cost-based matching (distance + intensity) · lifecycle states      │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Graph Construction                                                  │
│   temporal edges within a track · spatial edges between tracks       │
│   chronological split (train → validation → test)                   │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ GNN PROTOTYPE  [not validated against observed weather]             │
│   message passing → next-state delta for a tracked event            │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ AI DOWNSCALING PROTOTYPE  [loses to cubic interpolation on MAE]      │
│   residual CNN, ~12 km → ~5 km                                      │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Alert Engine  [severities are PROJECT categories, not IMD/WMO]      │
│   risk score → rule engine → lifecycle (acknowledge / resolve)      │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Realtime Orchestration                                              │
│   deterministic run identity · idempotency · per-stage status       │
│   optional model stages report SKIPPED, never an invented result    │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Evaluation & Explainability                                         │
│   every quantity tagged MEASURED / DERIVED / UNAVAILABLE /          │
│   INSUFFICIENT_DATA, with the evidence that produced it             │
└────────────────────────────┬────────────────────────────────────────┘
                             ↓
┌─────────────────────────────────────────────────────────────────────┐
│ Operations / Dashboard                                              │
│   /health · /ready · /api/v1/system/status · metrics · integrity    │
│   React + Leaflet dashboard, one panel per stage                    │
└─────────────────────────────────────────────────────────────────────┘
```

## Deployment topology

```text
┌────────────────────┐        ┌────────────────────┐
│  frontend          │        │  backend           │
│  React 19 + Vite   │───────▶│  FastAPI + uvicorn │
│  node:22-alpine    │  HTTP  │  python:3.13-slim  │
│  :5173             │  CORS  │  :8000             │
│                    │        │  non-root uid 1001 │
└────────────────────┘        └─────────┬──────────┘
     waits for                        │ volumes
     service_healthy                  ▼
                              ┌──────────────────┐
                              │ ./data  ./models │
                              └──────────────────┘
```

## Cross-cutting services

| Concern | Module | Why it exists |
| --- | --- | --- |
| Path safety | `security.py` | Refuse any caller-supplied path outside `DATA_DIR` |
| Redaction | `observability/logging.py` | Mask credential-shaped content in every log line and response |
| Metrics | `observability/metrics.py` | System counters only; deliberately no model-quality metrics |
| Timing | `observability/timing.py` | Per-stage duration into both the log and the registry |
| Middleware | `api/middleware.py` | `X-Request-ID` correlation and one error envelope |
| Integrity | `integrity.py` | SHA-256 over checkpoints and manifests |

## Key architectural decisions

**Compose, do not duplicate.** `/api/v1/system/status` reads the readiness
verdict from the operations router and model availability from the GNN and
downscaling services, rather than recomputing them. Two sources of truth would
drift, and a status page that disagrees with `/ready` is worse than none.

**Liveness is not readiness.** `/health` never touches the filesystem or a
checkpoint, so a missing model cannot restart-loop a healthy container.
`/ready` answers the different question and reports `READY` / `DEGRADED` /
`NOT_READY`.

**A failed stage stops the pipeline.** Running anomaly detection on data that
failed its quality gate produces plausible-looking nonsense. The exceptions are
the optional model stages, which report `SKIPPED` — never `FAILED`, and never a
substituted prediction.

**One orchestrator, many entry points.** The CLI, the API and the demo all call
`PipelineOrchestrator`. A stage that is reachable one way and not the other is a
stage whose behaviour nobody has tested.
