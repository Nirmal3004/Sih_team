# SIH demo guide (SIH26078, Phase 11)

> **This is a research prototype.** It is not an operational disaster-warning
> or emergency-response system, it is not validated against observed weather or
> an official warning catalogue, and it demonstrates no forecast skill. The
> alert severities are this project's own categories, **not** official IMD or
> WMO warnings. No accuracy, lives-saved or economic-benefit figure has been
> measured, and none is implied anywhere in this project.

This guide takes you from a fresh clone to a working demonstration, and tells
you exactly what you will and will not see.

---

## 1. Objective

Build a system that can ingest NWP-style and historical weather data, validate
and preprocess it consistently, detect extreme weather anomalies, track those
events over time, apply graph-based and convolutional prototypes to them, and
present the result through an API and an interactive dashboard.

The honest scope: the engineering pipeline is complete and reproducible; the
**meteorological validation is not**, because no verified ground-truth warning
catalogue was available. That gap is reported throughout rather than papered
over.

## 2. Architecture overview

```text
Weather Data (synthetic fixture | ERA5 reanalysis | local NetCDF)
    ↓
Ingestion & Quality Gate          freshness, variable whitelist, retry
    ↓
Preprocessing                     validation, gap fill, derived wind fields
    ↓
Anomaly Detection                 climatology + scores + thresholds
    ↓
Event Tracking                    spatio-temporal matching, lifecycle
    ↓
Graph Construction                temporal + spatial edges, chronological split
    ↓
GNN Prototype  [PROTOTYPE]        predicts the next state of a tracked event
    ↓
AI Downscaling    [PROTOTYPE]     ~12 km → ~5 km residual CNN
    ↓
Alert Engine                      rule-based triage  [PROTOTYPE SEVERITIES]
    ↓
Realtime Orchestration            idempotent runs, stage-level status
    ↓
Evaluation & Explainability       MEASURED / UNAVAILABLE, never a fake zero
    ↓
Operations / Dashboard            health, readiness, metrics, integrity
```

Boxes marked `[PROTOTYPE]` are research components. See
[docs/sih/07_limitations.md](sih/07_limitations.md) for what each one cannot do.

## 3. Phase 0–11 progression

| Phase | Delivered |
| --- | --- |
| 0 | Synthetic NWP-style data generation, preprocessing, baseline downscaling, dataset API |
| 1 | Climatology, anomaly scores, threshold ladders, event detection |
| 2 | Event tracking: matching, trajectories, lifecycle, movement metrics |
| 3 | GNN prototype: graph dataset, model, training, inference, checkpoint contract |
| 4 | AI downscaling: dataset builder, residual CNN, training, service |
| 5 | Alerting: risk scoring, rules, lifecycle, acknowledge/resolve, notification stubs |
| 6 | Real historical data ingestion (ERA5 via Open-Meteo) with manifests and integrity |
| 7 | Leakage-free validation: chronological splits, train-only climatology, calibration |
| 8 | Near-real-time orchestration: providers, quality gate, freshness, retry, scheduler |
| 9 | Evaluation, explainability and uncertainty, each metric tagged MEASURED / UNAVAILABLE |
| 10 | Production hardening: security module, redaction, metrics, error envelope, readiness, deployment validation |
| 11 | SIH demo pipeline, system status endpoint, reproducibility verification, submission package |

## 4. Prerequisites

| Requirement | Notes |
| --- | --- |
| Python 3.11+ | Developed and validated on 3.13 |
| Node.js 20+ | Only for the dashboard |
| Docker (optional) | Only for the containerised run |
| ~2 GB free disk | PyTorch plus the scientific stack |
| Network (optional) | Only for `--provider open-meteo`; the default needs none |

No API keys are required. No secret is needed to run anything in this guide.

## 5. Setup

```bash
# From the repository root
python -m venv .venv
## 6. The demo command

One command runs the whole pipeline and prints an honest summary:

```bash
# Safe first move: runs every stage, writes nothing at all
python scripts/run_sih_demo.py --dry-run

# The real demonstration
python scripts/run_sih_demo.py

# Reuse artifacts already on disk instead of recomputing
python scripts/run_sih_demo.py --reuse-existing
```

Flags: `--dry-run`, `--clean`, `--reuse-existing`, `--verbose`, `--json`,
`--provider {local,fixture,open-meteo}`, `--file PATH`.

### Expected output

```text
========================================================================
DeepWeather AI (SIH26078) - SIH demo pipeline
========================================================================
Data source        : LOCAL FILE (provenance depends on the file)
Mode               : LIVE
Pipeline run ID    : run-3124830627d8af4f

Stages
------------------------------------------------------------------------
  [         OK] pipeline
                  run run-3124830627d8af4f: 8 stages completed.
                    - time_steps: 72
                    - events: 8
                    - alerts: 8
  [         OK] tracks
                  1 track(s) read back from tracks.json.
  [         OK] graph
                  8 nodes, 7 edges (7 temporal, 0 spatial) from 1 track(s).
  [         OK] models
                  GNN available; AI downscaling available.
  [UNAVAILABLE] evaluation
                  ... 0 metric(s) measured, 3 reported as unavailable with a reason.
  [         OK] operations
                  readiness=READY; environment=development; data_source=SYNTHETIC.
  [         OK] raw-integrity
                  7 path(s) under data/raw, unchanged by this run.
```

Exact counts depend on your data. What does **not** vary is the honesty
properties:

* `data/raw` is never modified — the script asserts this and reports it;
* a stage with no trained checkpoint says `SKIPPED` with the reason;
* a metric that could not be computed says `unavailable`, never `0`;
* the prototype disclaimer is always printed.

## 7. API endpoints used

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Liveness: is the process running? |
## 8. Frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
```

The backend must be running on port 8000 (see above). To point the dashboard
elsewhere, set `VITE_API_BASE_URL` in `frontend/.env`.

The dashboard opens on **System status**, which shows the readiness badge, the
data source and its timestamp, each component's availability, and the stored
counts. A banner at the top states the prototype limitation on every page.

`npm run build` produces a production bundle in `frontend/dist`.

## 9. Docker

```bash
docker compose config          # validate the compose file
docker compose up --build      # backend :8000, frontend :5173
docker compose down
```

The backend container runs as a non-root user and declares a liveness probe on
`/health`. The frontend waits for that probe before starting.

## 10. Validation

```bash
# Everything: tests, frontend build, compose, CLI help, API probes
python scripts/validate_deployment.py

# Just the test suite
python -m pytest -q

# Determinism of the reproducible components
python scripts/verify_reproducibility.py
## 12. What is real, what is synthetic

| Source | What it actually is |
| --- | --- |
| `fixture` provider | **Synthetic.** A smooth analytic field plus a diurnal term, generated locally. Not an observation, not a forecast. Detects 0 extreme events by design. |
| `data/raw/sample_history.nc` | **Synthetic**, with one deliberately injected extreme event so the detection, tracking and alerting stages have something to work on. |
| `open-meteo` provider | **Historical reanalysis** (ERA5) from the Open-Meteo archive. A reconstruction of the past, carrying model error. Not a live feed, not a forecast. |
| `data/raw/real/*.nc` | **Historical reanalysis**, retrieved and cached. Three days, a 3×3 grid, one region. |
| Realtime fixture | **Simulated.** Exercises the orchestration path; nothing observed is involved. |

Nothing in this system ingests a live feed. "Near-real-time" describes the
*orchestration loop*, not the data.

## 13. What is measured, what is unavailable

Measured and reported:

* event, track and alert counts, with their ids;
* graph node, edge and split sizes;
* stage durations and request counters;
* baseline comparisons on a chronological test split, where a reference exists.

Deliberately **unavailable**, reported as such and never as zero:

* forecast accuracy, precision, recall or F1 — no verified ground-truth
  warning catalogue exists for this deployment;
* any claim of lives saved, economic benefit or disaster prevention;
* meteorological validation of the GNN or downscaling models — they are trained
  on synthetic data and evaluated on reanalysis, which is not the same thing as
  being right;
* the AI downscaler **loses to cubic interpolation** on MAE on the available
  data. That is recorded in `docs/AI_DOWNSCALING.md` and is not hidden.

## 14. Limitations

The complete list is in [docs/sih/07_limitations.md](sih/07_limitations.md) and
§20 of the README. The three that matter most:

1. **No forecast skill is demonstrated.** A reanalysis describes the past; it
   is not a live feed, not real-time, and not a forecast.
2. **No authentication.** The API is unauthenticated, as it has been since
   Phase 0. It needs an authenticating reverse proxy on any untrusted network.
3. **The models are prototypes.** The GNN overfits visibly on the available
   data, and the AI downscaler does not beat the cubic baseline. Both are
   reported rather than concealed.

```

`verify_reproducibility.py` runs the pipeline twice into two scratch
directories and compares what is *promised* to be identical (event, track and
alert ids, graph structure, provenance, run ids). Timestamps and durations are
deliberately excluded, because they are meant to differ. It reports
`PASS` / `FAIL` / `NOT_APPLICABLE` per check.

Current recorded results: see [FINAL_VALIDATION.md](FINAL_VALIDATION.md).

## 11. Troubleshooting

| Symptom | What to check |
| --- | --- |
| `ECONNRESET` / "socket connection was closed unexpectedly" | The client gave up before the server finished importing. Importing the app costs 15–36 s (PyTorch). Poll `/health` until it answers; never request immediately after spawning. |
| `ConnectError` from the dashboard | The backend is not running, or `VITE_API_BASE_URL` is wrong. |
| Every stage reports `SKIPPED` for GNN/downscaling | No trained checkpoint. Run `scripts/train_gnn.py` / `scripts/train_downscaling.py`, or accept it as the documented state. |
| `/ready` says `NOT_READY` | `GET /ready` → `components` → `directories`, then read `notes`. Usually an unwritable `DATA_DIR`/`MODEL_DIR`. |
| `NOT EVALUATED` on the dashboard | No evaluation has been run. Nothing is shown in place of a score, by design. |
| Dashboard loads but every panel errors | CORS: the browser origin must appear in `CORS_ORIGINS`. |
| The demo writes nothing | You passed `--dry-run`. That is its purpose. |
| Container exits immediately | Host `data/` and `models/` are not writable by uid 1001. |

| `GET /ready` | Readiness: `READY` / `DEGRADED` / `NOT_READY` |
| `GET /api/v1/system/status` | **Start here.** Identity, data source, components, counts |
| `GET /api/v1/datasets` | NetCDF files under the data directory |
| `GET /api/v1/events` | Detected extreme events |
| `GET /api/v1/tracks` | Tracked trajectories |
| `GET /api/v1/gnn/status` | GNN model availability and provenance |
| `GET /api/v1/downscaling/status` | Downscaling model availability |
| `GET /api/v1/alerts` | Generated alerts and their lifecycle |
| `GET /api/v1/pipeline/status` | Current run, freshness, per-stage status |
| `GET /api/v1/evaluation/status` | Evaluation state |
| `GET /api/v1/operations/metrics` | System counters and histograms |
| `GET /api/v1/operations/integrity` | SHA-256 checks on checkpoints and manifests |

```bash
# Start the API
python -m uvicorn extreme_weather.api.main:app --port 8000

# Then, in another terminal
curl localhost:8000/api/v1/system/status
```

`/api/v1/system/status` is the best single request: it tells you what the
system is, what data it has, which models actually load, and how much it has
produced — with `null` wherever a quantity is unknown.

# Windows:   .venv\Scripts\activate
# POSIX:     source .venv/bin/activate

pip install -r requirements.txt

# Optional: generate the sample dataset with an injected extreme event
python scripts/generate_sample_data.py
```
