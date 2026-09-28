# DeepWeather AI - SIH26078

**Phase 11 - Final SIH demo, integration, packaging and submission readiness (prototype).**

DeepWeather AI is an AI-based extreme weather intelligence system. It analyses
large-scale Numerical Weather Prediction (NWP) and historical weather data to
detect extreme weather anomalies, track their location, movement and intensity
over time, and to provide higher-resolution (local) impact information.

## The one-command demo

```bash
pip install -r requirements.txt
python scripts/generate_sample_data.py
python scripts/run_sih_demo.py          # the complete pipeline, honestly reported
python -m uvicorn extreme_weather.api.main:app --port 8000
curl localhost:8000/api/v1/system/status
```

Full walkthrough: **[docs/SIH_DEMO_GUIDE.md](docs/SIH_DEMO_GUIDE.md)**.
Live recorded results: **[docs/FINAL_VALIDATION.md](docs/FINAL_VALIDATION.md)**.
Submission package: **[docs/sih/](docs/sih/)**.

> **This is a research prototype, not an operational disaster-warning system.**
> It demonstrates no forecast skill, is validated against no observed weather
> and no official warning catalogue, and no accuracy, lives-saved or
> economic-benefit figure has been measured. Where a quantity could not be
> computed, this system reports it as **unavailable** â€” never as zero.

## Phases delivered

| Phase | Delivered |
| --- | --- |
| 0 | Synthetic NWP-style data generation, preprocessing, baseline downscaling, dataset API, dashboard shell |
| 1 | Climatology, anomaly scores, threshold ladders, extreme-cell detection, event clustering, events API |
| 2 | Event tracking: matching, trajectories, movement metrics, track lifecycle, tracks API and panel |
| 3 | GNN prototype: graph dataset, model, training, inference, checkpoint contract |
| 4 | AI downscaling: dataset builder, residual CNN, training, service |
| 5 | Alerting: risk score, rules, lifecycle with acknowledge/resolve, API, dashboard, notification stubs |
| 6 | Real/historical ingestion: NetCDF + optional GRIB, ERA5 source, unit/coordinate normalisation, manifests, registry, API, dataset browser |
| 7 | Leakage-free validation: chronological splits, train-only climatology, threshold calibration, baseline comparisons |
| 8 | Near-real-time orchestration: providers, freshness/quality gate, retrying orchestrator, idempotency, scheduler, API, panel |
| 9 | Evaluation, explainability, uncertainty: leak-checked splits, baselines, seeded bootstrap, per-event evidence |
| 10 | Production hardening: redacting structured logs, request correlation, error envelope, readiness probe, path-traversal enforcement, SHA-256 integrity, metrics, timing, non-root container, one validation command |
| 11 | **SIH demo pipeline, `/api/v1/system/status`, reproducibility verification, dashboard system-status panel, submission package** |

Backed by both deterministic synthetic data and genuine historical
reanalysis, with a fully automated test suite.

> The Phase 3, Phase 4 and Phase 5 components are **research prototypes on
> synthetic data**. None of them is an operational forecast or warning, and none
> may be used for any emergency decision. In particular, **a finer grid is not
> automatically a more accurate one**: on the demo data the AI downscaler beats
> the nearest-neighbour baseline but **loses to cubic interpolation** on MAE
> (see docs/AI_DOWNSCALING.md), and **the Phase 5 alert severities are this
> project's own categories** - they are NOT official IMD or WMO warnings, and
> the risk score is a rule-based triage aid, not a validated disaster-risk index
> (see docs/ALERTING.md). Phase 6 ingests **real historical data** (ERA5
> reanalysis) and runs the unchanged Phase 1-5 pipeline over it, but **no skill
> score is reported because no ground-truth reference dataset exists**; see
> docs/REAL_DATA_VALIDATION.md. A historical reanalysis describes the past: it is
> not a live feed, not real-time, and not a forecast.
> Phase 8 adds a near-real-time ingestion loop, which does **not** make any of this
> operational. It has no SLA, no failover and no availability guarantee, and it
> demonstrates no forecast skill: a `FRESH` badge says how old the data is, not that
> anything about it is accurate. A missing GNN or downscaling checkpoint is reported
> as `SKIPPED` with a reason - no prediction is ever invented to fill the gap. See
> [docs/REALTIME_PIPELINE.md](docs/REALTIME_PIPELINE.md).
> Phase 9 measures the models rather than adding capability: every number it
> reports is tagged MEASURED, DERIVED, UNAVAILABLE or INSUFFICIENT_DATA, and an
> unavailable metric is never shown as zero. No model is called better than a
> baseline unless the measurement says so on that specific metric, uncertainty
> intervals are never called probabilities, and feature evidence is never
> described as causal. **No accuracy percentage is claimed**, because no verified
> ground-truth warning catalogue exists for this project. See
> [docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) and
> [docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md).
> Phase 10 hardens the deployment without changing what the system claims. **It does
> not make this an operational disaster-warning system, and there is still no
> authentication**: the API is suitable for a trusted network or a demo machine
> and requires an authenticating reverse proxy on anything else. Runtime metrics
> are *system* counters only - request volume, errors, durations - and no accuracy
> or skill figure is produced anywhere, because none has been measured. See
> [docs/PRODUCTION_READINESS.md](docs/PRODUCTION_READINESS.md),
> [docs/OBSERVABILITY.md](docs/OBSERVABILITY.md) and
> [docs/SECURITY.md](docs/SECURITY.md).

---

## 1. Project overview

| Item | Value |
| --- | --- |
| Project ID | `SIH26078` |
| Project name | DeepWeather AI |
| Current phase | Phase 10 (hardening, observability and deployment) |
| Backend | Python 3.11+, FastAPI, Pydantic, Uvicorn |
| Scientific stack | NumPy, Pandas, Xarray, Dask, NetCDF4, SciPy, MetPy |
| AI foundation | PyTorch (used by the Phase 3 GNN prototype) |
| Future GNN | **PyTorch Geometric** (FUTURE full spatio-temporal tracker, see `docs/ARCHITECTURE.md`) |
| Frontend | React, Vite, TypeScript, Leaflet, React-Leaflet, Axios |
| Deployment | Docker, Docker Compose |

## 2. SIH26078 objective

Build a system that can:

1. ingest NWP and historical weather data,
2. validate and preprocess it consistently,
3. detect extreme weather anomalies,
4. identify events and track them over time,
5. use GNN-based spatio-temporal tracking,
6. downscale coarse ~12 km fields to ~5 km local impact resolution,
7. serve the results through a FastAPI backend, an interactive React map
   dashboard and an alerting layer.

Phases 0-2 deliver the reproducible foundation (items 1, 2 and 6) plus the
interfaces (API, dashboard, registries) that the later phases plug into.

## 3. Architecture (Phase 0-11)

```text
     synthetic generator (seeded)      real / historical sources
                 |                                      |
                 v                                      v
        data/raw/synthetic/               data/raw/real/ + data/manifests/
        <dataset>.nc                           <dataset>.nc     <dataset>.json
                 |                                      |
                 +------------------+-------------------+
                                    v
      validate -> normalise units/coords -> missing -> derived -> save
                                    |
                                    v
      data/processed/<dataset>_processed.nc
                 |
                 v
        CLIMATOLOGY (reference distribution)            [Phase 1, IMPLEMENTED]
                 |
                 v
        ANOMALY SCORE (absolute | standardized | percentile
                 |      | extremeness | EFI-style prototype)
                 v
        THRESHOLDS -> extreme-cell mask                 [Phase 1, IMPLEMENTED]
                 |
                 v
        CLUSTERING -> ExtremeWeatherEvent objects        [Phase 1, IMPLEMENTED]
                 |
                 v
        EVENT MATCHING -> EventTrack trajectories       [Phase 2, IMPLEMENTED]
                 |      (distance / type / gap / intensity, one-to-one)
                 v
        distance, speed, bearing, compass, duration, lifecycle
                 |
                 v
        tracks.json
                 |
                 v
        GRAPH CONSTRUCTION                             [Phase 3, IMPLEMENTED]
                 |      nodes = event states, temporal + optional spatial edges
                 v
        GNN EVOLUTION PREDICTION                       [Phase 3, PROTOTYPE]
                 |      delta lat/lon/intensity -> next state
                 v
        models/gnn/best.json  ->  GNN API               [Phase 3, PROTOTYPE]
                 |
                 v
        GRAPH CONSTRUCTION                              [Phase 4, IMPLEMENTED]
                 |      synthetic ~5 km reference, chronological split
                 v
        AI DOWNSCALING (residual CNN)                    [Phase 4, PROTOTYPE]
                 |      bilinear baseline + learned non-negative residual
                 v
        models/downscaling/best.pt -> Downscaling API    [Phase 4, PROTOTYPE]
                 |
                 v
        ALERT ENGINE                                    [Phase 5, PROTOTYPE]
                 |      risk score -> rules -> NEW/ACTIVE/ACK/RESOLVED
                 v
        NEAR-REAL-TIME ORCHESTRATION                     [Phase 8, IMPLEMENTED]
                 |      providers, quality gate, freshness, retry, idempotency
                 v
        EVALUATION & EXPLAINABILITY                     [Phase 9, IMPLEMENTED]
                 |      MEASURED / DERIVED / UNAVAILABLE / INSUFFICIENT_DATA
                 v
        OPERATIONS & DASHBOARD                          [Phase 10-11, IMPLEMENTED]
                 |      /health /ready /api/v1/system/status, metrics, integrity
                 v
        React dashboard + FastAPI backend
```

**Prototype markers are load-bearing.** GNN, AI downscaling and the alert
severities are prototypes: they are not validated against observed weather, and
the severities are this project's own categories, not official IMD or WMO
warnings. The GNN overfits visibly on the available data and the AI downscaler
loses to cubic interpolation on MAE; both are recorded in
[`docs/AI_DOWNSCALING.md`](docs/AI_DOWNSCALING.md) rather than hidden.

Full detail, including data contracts and the roadmap, is in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); the SIH submission package is in
[`docs/sih/`](docs/sih/); the anomaly methodology is in
[`docs/ANOMALY_DETECTION.md`](docs/ANOMALY_DETECTION.md), the tracking design in
[`docs/EVENT_TRACKING.md`](docs/EVENT_TRACKING.md) and the GNN design in
[`docs/GNN.md`](docs/GNN.md), the ingestion design in
[`docs/DATA_INGESTION.md`](docs/DATA_INGESTION.md), the checks in
[`docs/DATA_QUALITY.md`](docs/DATA_QUALITY.md), and the real-data evaluation
methodology and limitations in
[`docs/REAL_DATA_VALIDATION.md`](docs/REAL_DATA_VALIDATION.md).

## 4. Folder structure

```text
.
â”œâ”€â”€ src/extreme_weather/
â”‚   â”œâ”€â”€ config.py                     # environment-driven settings (pydantic-settings)
â”‚   â”œâ”€â”€ data/
â”‚   â”‚   â”œâ”€â”€ generator.py              # deterministic synthetic weather data + manifest
â”‚   â”‚   â”œâ”€â”€ loader.py                 # NetCDF load/save, safe path resolution
â”‚   â”‚   â”œâ”€â”€ validator.py              # data contract (dims, variables, coordinates)
â”‚   â”‚   â”œâ”€â”€ missing.py                # max_gap gap filling
â”‚   â”‚   â”œâ”€â”€ derived.py                # wind_speed / wind_direction
â”‚   â”‚   â””â”€â”€ preprocessing.py          # load -> validate -> missing -> derived -> save
â”‚   â”œâ”€â”€ anomaly/                      # Phase 1
â”‚   â”‚   â”œâ”€â”€ climatology.py            # reference distribution (month / day_of_year)
â”‚   â”‚   â”œâ”€â”€ scores.py                 # absolute | standardized | percentile |
â”‚   â”‚   â”‚                             # extremeness | EFI-style prototype
â”‚   â”‚   â”œâ”€â”€ thresholds.py             # configurable severity ladders (per variable)
â”‚   â”‚   â”œâ”€â”€ detector.py               # score + thresholds -> extreme-cell mask
â”‚   â”‚   â”œâ”€â”€ clustering.py             # connected cells -> ExtremeWeatherEvent
â”‚   â”‚   â”œâ”€â”€ models.py                 # AnomalyAnalysisResult
â”‚   â”‚   â”œâ”€â”€ service.py                # orchestration + file-backed EventStore
â”‚   â”‚   â””â”€â”€ visualization.py          # optional 4-panel PNG
â”‚   â”œâ”€â”€ downscaling/                  # Phase 4
â”‚   â”‚   â”œâ”€â”€ reference.py              # synthetic high-resolution target (area-conserving)
â”‚   â”‚   â”œâ”€â”€ features.py               # 10 documented input channels
â”‚   â”‚   â”œâ”€â”€ dataset.py                # chronological split + training-only normalisation
â”‚   â”‚   â”œâ”€â”€ model.py                  # residual CNN (projection -> 3 blocks -> ReLU residual)
â”‚   â”‚   â”œâ”€â”€ metrics.py                # MAE/RMSE/bias/correlation + POD/FAR/CSI
â”‚   â”‚   â”œâ”€â”€ inference.py              # AI + baseline paths, extreme-event metrics
â”‚   â”‚   â”œâ”€â”€ api_models.py             # status/model/run/evaluate schemas
â”‚   â”‚   â”œâ”€â”€ training.py               # full-batch Adam, best-on-validation
â”‚   â”‚   â”œâ”€â”€ checkpoint.py             # models/downscaling/best.pt + integrity checks
â”‚   â”‚   â””â”€â”€ service.py                # run + baseline-vs-AI comparison
â”‚   â”œâ”€â”€ gnn/                         # Phase 3
â”‚   â”‚   â”œâ”€â”€ features.py                # 16 node + 6 edge features, delta targets
â”‚   â”‚   â”œâ”€â”€ graph.py                  # temporal + optional spatial edges
â”‚   â”‚   â”œâ”€â”€ dataset.py                # chronological train/val/test split
â”‚   â”‚   â”œâ”€â”€ model.py                  # mean-aggregation message-passing GNN
â”‚   â”‚   â”œâ”€â”€ training.py               # full-batch Adam, best-on-validation
â”‚   â”‚   â”œâ”€â”€ checkpoint.py             # models/gnn/best.json + integrity checks
â”‚   â”‚   â”œâ”€â”€ models.py                  # GNNModelStatus / GNNModelInfo / GNNPrediction
â”‚   â”‚   â””â”€â”€ service.py                # checkpoint + track -> prediction
â”‚   â”œâ”€â”€ tracking/                     # Phase 2
â”‚   â”‚   â”œâ”€â”€ metrics.py                # distance, speed, bearing, compass, trend
â”‚   â”‚   â”œâ”€â”€ matching.py               # deterministic one-to-one event matching
â”‚   â”‚   â”œâ”€â”€ tracker.py                # track building + NEW/ACTIVE/ENDED lifecycle
â”‚   â”‚   â”œâ”€â”€ models.py                 # TrackResult
â”‚   â”‚   â”œâ”€â”€ service.py                # TrackingService orchestration
â”‚   â”‚   â””â”€â”€ store.py                  # file-backed TrackStore (tracks.json)
â”‚   â”œâ”€â”€ downscaling/
â”‚   â”‚   â”œâ”€â”€ baseline.py               # nearest / cubic interpolation
â”‚   â”‚   â””â”€â”€ registry.py               # method registry + downscale() factory
    â”œâ”€â”€ ingestion/                    # Phase 6: real / historical data
    â”‚   â”œâ”€â”€ base.py                 # WeatherDataSource protocol, DataOrigin
    â”‚   â”œâ”€â”€ netcdf.py               # NetCDF reader (lazy + chunked loading)
    â”‚   â”œâ”€â”€ grib.py                 # GRIB reader (optional backend, detected)
    â”‚   â”œâ”€â”€ era5.py                 # ERA5 reanalysis via the Open-Meteo archive
    â”‚   â”œâ”€â”€ nwp.py                  # NWP model output (local or configured endpoint)
    â”‚   â”œâ”€â”€ normalizer.py           # unit + coordinate normalisation, recorded
    â”‚   â”œâ”€â”€ quality.py              # DataQualityReport (errors block, warnings do not)
    â”‚   â”œâ”€â”€ manifest.py             # DatasetManifest, ProvenanceStep, hashes
    â”‚   â”œâ”€â”€ registry.py             # DatasetRegistry (synthetic / real kept apart)
    â”‚   â”œâ”€â”€ service.py              # the ingest chain used by the CLI and the API
    â”‚   â””â”€â”€ exceptions.py            # one specific error per failure mode
â”‚   â”œâ”€â”€ api/
â”‚   â”‚   â”œâ”€â”€ main.py                   # FastAPI app (uvicorn extreme_weather.api.main:app)
â”‚   â”‚   â””â”€â”€ routers/                  # health, datasets, events, tracks, gnn, downscaling
â”‚   â””â”€â”€ schemas/
â”‚       â”œâ”€â”€ dataset.py                # Pydantic dataset responses
â”‚       â”œâ”€â”€ event.py                  # ExtremeWeatherEvent + event responses
â”‚       â”œâ”€â”€ track.py                  # EventTrack / TrackPoint / trajectory responses
â”‚       â””â”€â”€ (GNN responses live in extreme_weather/gnn/models.py)
â”œâ”€â”€ tests/                            # pytest suite (1105 tests)
â”‚   â”œâ”€â”€ conftest.py
â”‚   â”œâ”€â”€ test_data.py
â”‚   â”œâ”€â”€ test_preprocessing.py
â”‚   â”œâ”€â”€ test_downscaling.py
â”‚   â”œâ”€â”€ test_api.py
â”‚   â”œâ”€â”€ test_climatology.py
â”‚   â”œâ”€â”€ test_anomaly.py
â”‚   â”œâ”€â”€ test_event_clustering.py
â”‚   â”œâ”€â”€ test_event_api.py
â”‚   â”œâ”€â”€ test_tracking_metrics.py
â”‚   â”œâ”€â”€ test_tracking_matching.py
â”‚   â”œâ”€â”€ test_tracking_service.py
â”‚   â”œâ”€â”€ test_tracking_api.py
â”‚   â”œâ”€â”€ test_gnn_features.py
â”‚   â”œâ”€â”€ test_gnn_graph.py
â”‚   â”œâ”€â”€ test_gnn_dataset.py
â”‚   â”œâ”€â”€ test_gnn_model.py
â”‚   â”œâ”€â”€ test_gnn_training.py
â”‚   â”œâ”€â”€ test_gnn_inference.py
â”‚   â””â”€â”€ test_gnn_api.py
   â”‚ â”œâ”€â”€ test_alert_models.py
   â”‚ â”œâ”€â”€ test_alert_severity.py
   â”‚ â”œâ”€â”€ test_alert_rules.py
   â”‚ â”œâ”€â”€ test_alert_risk.py
   â”‚ â”œâ”€â”€ test_alert_store.py
   â”‚ â”œâ”€â”€ test_alert_notifications.py
   â”‚ â”œâ”€â”€ test_alert_service.py
   â”‚ â”œâ”€â”€ test_alert_api.py
   â”‚ â”œâ”€â”€ test_alert_simulation.py
   â”‚ â”œâ”€â”€ test_ingestion_base.py
   â”‚ â”œâ”€â”€ test_ingestion_netcdf.py
   â”‚ â”œâ”€â”€ test_ingestion_normalizer.py
   â”‚ â”œâ”€â”€ test_ingestion_quality.py
   â”‚ â”œâ”€â”€ test_ingestion_manifest.py
   â”‚ â”œâ”€â”€ test_ingestion_registry.py
   â”‚ â”œâ”€â”€ test_ingestion_service.py
   â”‚ â”œâ”€â”€ test_ingestion_api.py
   â”‚ â”€â”€ test_real_data_pipeline.py
â”œâ”€â”€ scripts/
â”‚   â”œâ”€â”€ generate_sample_data.py
â”‚   â”œâ”€â”€ preprocess_data.py
â”‚   â”œâ”€â”€ detect_anomalies.py
â”‚   â”œâ”€â”€ track_events.py
â”‚   â”œâ”€â”€ build_graph_dataset.py
â”‚   â”œâ”€â”€ train_gnn.py
â”‚   â”œâ”€â”€ run_gnn_inference.py
â”‚   â”œâ”€â”€ build_downscaling_dataset.py
â”‚   â””â”€â”€ train_downscaling.py
â”‚   â”œâ”€â”€ generate_alerts.py
â”‚   â”œâ”€â”€ simulate_alerts.py
â”‚   â”œâ”€â”€ ingest_weather.py
â”‚   â””â”€â”€ evaluate_real_data.py
â”œâ”€â”€ data/{raw,processed,samples}/     # generated data lives here (git-ignored)
â”‚   â”œâ”€â”€ raw/synthetic/            # SYNTHETIC datasets (never mixed with real)
â”‚   â”œâ”€â”€ raw/real/                 # REAL / historical datasets
â”‚   â”œâ”€â”€ manifests/               # one provenance JSON per ingested dataset
â”‚   â”œâ”€â”€ reference/               # optional ground-truth / event reference data
â”‚   â””â”€â”€ cache/                  # remote downloads (bounded, size-limited)
â”œâ”€â”€ models/                           # future model artefacts
â”œâ”€â”€ frontend/                         # React + Vite + TypeScript dashboard
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ ARCHITECTURE.md
â”‚   â”œâ”€â”€ ANOMALY_DETECTION.md
â”‚   â””â”€â”€ EVENT_TRACKING.md
â”œâ”€â”€ docker-compose.yml
â”œâ”€â”€ Dockerfile
â”œâ”€â”€ pyproject.toml
â””â”€â”€ requirements.txt
```

## 5. Setup (quick start)

```bash
# 1. Python environment (recommended)
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
pip install -e .            # makes `extreme_weather` importable (uvicorn needs this)

# 3. Generate the synthetic dataset
python scripts/generate_sample_data.py

# 4. Run the preprocessing pipeline
python scripts/preprocess_data.py

# 5. Run the tests
pytest -q

# 6. Start the backend
uvicorn extreme_weather.api.main:app --reload

# 7. In a second terminal, start the frontend
cd frontend
npm install
npm run dev
```

> `scripts/*.py` and `pytest` run straight from a clone (the scripts add `src/`
> to `sys.path`, and `pyproject.toml` sets pytest's `pythonpath = ["src"]`).
> `uvicorn extreme_weather.api.main:app` needs the package importable, so run
> `pip install -e .` once - or start it with `uvicorn extreme_weather.api.main:app
> --app-dir src` if you prefer not to install anything.

Optional: copy `.env.example` to `.env` and adjust it (paths, CORS origins,
generator defaults). No API keys are required anywhere - all data is synthetic.

## 6. Python environment

- Required: **Python 3.11 or newer** (developed and validated on 3.13).
- A virtual environment (venv or conda) is strongly recommended.
- The package uses a `src/` layout; `pyproject.toml` declares it, so
  `pip install -e .` makes `extreme_weather` importable everywhere.
  Without installation, the scripts and `pyproject.toml`'s pytest
  `pythonpath = ["src"]` make the repository runnable as-is.

## 7. Dependency installation

```bash
pip install -r requirements.txt      # runtime + test dependencies
pip install -e ".[dev]"             # editable install with dev extras
pip install -e ".[gnn]"             # FUTURE: torch-geometric for the GNN tracker
```

`requirements.txt` is intentionally unpinned: every package ships wheels for
CPython 3.11-3.13 on Windows, Linux and macOS, and there is no compatibility
conflict to work around yet. `bottleneck` is deliberately **not** required -
gap filling is implemented explicitly in `extreme_weather.data.missing`.

## 8. Generate sample data

```bash
python scripts/generate_sample_data.py
# -> data/raw/sample_history.nc
# -> data/raw/sample_history.manifest.json
```

Useful options:

```bash
python scripts/generate_sample_data.py --seed 7 --time-steps 48
python scripts/generate_sample_data.py --resolution 0.125
python scripts/generate_sample_data.py --latitude-bounds 8 20 --longitude-bounds 70 82
python scripts/generate_sample_data.py --no-event
python scripts/generate_sample_data.py --output data/samples/quiet.nc
```

The generator is deterministic: the same seed and configuration always produce
the same fields. The NetCDF global attributes and the JSON manifest record the
generator version, seed, variables, dimensions, grid resolution and the injected
event configuration. **The data is clearly labelled synthetic - it is not
observed weather data and not an operational NWP product.**

## 9. Run preprocessing

```bash
python scripts/preprocess_data.py
# -> data/processed/sample_history_processed.nc
```
```bash
python scripts/preprocess_data.py --max-gap 2                 # stricter gap filling
python scripts/preprocess_data.py --max-gap -1                # fill every interior gap
python scripts/preprocess_data.py --no-wind-direction
python scripts/preprocess_data.py --input data/raw/sample_history.nc --output /tmp/out.nc
```

Pipeline: `load -> validate -> missing values (max_gap) -> derived variables -> save`.
The raw input is never modified, output directories are created automatically,
and a JSON report is stored in the output file's global attributes.

### Missing-value policy (`max_gap`)

- Only **interior** gaps are filled (a gap bracketed by valid values on both sides).
- A run of `L` missing steps is filled only when `L <= max_gap`; longer runs stay missing.
- Leading/trailing gaps are preserved (no extrapolation at the series edges).
- `max_gap=0` fills nothing, `max_gap=None` fills every interior gap.
- All-missing arrays are returned unchanged and never crash the pipeline.

## 10. Detect extreme weather events (Phase 1)

```bash
python scripts/detect_anomalies.py
# -> data/processed/events.json
# -> data/processed/anomaly.nc
```

```bash
python scripts/detect_anomalies.py --reference data/raw/sample_history.nc  # explicit climatology
python scripts/detect_anomalies.py --score efi                            # EFI-style score
python scripts/detect_anomalies.py --variables precipitation temperature
python scripts/detect_anomalies.py --min-cells 3                         # drop noise clusters
python scripts/detect_anomalies.py --figure data/processed/anomaly_overview.png
```

Pipeline: `climatology -> anomaly score -> thresholds -> extreme cells ->
clustering -> ExtremeWeatherEvent`. The CLI prints every detected event and
writes machine-readable output; raw data is never modified.

| Step | Module | Note |
| --- | --- | --- |
| reference distribution | `anomaly.climatology` | monthly / day-of-year / pooled grouping |
| anomaly score | `anomaly.scores` | `absolute`, `standardized`, `percentile`, `extremeness`, `efi` (prototype) |
| severity thresholds | `anomaly.thresholds` | configurable ladders, per variable, prototype values |
| grid-cell detection | `anomaly.detector` | score + ladder -> boolean mask (NaN never extreme) |
| event clustering | `anomaly.clustering` | connected cells -> events with centroid and radius |
| orchestration | `anomaly.service` | `AnomalyDetectionService` + file-backed `EventStore` |

Full methodology, the exact EFI-style definition, the threshold tables and the
limitations are documented in
[`docs/ANOMALY_DETECTION.md`](docs/ANOMALY_DETECTION.md).

> The anomaly pipeline runs on synthetic data with **prototype** thresholds. It
> is not an operational forecast or an official warning system.

## 11. Track events (Phase 2)

```bash
python scripts/track_events.py
# -> data/processed/tracks.json
```

```bash
python scripts/track_events.py --max-distance-km 150 --max-speed-kmh 80
python scripts/track_events.py --max-time-gap-hours 3 --intensity-weight 0.5
python scripts/track_events.py --no-write          # dry run, nothing stored
```

Pipeline: `events.json -> TrackingService -> tracks.json`. Detections of the same
event are linked across consecutive time steps and reported as `EventTrack`
objects with trajectory, distance, speed, bearing, compass direction, duration,
peak intensity and lifecycle status.

| Step | Module | Note |
| --- | --- | --- |
| movement metrics | `tracking.metrics` | haversine distance, speed, bearing, 16-point compass, intensity trend, `CALM` for stationary |
| event matching | `tracking.matching` | deterministic one-to-one assignment gated by `MatchingConfig` |
| track building | `tracking.tracker` | lifecycle `NEW` / `ACTIVE` / `ENDED`, ids `TRK-<TYPE>-<TIME>-<NNN>` |
| orchestration | `tracking.service` | `TrackingService`: load, sort, group, match, build, store |
| persistence | `tracking.store` | `TrackStore` (`save`/`load`/`get`/`list`/`clear`) |

Design decisions, the matching cost formula and the limitations are documented
in [`docs/EVENT_TRACKING.md`](docs/EVENT_TRACKING.md).

> Tracking links **synthetic** detections with a fixed, explainable cost
> function. It is not a forecast, and the GNN tracker remains FUTURE.

## 12. Build, train and use the GNN prototype (Phase 3)

```bash
python scripts/build_graph_dataset.py    # tracks.json -> graph + chronological split
python scripts/train_gnn.py              # -> models/gnn/best.json
python scripts/run_gnn_inference.py --track-id TRK-EXTREME_RAINFALL-20240602T0200-001
```

```bash
python scripts/build_graph_dataset.py --no-spatial-edges     # temporal graph only
python scripts/build_graph_dataset.py --max-spatial-distance-km 80
python scripts/train_gnn.py --epochs 500 --learning-rate 0.02 --seed 7
python scripts/train_gnn.py --verbose                       # per-epoch losses
python scripts/run_gnn_inference.py --list-tracks
```

| Step | Module | Note |
| --- | --- | --- |
| features | `gnn.features` | 16 node features, 6 edge features, delta targets, documented units |
| graph | `gnn.graph` | nodes = event states; temporal edges + optional same-timestamp spatial edges |
| split | `gnn.dataset` | **chronological** 60/20/20 - never shuffled |
| model | `gnn.model` | input projection -> 2 x mean-aggregation message passing -> MLP head |
| training | `gnn.training` | full-batch Adam, best-on-validation, seeded |
| checkpoint | `gnn.checkpoint` | `models/gnn/best.json` with integrity validation |
| service | `gnn.service` | checkpoint + track -> validated prediction |

> **The model predicts changes, not positions**: it regresses
> `[?latitude, ?longitude, ?intensity]`, which are added to the last observed
> state. On the demo data (one track, 4 training samples) the prediction is
> physically implausible and the losses show a clear overfit. This is a
> **research prototype on synthetic data** - not a forecast, no skill score, no
> accuracy figure. See [`docs/GNN.md`](docs/GNN.md).

## 13. AI downscaling (Phase 4)

```bash
python scripts/build_downscaling_dataset.py   # coarse + synthetic reference -> features/targets
python scripts/train_downscaling.py           # -> models/downscaling/best.pt
```

```bash
python scripts/build_downscaling_dataset.py --target-resolution 0.05 --no-write
python scripts/train_downscaling.py --epochs 40 --learning-rate 0.001 --verbose
```

| Step | Module | Note |
| --- | --- | --- |
| reference | `downscaling.reference` | synthetic ~5 km target; **area-average conserving** |
| features | `downscaling.features` | 10 documented channels, coarse field + coordinates only |
| split | `downscaling.dataset` | **chronological** 60/20/20, normalisation from training only |
| model | `downscaling.model` | residual CNN, ReLU-constrained output |
| metrics | `downscaling.metrics` | MAE / RMSE / bias / correlation / POD / FAR / CSI |
| comparison | `downscaling.service` | nearest vs cubic vs ai on the same test frames |

The Phase 0 `nearest` and `cubic` baselines are **preserved**; the learned method
joins them in the same registry, so `downscale(field, 0.05, method="ai")` works
alongside `method="nearest"`.

> **The AI does not beat cubic interpolation on this data.** Measured on the same
> test frames: MAE 0.12598 (ai) vs 0.11826 (cubic) vs 0.14363 (nearest). The AI
> has the best correlation, the best event-peak and event-centre preservation,
> but over-predicts (bias +0.060) and raises FAR. A smooth interpolant is a
> hard baseline to beat on 43 training frames. Reported as measured - see
> `docs/AI_DOWNSCALING.md`.

## 14. Generate alerts and demonstrate the lifecycle (Phase 5)

`ash
python scripts/generate_alerts.py            # events + tracks (+ optional GNN) -> alerts.json
python scripts/generate_alerts.py --dry-run  # print what would happen, write NOTHING
python scripts/generate_alerts.py --use-gnn  # also enrich with the Phase 3 prediction
python scripts/simulate_alerts.py            # deterministic lifecycle demo (synthetic)
`

generate_alerts.py writes data/processed/alerts.json and
data/processed/notifications.json. The severities it produces are **project
prototype categories, not official IMD or WMO warnings**, and the risk score is
a rule-based triage aid - see [docs/ALERTING.md](docs/ALERTING.md).

## 15. Ingest real / historical data (Phase 6)

The pipeline above runs on synthetic data. Phase 6 lets it run on **real**
weather files as well, without changing Phases 0-5.

```bash
# Inspect a file first. --validate writes nothing at all.
python scripts/ingest_weather.py --file data/raw/real/my_data.nc --validate

# Ingest a local NetCDF/GRIB file: validate -> normalise -> manifest -> register.
python scripts/ingest_weather.py --file data/raw/real/my_data.nc

# ...and additionally run the Phase 0 preprocessing pipeline.
python scripts/ingest_weather.py --file data/raw/real/my_data.nc --process

# Fetch a small ERA5 subset (real past reanalysis, no API key required).
python scripts/ingest_weather.py --fetch-era5 \
    --era5-lats 12.5,13.0,13.5 --era5-lons 76.5,77.0,77.5 \
    --start 2024-06-01 --end 2024-06-03 --dataset-id era5_real --process
```

Then continue with the ordinary Phase 1-5 commands on the processed file:

```bash
python scripts/detect_anomalies.py --input data/processed/era5_real_processed.nc \
    --events-output data/processed/real_events.json
python scripts/track_events.py --events data/processed/real_events.json \
    --tracks-output data/processed/real_tracks.json
python scripts/generate_alerts.py --events data/processed/real_events.json \
    --tracks data/processed/real_tracks.json \
    --alerts-output data/processed/real_alerts.json

# Reproducible evaluation report (JSON + Markdown).
python scripts/evaluate_real_data.py --real-id era5_real --markdown
```

Three things this preserves:

* **Synthetic and real data never mix.** They are written to
  `data/raw/synthetic/` and `data/raw/real/` and carry an `origin` of
  `SYNTHETIC`, `HISTORICAL` or `EXTERNAL`. There is no default origin.
* **Provenance is never invented.** A local file is recorded as
  `source="local_file"` with **no URL**; the manifest constructor refuses that
  combination outright.
* **Nothing is fabricated.** A missing file, unreachable provider, missing GRIB
  engine or absent credential each raise a specific, actionable error instead of
  producing an empty dataset that would look like "no events found".

> **Historical is not live.** ERA5 is a *reanalysis of the past* - real
> measurements fused into a model. It is not a live feed, not real-time, and not
> a forecast. Nothing in this project ingests a live feed.

See [docs/DATA_INGESTION.md](docs/DATA_INGESTION.md) and
[docs/DATA_QUALITY.md](docs/DATA_QUALITY.md).

## 16. Validate on real data (Phase 7)

Phase 6 showed the pipeline *can* read real data. Phase 7 measures what it
*produces* from it, under a method that cannot leak the test period into
training.

```bash
# Fetch a bounded ERA5 subset and run every stage end to end.
python scripts/run_phase7_validation.py --days 14 --end-date 2024-06-30

# -> data/validation/reports/phase7_summary.{json,md}
# -> data/validation/provenance/<run_id>.json

# Reproduce without re-downloading.
python scripts/run_phase7_validation.py --days 14 --end-date 2024-06-30 --reuse-existing
```

The pipeline splits the record **strictly chronologically** (train / validation
/ test, never shuffled), builds the climatology and calibrates the thresholds
from the **training slice only**, scores the test split **once**, and compares
every model against a real baseline.

What it will **not** print, and why:

| Metric | Status |
| ------ | ------ |
| POD, FAR, CSI, precision, recall, accuracy | **Not computed** - no ground-truth catalogue ships with this project |
| Day-of-year climatology | **Insufficient** - 14 days is not a climatology |
| GNN score | **Not computed** on this record - too few test nodes |
| AI downscaling score | **Not computed** - no trained checkpoint |

Every uncomputable metric is reported as `null` with an explicit reason, never as
`0.0`: a zero reads as "no skill", which is a different and false claim. Where
the GNN or the downscaler *can* be evaluated, the baseline comparison is
mandatory and a loss is reported as a loss - see
[docs/REAL_DATA_VALIDATION.md](docs/REAL_DATA_VALIDATION.md) for the full
methodology, results and limitations, and
[docs/PHASE7_BASELINE.md](docs/PHASE7_BASELINE.md) for the pre-Phase-7
baseline.

## 17. Run the near-real-time pipeline (Phase 8)

Phase 7 measured what the pipeline produces. Phase 8 lets it run repeatedly on
new data, with the data's age and structure checked before anything is
processed.

```bash
# See what would happen. Writes nothing.
python scripts/run_realtime_pipeline.py --dry-run

# One run against the deterministic fixture provider (no network, no API key).
python scripts/run_realtime_pipeline.py --once

# Poll ERA5 reanalysis every hour.
python scripts/run_realtime_pipeline.py --interval-minutes 60 --provider open-meteo

# Ingest a file already on disk.
python scripts/run_realtime_pipeline.py --once --provider local --file data/raw/x.nc
```

```text
provider -> ingestion -> quality gate -> preprocess -> anomaly -> track
         -> gnn (optional) -> downscale (optional) -> alert
```

Three properties are worth knowing before relying on it:

| Property | Behaviour |
| --- | --- |
| Stale data is never silently treated as current | Every run carries a `FRESH` / `STALE` / `EXPIRED` / `UNKNOWN` verdict. An unknown timestamp is **not** treated as fresh. |
| A rejected dataset stops the run | A `REJECTED` quality verdict ends the run **before** anomaly detection, because the downstream stages would otherwise return a confident, meaningless result. |
| A missing model is skipped, not faked | No GNN or downscaling checkpoint means `SKIPPED` with the reason. No prediction is ever invented. |

Re-running the same ingestion is a no-op: the `run_id` is a hash of the
provider, window, newest timestamp and data checksum, so a scheduled job that
fires twice does not double the event catalogue. Use `--force` to reprocess.

```bash
curl localhost:8000/api/v1/pipeline/status
curl -X POST 'localhost:8000/api/v1/pipeline/run?dry_run=true' \
  -H 'Content-Type: application/json' \
  -d '{"provider": "fixture", "hours": 6}'
```

Full design, failure policy, incremental-processing rules and limitations:
[docs/REALTIME_PIPELINE.md](docs/REALTIME_PIPELINE.md).

## 18. Evaluate the models (Phase 9)

Phase 8 made the pipeline runnable on schedule. Phase 9 measures what the
models in it actually do.

```bash
# See what would be evaluated. Writes nothing.
python scripts/run_phase9_evaluation.py --dry-run

# Self-test the evaluator on deterministic synthetic samples.
python scripts/run_phase9_evaluation.py --model fixture --sample-count 60

# Evaluate a real ERA5 record.
python scripts/run_phase9_evaluation.py --dataset data/validation/processed/phase7_era5_processed.nc
```

```bash
curl localhost:8000/api/v1/evaluation/status
curl localhost:8000/api/v1/evaluation/gnn
curl localhost:8000/api/v1/evaluation/uncertainty
curl localhost:8000/api/v1/evaluation/explain/<event_id>
```

| Question | Answer | Status vocabulary |
| --- | --- | --- |
| GNN vs baselines? | Persistence **and** constant velocity, per metric | `MEASURED` / `UNAVAILABLE` |
| Downscaler vs interpolation? | Nearest and cubic on identical test frames | `MEASURED` / `UNAVAILABLE` |
| What evidence supports an event? | Recorded values, typed and sourced | `supporting` / `context` / `unavailable` |
| How uncertain is a prediction? | Validation-residual bootstrap, with measured coverage | `DERIVED` / `INSUFFICIENT_DATA` |
| Is it reproducible? | Seeded; identical inputs give identical records | verified by test |

Three things this phase will **not** do, by design:

* report an unavailable metric as `0.0` - a zero reads as a measurement;
* call an interval a probability - it is an empirical error distribution whose
  calibration is *measured*, and reported as uncalibrated when it is not;
* describe feature evidence as causal - it is input sensitivity of a fitted
  function, which is a different claim entirely.

Full methodology, failure modes and limitations:
[docs/MODEL_EVALUATION.md](docs/MODEL_EVALUATION.md) and
[docs/EXPLAINABILITY.md](docs/EXPLAINABILITY.md).

## 19. Validate the deployment (Phase 10)

One command runs every check that does not need the network.

```bash
python scripts/validate_deployment.py            # everything
python scripts/validate_deployment.py --offline  # without the network test
python scripts/validate_deployment.py --fast     # config, API, CLI, compose
python scripts/validate_deployment.py --json     # machine-readable
```

It checks configuration, the pytest suite, the frontend build, `docker compose
config`, every CLI's `--help`, the API import, and the same 15 endpoints
probed **twice**: in process over ASGI, and over a real socket against a
uvicorn subprocess started by the validator.

The live probe is what a deployment actually does, so it is started
deliberately: the port is assigned by the OS, stdout and stderr are captured to
a log, `/health` is polled until it answers before any other request is made,
and the process is always terminated. A start-up failure is reported **with
the server log** rather than as a bare timeout.

> A start-up race, not a backend fault, produced an `ECONNRESET` here once.
> Importing the app costs ~15-36 s (PyTorch dominates), so a client that did
> not wait for readiness saw a refused socket and reported it as a crash.
> Readiness is awaited, never assumed.

```bash
curl localhost:8000/health    # liveness: is the process running
curl localhost:8000/ready     # readiness: READY | DEGRADED | NOT_READY
curl localhost:8000/api/v1/operations/metrics
curl localhost:8000/api/v1/operations/integrity
```

| Control | State |
| --- | --- |
| Path traversal, extension and boundary enforcement | Implemented |
| Credential redaction in logs, manifests and responses | Implemented |
| SHA-256 integrity on checkpoints and manifests | Implemented |
| Non-root container, health and readiness gating | Implemented |
| **Authentication / authorisation** | **Not implemented** |

A missing model yields `DEGRADED`, never `READY`. No model ships trained, so
`DEGRADED` is the expected reading for a fresh install.

## 20. Run tests

```bash
pytest -q
```

```bash
pytest -q tests/test_data.py           # single module
pytest -q -k missing                   # single topic
pytest --cov=src -q                    # with coverage (pip install pytest-cov)
```

The suite covers synthetic generation and determinism, dimensions and variables,
manifest provenance, the injected extreme event, dataset validation, derived wind
variables, the `max_gap` gap semantics, the preprocessing pipeline and its NetCDF
output, downscaling, the FastAPI endpoints (Phase 0), the complete anomaly
detection stack (Phase 1) and the complete tracking stack - metrics, matching,
lifecycle, service, store and API (Phase 2) and the complete GNN stack -
features, graph, chronological split, model, message passing, loss, training,
checkpoint integrity, inference and API (Phase 3), including tests that the
split cannot leak the future into training, plus the complete downscaling stack -
resolution detection, the area-conserving reference, features, the
training-only normalisation, the chronological split, the model, the metrics,
checkpoint integrity, baseline-vs-AI comparison, extreme-event metrics and the
API (Phase 4). All data used in the tests is synthetic; no network is required.

## 21. SIH demo, reproducibility and submission (Phase 11)

```bash
# The complete pipeline, honestly reported
python scripts/run_sih_demo.py --dry-run     # every stage, writes nothing
python scripts/run_sih_demo.py               # the real run
python scripts/run_sih_demo.py --reuse-existing

# Determinism of the reproducible components
python scripts/verify_reproducibility.py

# Regenerate docs/FINAL_VALIDATION.md from actual results
python scripts/generate_final_validation.py
```

| Deliverable | What it does |
| --- | --- |
| `scripts/run_sih_demo.py` | One command, the whole pipeline, with `--dry-run` / `--clean` / `--reuse-existing` / `--verbose` / `--json`. Reuses the Phase 0â€“10 services rather than reimplementing them, never writes to `data/raw`, and prints `unavailable` where nothing was measured. |
| `GET /api/v1/system/status` | Identity, phase, environment, readiness, data source and timestamp, per-component availability **with reasons**, stored counts, and the prototype disclaimer. Unavailable is `null`, never `0`. |
| `scripts/verify_reproducibility.py` | Runs the pipeline twice into separate scratch directories and compares event ids, track ids, alert ids, graph structure, provenance and run ids. Reports `PASS` / `FAIL` / `NOT_APPLICABLE`. Excludes only fields that are genuinely volatile. |
| `scripts/generate_final_validation.py` | Regenerates the validation record from real command output, so no number in it is typed by hand. |
| `docs/SIH_DEMO_GUIDE.md` | From clone to demonstration, with expected output and troubleshooting. |
| `docs/sih/` | The submission package: problem, solution, architecture, stack, algorithms, results, limitations, demo, API. |

**Three real defects were found and fixed while building this**, all now
covered by regression tests. None was found by inspection:

1. The orchestrator accepted a `settings` argument but built its event, track
   and alert stores through services that defaulted to the process-wide
   `get_settings()`. A run pointed at a scratch directory wrote into the
   *repository's* data directory.
2. `TrackingService.run()` defaults to `write_output=False`, and the
   orchestrator called it without that flag â€” so tracks were computed,
   returned in memory and discarded, leaving `GET /api/v1/tracks` empty after
   a successful run.
3. The reproducibility report right-aligned its verdict inside brackets, so
   the validation generator found no matches and reported "0 passed" - a number
   that reads exactly like a pass.

## 22. Start the backend

```bash
uvicorn extreme_weather.api.main:app --reload
```

```bash
# verify
curl http://localhost:8000/health
# {"status":"ok","project":"SIH26078","service":"DeepWeather AI","version":"0.1.0","environment":"development"}
```

Interactive API documentation: <http://localhost:8000/docs>

| Method | Endpoint | Purpose |
| --- | --- | --- |
| GET | `/` | service information |
| GET | `/health` | liveness: is the process running? |
| GET | `/ready` | readiness: `READY` / `DEGRADED` / `NOT_READY` |
| GET | `/api/v1/system/status` | **the single best request** (Phase 11) |
| GET | `/api/v1/datasets` | list NetCDF datasets in the data directory |
| GET | `/api/v1/datasets/{path}/summary` | validated metadata of one dataset |
| POST | `/api/v1/datasets/validate` | dry-run the data contract |
| GET | `/api/v1/events` | detected extreme weather events (Phase 1) |
| GET | `/api/v1/events/{event_id}` | one detected event |
| GET | `/api/v1/tracks` | tracked event trajectories (Phase 2) |
| GET | `/api/v1/tracks/{track_id}` | one tracked event |
| GET | `/api/v1/tracks/{track_id}/trajectory` | positions of one track |
| GET | `/api/v1/gnn/status` | GNN model availability and provenance (Phase 3) |
| GET | `/api/v1/gnn/model` | GNN architecture, features and hyper-parameters |
| POST | `/api/v1/gnn/predict` | prototype next-state prediction for one track |
| GET | `/api/v1/downscaling/status` | AI downscaling model availability (Phase 4) |
| GET | `/api/v1/downscaling/model` | architecture, features, normalisation |
| POST | `/api/v1/downscaling/run` | downscale one frame (nearest / cubic / ai) |
| POST | `/api/v1/downscaling/evaluate` | compare the three methods on one test set |

The events endpoints read the analysis written by `scripts/detect_anomalies.py`
(`data/processed/events.json`); the tracks endpoints read
`data/processed/tracks.json` written by `scripts/track_events.py`. They never
fabricate data: without the file the response is `count: 0` plus a message
explaining how to produce one. Query filters: `event_type`, `severity`,
`start_time`, `end_time`, `min_cell_count` (events) and `event_type`, `status`,
`start_time`, `end_time` (tracks). The GNN endpoints read the checkpoint written
by `scripts/train_gnn.py`; `/gnn/status` never fails, reporting
`available: false` when no model has been trained yet, and `/gnn/predict`
answers 503 (no checkpoint), 404 (unknown track) or 422 (invalid request). Every
GNN response carries the prototype `notice` field.

## 23. Start the frontend

```bash
cd frontend
npm install
npm run dev        # http://localhost:5173
npm run build      # type-check + production bundle into frontend/dist
npm run preview    # serve the production bundle
```

The dashboard shows the project name and ID, the live backend health panel
(loading / success / error states), the **tracked extreme events** panel (Phase 2:
track count, type, period, duration, peak intensity, max speed, direction, latest
location, lifecycle status), the **GNN panel** (model status, version,
architecture, recorded losses, a track selector and a prototype next-state
prediction, with a permanent "Research Prototype - Not an Operational Forecast"
banner and no accuracy percentage), the dataset inventory and a Leaflet map
shell highlighting the synthetic domain. Point it at a different backend with
`VITE_API_BASE_URL` (see `frontend/.env.example`).

## 24. Docker usage

```bash
docker compose config          # validate the compose file
docker compose build           # build both images
docker compose up --build      # start backend (:8000) and frontend (:5173)
docker compose run --rm backend pytest -q   # run the test suite in the container
docker compose down            # stop the stack
```

`docker-compose.yml` intentionally contains only two services (backend,
frontend). No database, cache, queue or Kubernetes manifests are needed for
Phase 2.

## 25. Current limitations (Phases 3-11)

**Inherited from Phase 0/1**

- **Data**: the default demo is still synthetic. Phase 6 adds real/historical
  ingestion (NetCDF, optional GRIB, an ERA5 archive source, an NWP model-output
  source) but **not** OPeNDAP, HDF, or arbitrary provider APIs.
- **Domain**: the default synthetic domain is 6Â° x 6Â° at 0.125Â° (~12 km) with 72
  hourly steps - enough to exercise the pipeline, not a full forecast domain.
- **Downscaling**: only classical interpolation (`nearest`, `cubic`). No
  AI/diffusion super-resolution yet.
- **Frontend**: dashboard shell - health, tracks, dataset inventory and a map
  without weather layers, trajectories or alerting UI.
- **Ops**: single-container processes, no reverse proxy, no CI pipeline.

**Real / historical ingestion (Phase 6)**

- **No ground truth.** This repository ships no reference dataset of observed
  events, so **no skill score is reported** - not POD, FAR, CSI, accuracy or
  precision. `evaluate_real_data.py` writes "Ground truth unavailable" instead of
  zeros, because zeros would read as "zero skill", a different and false claim.
- **Thresholds are not calibrated.** The Phase 1 thresholds were chosen for the
  synthetic generator, not derived from real climatology. A real dataset
  producing fewer events may simply mean the thresholds are too high for it.
- **Reanalysis is not observation.** ERA5 is real measurements fused into a
  model: a reconstruction of the past, carrying model error.
- **Historical is not live.** Nothing here ingests a live feed. A downloaded
  reanalysis must never be described as real-time or as a forecast.
- **Validated on a tiny sample**: three days, a 3x3 grid (~53 km), one region.
  That proves the pipeline handles real data; it says nothing about skill.
- **GRIB is untested in this environment.** The optional backend is not
  installed, so the GRIB path was exercised only through its
  missing-dependency error handling.
- **No authentication**, and the API is unauthenticated as it has been through
  Phases 0-5.

**Anomaly detection (Phase 1)**

- **Prototype only**: synthetic "climatology", prototype thresholds, bounded
  deterministic EFI-style score - not an operational product.
- **Short reference**: 72 hourly steps is not a climatology.
- **Three event types**; no physics-based classification and no alerting.

**Event tracking (Phase 2)**

- **Rule-based, not learned**: a fixed cost function links detections; there is
  no uncertainty and no ensemble.
- **Greedy one-to-one assignment**, not a global optimum; no merge/split logic,
  so two storms that merge stay two tracks.
- **Centroid-based**: split/merge events and multi-centre systems can confuse it.
- **No extrapolation**: the Phase 2 tracker never predicts; a track simply ends.
- **Prototype thresholds and matching gates**: not tuned for any region or time
  resolution.
- **Single file, no database**: the API serves the latest `tracks.json`; there is
  no history and no authentication.
- **Verification is synthetic-only**: tests check that the injected event is
  tracked coherently - not skill against observed storms.

**GNN event evolution (Phase 3)**

- **Research prototype, not a forecast.** Trained on synthetic data, never
  validated against observed weather, and explicitly not usable for emergency
  decisions. There is no accuracy figure anywhere in the API or the dashboard,
  because none exists.
- **The demo prediction is physically implausible**: on the available data (one
  track, 4 training samples) the model predicts a ~109 km/h displacement. Train
  loss 0.0014 vs validation 5.88 and test 64.56 - a clear overfit, reported
  rather than hidden.
- **No skill score.** The reported losses are a regression error on synthetic
  trajectories, not a forecast verification of any kind.
- **Tiny training set**: the pipeline yields one track; meaningful training needs
  a multi-year, multi-storm archive.
- **Single-step and single-variable**: no multi-step rollout, no ensemble, no
  uncertainty estimate, and one variable per track.
- **Delta targets are not scale-free**: decoding clamps the predicted position to
  valid ranges, which prevents an invalid coordinate but does not make the value
  plausible.
- **Fixed normalisation** and a **static graph** built once per run: no streaming,
  no online updating, no re-linking.
- **Prototype-scale model** (~6.3k parameters, 2 mean-aggregation layers, no
  attention), chosen for CPU inspectability rather than capability.
- **The Phase 2 linking is unchanged**: the graph predicts the evolution of an
  already-tracked event; a learned GNN *tracker* remains future work.

**AI downscaling (Phase 4)**

- **The high-resolution target is synthetic.** It is generated from the coarse
  field so a model has something with known fine-scale structure to be measured
  against. It is **not** observational truth, satellite data or reanalysis, and
  every metric describes agreement with that generated target only.
- **The AI does not beat cubic interpolation.** Measured on the same test frames:
  MAE 0.12598 (ai) vs 0.11826 (cubic) vs 0.14363 (nearest). The AI beats
  `nearest`, has the best correlation, the best event-peak and event-centre
  preservation, but over-predicts (bias +0.060) and raises FAR to 0.162.
  A smooth interpolant is hard to beat on 43 training frames. Reported as
  measured.
- **Resolution is not accuracy.** All three methods emit a finer grid; only the
  reference comparison says anything about correctness.
- **One variable (precipitation)**, no multi-step rollout, no ensemble, no
  uncertainty, no coupling between variables.
- **Monotone residual**: the ReLU output means the model cannot suppress the
  baseline where the true field is lower.
- **Unvalidated**: no comparison against observed storms, no independent
  verification data, no benchmark against published downscaling methods.
- **Diffusion / GAN super-resolution and real NWP ingestion are future work**;
  neither is implemented here.

**Alerting (Phase 5)**

- **The severities are this project's own categories.** INFO, WATCH,
  ADVISORY, WARNING, SEVERE_WARNING and EXTREME are **not** IMD or WMO
  warning categories and are not calibrated against real weather. Every payload
  that reports one carries that disclaimer.
- **The 0-100 risk score is a rule-based triage aid, not a risk index.** It is a
  weighted sum of nine normalised factors whose reference values (50 mm/h,
  150 km, 60 km/h, 12 h) were chosen to make the demo readable and have no
  meteorological justification. It is not calibrated against impact, casualties
  or exposure, carries no probability meaning, and is not comparable across
  regions, seasons or event types.
- **A missing model reduces the evidence, it does not improve the score.** With
  no GNN checkpoint the prediction factors contribute zero and say so; the alert
  is not more confident, it is *less* evidenced (confidence_label drops).
- **The demo dataset is extreme rainfall only.** The wind and temperature paths
  are structurally present but empirically untested.
- **No real notification is sent.** The only enabled provider writes a local JSON
  log; the email and webhook adapters are stubs that read no credential.
- **No authentication or authorisation** on the acknowledge/resolve endpoints.
  They are open in this prototype, exactly as the read endpoints are.
- **No escalation.** No on-call rotation, no paging policy, no SLA. An alert
  sitting ACKNOWLEDGED is only as good as someone noticing it.
- See [docs/ALERTING.md](docs/ALERTING.md) for the full list.

## 26. Future phases

| Phase | Scope | Key deliverables |
| --- | --- | --- |
| 0 | Data foundation | synthetic generator, validation, preprocessing, baseline downscaling, API, dashboard (**done**) |
| 1 | Anomaly detection | climatology, anomaly scores (incl. EFI-style prototype), thresholds, grid-cell detection, event clustering, events API (**done**) |
| 2 | Event tracking | deterministic event matching, trajectories, movement metrics, lifecycle, tracks API, dashboard panel (**done**) |
| 3 | GNN event evolution | **prototype done**: graph over event states, message passing, next-state prediction. FUTURE: full spatio-temporal GNN (PyG), learned event *linking*, tracking benchmarks |
| 4 | AI downscaling | **prototype done**: synthetic reference, residual CNN, chronological split, baseline comparison, extreme-event metrics. FUTURE: diffusion/GAN super-resolution, multi-variable, real NWP truth |
| 5 | Alerts & operations | **prototype done**: deterministic risk score, alert rules R1-R7, lifecycle, alerts API, dashboard panels, local notification mock. FUTURE: real notification channels, persistence and authentication |
| 6 | Real data & scale | NWP/historical ingestion (Dask-chunked), operational deployment, CI/CD, monitoring |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the technical design of
each phase, [`docs/ANOMALY_DETECTION.md`](docs/ANOMALY_DETECTION.md) for the
anomaly methodology and [`docs/EVENT_TRACKING.md`](docs/EVENT_TRACKING.md) for
the tracking design.
