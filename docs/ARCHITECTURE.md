# DeepWeather AI (SIH26078) - Architecture

> **Status legend**

> **IMPLEMENTED (Phase 0)** - exists in the repository, covered by tests.

> **PROTOTYPE** - implemented, but validated against no observed weather and no
> official warning catalogue. Marked wherever it appears, because "it runs" and
> "it is trustworthy" are different claims.
>
> **FUTURE** - designed here, deliberately not implemented yet.

---

## 0. Where each component sits

~~~text
Weather Data
    ↓
Ingestion & Quality
    ↓
Preprocessing
    ↓
Anomaly Detection
    ↓
Event Tracking
    ↓
Graph Construction
    ↓
GNN Prototype            ← PROTOTYPE
    ↓
AI Downscaling           ← PROTOTYPE
    ↓
Alert Engine             ← PROTOTYPE severities (project categories, not IMD/WMO)
    ↓
Realtime Orchestration
    ↓
Evaluation & Explainability
    ↓
Operations / Dashboard
~~~

Per-layer detail and data contracts follow in sections 2-3. The presentation-ready
version of this diagram is in
[docs/sih/03_system_architecture.md](sih/03_system_architecture.md); the submission
package is in [docs/sih/](sih/).

## 1. System goal

An AI-based extreme weather intelligence system that:

1. ingests large-scale NWP and historical weather data,

2. validates and preprocesses it,

3. detects extreme weather anomalies,

4. identifies events and tracks their location, movement and intensity,

5. applies GNN-based spatio-temporal tracking,

6. downscales ~12 km fields to ~5 km local impact information,

7. serves everything through a FastAPI backend, an interactive map dashboard and

   an alerting layer.

## 2. End-to-end data flow

```text

 [A] synthetic generator            [B] NWP / historical sources   (FUTURE)

     (IMPLEMENTED)                       (FUTURE: GRIB, OPeNDAP, HDF, providers)

            |                                        |

            +-------------------+--------------------+

                                v

                      [C] data/raw/*.nc (+ manifest.json)

                                v

                      [D] validation (data contract)

                                v

                      [E] preprocessing

                         missing values (max_gap) -> derived variables

                                v

                      [F] data/processed/*.nc

                                v

        +-----------------------+-------------------------------+

        v                       v                               v

 [G] anomaly detection   [K] downscaling registry        [L] FastAPI

     (IMPLEMENTED Phase 1)   |  nearest / cubic  (IMPLEMENTED)       |

         v                    |  diffusion SR     (FUTURE Phase 4)    v

 [H] event detection         v                             [M] React dashboard

     (IMPLEMENTED Phase 1)  12 km -> 5 km grids            (IMPLEMENTED: health,

         |                                                  tracks, datasets, map)

         v

 [H2] event tracking       ->  tracks.json

     (IMPLEMENTED Phase 2)

         |

         v

 [I] graph construction    ->  spatio-temporal graph over event states

     (IMPLEMENTED Phase 3)     (temporal + optional spatial edges)

         |

         v

 [I2] GNN evolution prediction -> models/gnn/best.json

     (IMPLEMENTED Phase 3, prototype)  -> GNN API + GNNPanel

         |

         v

 [J] AI DOWNSCALING          ->  models/downscaling/best.pt

     (IMPLEMENTED Phase 4, prototype)  ~12 km -> ~5 km residual CNN

         |                                 -> Downscaling API + DownscalingPanel

         v

 [K] alerts                    ->  data/processed/alerts.json
     (IMPLEMENTED Phase 5)           deterministic risk score + rules R1-R7
         |                           -> REST API + AlertsPanel + AlertMap
         v                           -> notification adapters (local mock)
 [L] REST API (FastAPI)          ->  data/processed/notifications.json
     (IMPLEMENTED Phase 0-5)         acknowledge / resolve workflow

```

## 3. Layers

### [A] Synthetic data generator - IMPLEMENTED

`extreme_weather.data.generator`

| Property | Value |

| --- | --- |

| Dimensions | `time` x `latitude` x `longitude` |

| Default grid | 0.125 deg (~12 km), 6 deg x 6 deg, 72 hourly steps |

| Variables | `precipitation`, `temperature`, `pressure`, `relative_humidity`, `u_wind`, `v_wind` |

| Units | mm h-1, K, Pa, %, m s-1, m s-1 (CF-style attributes) |

| Determinism | `numpy.random.default_rng(seed)`; identical seed => identical arrays |

| Injected event | Gaussian spatial footprint x half-sine temporal pulse |

The injected event is the anchor for future phases: it is spatially localised

(`sigma = radius/2` around its centre), temporally bounded

(`start_index ... start_index + duration_steps`) and exactly zero outside that

window, so a Phase 1 detector can be scored against a known ground truth.

Provenance is written twice: as NetCDF global attributes and as a JSON manifest

(`<name>.manifest.json`) containing generator version, seed, variables,

dimensions, grid resolution (deg and km), the event configuration, the file size

and its SHA-256 checksum. Everything is explicitly labelled **synthetic**.

### [B] Real / historical data ingestion - IMPLEMENTED (Phase 6)

extreme_weather.ingestion turns a real weather file into a dataset the
Phase 0-5 pipeline consumes unchanged.

**Provider independence.** Every source implements one protocol
(WeatherDataSource: 
ame, vailable, etch, load, metadata,
default_origin), so no provider is hard-coded anywhere. Four ship: 
etcdf,
grib, era5 (Open-Meteo archive, no credentials required) and 
wp. A source
that is registered but unusable is still listed with vailable: false and the
reason, so a dashboard can show *why* rather than implying the format is
unsupported.

**Formats.** NetCDF is the reference reader (lazy by default, chunkable via
chunks=). GRIB uses an **optional** backend (cfgrib/eccodes); when it is
absent, MissingDependencyError names the install command rather than returning
an empty dataset that would look like "no events found". HDF and OPeNDAP remain
future work.

**Normalisation is explicit and recorded.** Source variable names are mapped to
the project contract (ALIAS_MAP), units are converted (degC->K, hPa->Pa,
km/h->m s-1, ...) and **an unrecognised unit raises** rather than being
guessed - guessing that mm/day is mm/h would silently corrupt every
downstream threshold. Coordinates are normalised (latitude ascending, longitude
wrapped to [-180, 180], time sorted and de-duplicated) and every transform is
written into the manifest.

**Quality control.** DataQualityReport separates blocking **errors** from
**warnings**, and the checks report rather than fix: an out-of-range value is
surfaced, never clipped. It runs twice per ingestion - once on the file as
delivered, once after normalisation.

**Provenance.** Every ingested dataset gets a DatasetManifest carrying the
source file's SHA-256, a content hash of the array values, the variable/unit/
coordinate transformations, and a three-stage provenance chain
(ingested -> validated -> normalized). A local file is recorded as
source="local_file" with **no URL**; the constructor refuses that combination.

**Synthetic and real never mix.** Each dataset declares an origin of
SYNTHETIC/HISTORICAL/EXTERNAL - there is no default - and is written to

aw/synthetic/ or 
aw/real/. DatasetRegistry filters by origin, time and
variable, so the two can never be silently combined in a single figure.

Credentials are read from the environment only, are never logged, and a missing
one raises an error naming the variable rather than any value. See
docs/DATA_INGESTION.md, docs/DATA_QUALITY.md and
docs/REAL_DATA_VALIDATION.md.

### [C] Data contract and validation - IMPLEMENTED

`extreme_weather.data.validator` defines the contract:

- required dimensions: `time`, `latitude`, `longitude` (each with a coordinate),

- required variables: the six core surface variables,

- `time`: `datetime64`, strictly increasing, >= 2 steps,

- `latitude`/`longitude`: 1-D, finite, strictly monotonic, inside valid ranges,

- every variable must span all required dimensions.

Findings are split into **errors** (unusable) and **warnings** (suspicious but

processable, e.g. a variable that is entirely NaN or a value outside its

plausible physical range). `collect_validation_issues()` returns them,

`validate_dataset()` raises `DatasetValidationError` with a bullet list, and the

API exposes both through `POST /api/v1/datasets/validate`.

### [D] Missing-value handling - IMPLEMENTED

`extreme_weather.data.missing`

```text

max_gap semantics

  interior gap of length L  -> filled iff L <= max_gap   (linear interpolation)

  leading/trailing gap      -> preserved (never extrapolated)

  max_gap = 0               -> fill nothing

  max_gap = None            -> fill every interior gap

  all-missing array         -> returned unchanged, never raises

```

Interpolation runs in float64 and is cast back to the input dtype; Xarray

coordinates, dimensions and attributes are preserved and the input is never

mutated. The implementation is explicit NumPy, so `bottleneck` is not required.

### [E] Derived variables - IMPLEMENTED

`extreme_weather.data.derived`

- `wind_speed = sqrt(u_wind^2 + v_wind^2)` (m s-1)

- `wind_direction = degrees(atan2(-u, -v)) mod 360` - **meteorological

  convention**: the direction the wind blows *from*, clockwise from true north

  (0 = N, 90 = E, 180 = S, 270 = W). The convention is stored in the variable's

  `convention` attribute so downstream consumers cannot misinterpret it.

FUTURE additions: CAPE/CIN, precipitable water, moisture flux convergence,

850 hPa temperature, wind shear indices.

### [F] Preprocessing pipeline - IMPLEMENTED

```text

load -> validate -> missing values (max_gap) -> derived variables -> save

```

`preprocess_dataset()` works on a deep copy, so the raw dataset is never

modified. `preprocess_file()` adds NetCDF I/O, creates output directories

automatically and returns a `PreprocessingReport` (variables, derived variables,

missing counts before/after, values filled, options, warnings) that is also

embedded in the output file's `preprocessing` global attribute as JSON.

The pipeline is deterministic - the same input and options give the same output.

### [G] Anomaly detection - IMPLEMENTED (Phase 1)

`extreme_weather.anomaly`

- `climatology.py`: per-grid-cell reference distribution (percentiles, IQR,

  `month` / `day_of_year` grouping) from the processed archive.

- `scores.py`: `absolute`, `standardized`, `percentile`, `extremeness` and a

  bounded, deterministic **EFI-style** prototype, registered in `SCORE_FUNCTIONS`.

- `thresholds.py`: configurable severity ladders (`NORMAL < WATCH < MODERATE <

  SEVERE < EXTREME`) per variable, with a `PROTOTYPE_THRESHOLD_WARNING`.

- `detector.py`: score + thresholds -> boolean extreme-cell mask per time step.

- `clustering.py`: connected components over the mask -> `ExtremeWeatherEvent`

  with centroid, area, peak intensity, bounding box and severity.

- `service.py` / `store.py`: orchestration and the file-backed event store.

Because the synthetic event is injected at a known place and time, the detector

can be scored against a known ground truth. Methodology:

[`ANOMALY_DETECTION.md`](ANOMALY_DETECTION.md).

### [H] Event detection - IMPLEMENTED (Phase 1)

`clustering.py` performs connected-component labelling (SciPy/sparse) over the

extreme-cell mask and groups adjacent cells into one `ExtremeWeatherEvent` per

time step. Events are **per time step** - nothing links them across time, and

the deterministic `event_id` is exactly what lets the next layer match on.

### [H2] Event tracking - IMPLEMENTED (Phase 2)

`extreme_weather.tracking` links the Phase 1 events into trajectories:

- `metrics.py`: haversine distance (R = 6371.0088 km), speed, bearing, 16-point

  compass direction (`CALM` when undefined), intensity trend, track aggregates.

  All helpers are total: `None`/non-finite input yields `0.0` or `None`, never

  an exception and never a division by zero.

- `matching.py`: `MatchingConfig` (250 km, 120 km/h, 6 h, 0.7 distance / 0.3

  intensity) and a **deterministic, greedy, one-to-one** assignment over

  normalised distance plus intensity mismatch. Rejects type mismatches,

  backward-time pairs, too-large gaps and physically impossible speeds.

- `tracker.py`: walks events chronologically (input order is irrelevant), extends

  open tracks, opens new ones, and assigns `NEW` / `ACTIVE` / `ENDED` plus

  `TRK-<TYPE>-<YYYYMMDDTHHMM>-<NNN>` ids.

- `service.py` / `store.py`: `TrackingService` and the file-backed `TrackStore`

  (`data/processed/tracks.json`, never touching `data/raw`).

Design decisions and limitations: [`EVENT_TRACKING.md`](EVENT_TRACKING.md).

### [I] Graph construction and GNN event evolution - IMPLEMENTED (Phase 3, prototype)

`extreme_weather.gnn` turns the Phase 2 tracks into a spatio-temporal graph and

trains a small message-passing network to predict the **next state** of a

tracked event.

| Module | Responsibility |

| --- | --- |

| `features.py` | 16 node features, 6 edge features, delta targets, documented units |

| `graph.py` | Nodes = event states; temporal edges + optional same-timestamp spatial edges |

| `dataset.py` | **Chronological** 60/20/20 train/validation/test split |

| `model.py` | Input projection -> 2 x mean-aggregation message passing -> MLP head |

| `training.py` | Full-batch Adam, best-on-validation snapshot, seeded |

| `checkpoint.py` | `models/gnn/best.json` with full integrity validation |

| `models.py` | Pydantic response schemas (status / model info / prediction) |

| `service.py` | Checkpoint + track -> validated prediction, typed errors |

Design points: the target is a **delta** (`Δlat`, `Δlon`, `Δintensity`) decoded

back onto the last observed state; the split is **never shuffled**, so no future

information reaches training; normalisation uses **fixed divisors** rather than

fitted statistics, so a checkpoint is readable without the training set.

**Why not PyTorch Geometric here.** PyG is a heavy, platform-sensitive

dependency, and the message passing needed at this scale is a few dozen lines of

`index_add_`. PyTorch is already a Phase 0 project dependency, so the prototype

uses it directly. The PyG comparison for a *full* spatio-temporal GNN is

retained below as the rationale for the future phase.

> **Research prototype on synthetic data.** The demo model is trained on one

> track (4 samples) and its prediction is physically implausible; it has no skill

> score and must not be used operationally. See `docs/GNN.md`.

### [I2] Full spatio-temporal GNN tracking - FUTURE (Phase 3b)

The prototype predicts the evolution of an already-tracked event. It does **not**

change *how* events are linked - that is still the Phase 2 rule-based cost

function. A learned GNN **tracker** (predicting the association matrix instead of

a cost) plus tracking benchmarks remains future work.

**Framework choice: PyTorch Geometric (PyG)** - not DGL. Rationale:

| Criterion | PyTorch Geometric | DGL |

| --- | --- | --- |

| Torch version support | First-class for torch 2.x | Historically lagged behind torch releases |

| Message-passing API | `MessagePassing`, well documented, stable | Mature but more boilerplate for custom layers |

| Spatio-temporal building blocks | `TemporalData` and temporal modules in `torch_geometric.nn` | Heterographs plus manual temporal wiring |

| Installation | One extra wheel (`torch-geometric`, CPU-friendly) | Framework-coupled build variants |

| Ecosystem/docs | Larger community, more tutorials and papers | Smaller |

Install with `pip install -e ".[gnn]"`; it is **not** a Phase 0 dependency.

Graph design: nodes = grid cells or detected event blobs, features =

(coordinates, anomaly magnitude, wind components, pressure gradient,

precipitation, time index); edges = spatial k-nearest neighbours plus temporal

links to the previous step. Message passing over these edges propagates storm

structure, and a per-node classification head decides whether two detections at

consecutive steps belong to the same event.

### [K] Downscaling - IMPLEMENTED (baselines) / FUTURE (AI)

`extreme_weather.downscaling`

```python

from extreme_weather.downscaling import downscale, available_methods

available_methods()                      # ('cubic', 'nearest')

fine = downscale(coarse, 0.05)           # 12 km (~0.125 deg) -> 5 km (~0.05 deg)

smooth = downscale(coarse, 0.05, method="cubic")

```

- `baseline.py`: `nearest_downscale` (reference "do not smooth" baseline) and

  `cubic_downscale` (cubic spline interpolation, needs >= 4 source points per

  interpolated dimension and SciPy).

- `registry.py`: name -> implementation registry with `register_method`,

  `available_methods`, `get_method` and the `downscale()` factory. Unknown method

  names raise `UnknownDownscalingMethodError` (a `ValueError`) whose message lists

  the available methods and points at `register_method`.

- The target grid always covers exactly the source domain, so no extrapolation is

  needed, and non-spatial dimensions, variable name and attributes are preserved.

FUTURE (Phase 4): a diffusion/CNN super-resolution model registered as

`"diffusion_sr"`, selected through the same `downscale(..., method=...)` call and

scored against the `nearest`/`cubic` baselines (RMSE, CSI, sharpness metrics).

### [L] API - IMPLEMENTED (Phase 0 scope)

`extreme_weather.api`

| Method | Endpoint | Purpose |

| --- | --- | --- |

| GET | `/` | service information |

| GET | `/health` | `{"status": "ok", "project": "SIH26078", "service": "DeepWeather AI", ...}` |

| GET | `/api/v1/datasets` | NetCDF files found under the data directory |

| GET | `/api/v1/datasets/{path}/summary` | validated metadata (dims, variables, grid, provenance) |

| POST | `/api/v1/datasets/validate` | dry-run the data contract, returns errors and warnings |

| GET | `/docs`, `/openapi.json` | generated OpenAPI documentation |

Design rules:

- absolute imports only (`extreme_weather...`), never fragile relative imports,

  so `uvicorn extreme_weather.api.main:app --reload` works from the repository

  and from an installed package;

- settings come from `Depends(get_settings)`, which makes the data directory

  injectable (and overridable in tests);

- dataset paths from clients are resolved inside the data directory and rejected

  on traversal attempts or a non-`.nc` extension;

- Pydantic response models live in `extreme_weather.schemas`, decoupled from the

  internal dataclasses;

- CORS is restricted to the configured dev origins.

FUTURE: `/api/v1/anomalies`, `/api/v1/events`, `/api/v1/track`, tile endpoints for

the 5 km grids, alert endpoints, authentication and rate limiting.

### [M] Dashboard - IMPLEMENTED (shell)

`frontend/` - React 19 + Vite + TypeScript + Leaflet/React-Leaflet + Axios.

- `services/api.ts`: typed Axios client (`fetchHealth`, `fetchDatasets`,

  `fetchDatasetSummary`, `describeApiError`) with a configurable

  `VITE_API_BASE_URL`.

- `components/HealthStatus.tsx`: live backend health with explicit

  loading / success / error states and a retry button.

- `components/WeatherIntelligencePanel.tsx`: dataset inventory plus the roadmap of

  FUTURE capabilities, each marked with its phase.

- `components/WeatherMap.tsx`: Leaflet map shell centred on the synthetic domain.

FUTURE: anomaly overlays, event markers and trajectories, a time slider/animation

loop, downscaled tile layers, alert banners, region filters and a fuller

responsive layout.

### [N] Validation and evaluation - IMPLEMENTED (Phase 7)

`extreme_weather.validation` measures what the pipeline produces, on real
historical data, without letting the test period influence anything.

**Strict chronological splitting** (`splits.py`). Train / validation / test with
`max(train) < min(validation) < min(test)` and no overlap. There is no shuffle
option and no random seed, because a random split of a time series is always
wrong. `assert_no_leakage` enforces the invariant on every split.

**Climatology coverage** (`climatology_validation.py`). A bucket with too few
samples is reported as *insufficient* rather than filled from another bucket. A
day-of-year climatology quietly built from a fortnight would be smooth,
confident and wrong, and the confidence is the danger.

**Threshold calibration** (`threshold_calibration.py`). The Phase 1 prototype
thresholds are replaced with percentiles of the **training** distribution. They
are statistically calibrated, never meteorologically validated, and every
report says so.

**Honest metrics** (`metrics.py`, `ground_truth.py`). A metric is either
computed from real data or reported as `null` with an explicit reason - never as
`0.0`, which would read as "zero skill". Skill metrics (POD / FAR / CSI) are
computed only against a catalogue with **stated provenance**, matched by event
type, a time window and a haversine distance.

**Mandatory baselines.** The GNN is scored against persistence; the downscaler
against nearest and cubic interpolation, on identical frames. `compare_metrics`
returns `"improves"`, `"tie"` or `"worse"`, and a loss is reported as a loss.

**Provenance** (`provenance.py`). Every run records the dataset's SHA-256, the
split boundaries, the configuration, each stage's outcome and the library
versions, so any figure can be traced to the bytes and code that produced it.

See `docs/REAL_DATA_VALIDATION.md` and `docs/PHASE7_BASELINE.md`.

### [K] Alerts - IMPLEMENTED (Phase 5, prototype)

extreme_weather.alerts

| Concern | Where |
| --- | --- |
| Severity ladder + configurable cut points | severity.py |
| Alert, RiskAssessment, deterministic ids | models.py |
| Deterministic 0-100 score (9 weighted factors) | engine.py |
| Rules R1-R7, individually testable | 
ules.py |
| Orchestration, lifecycle transitions, dedup | service.py |
| data/processed/alerts.json | store.py |
| NotificationProvider + local mock + stubs | 
otifications.py |
| Deterministic lifecycle demo | simulation.py |
| FastAPI request/response models | schemas.py |
| /api/v1/alerts*, /api/v1/notifications | pi/routers/alerts.py |

The score is sum(weight * normalised_value) over nine named factors whose
weights total 100, so it is a plain arithmetic sum a reader can redo by hand.
Missing optional inputs (GNN checkpoint, downscaled field, track) score zero and
say so in the factor detail, and the alert records which components it actually
used in source_components. There is no database: the store is a versioned JSON
file, mirroring Phases 1 and 2.

**Not operational.** The severities are project-invented categories (not IMD or
WMO), the score is not a validated risk index, and the notification adapters are
a local log plus two non-functional stubs. See docs/ALERTING.md.

### [P] Near-real-time pipeline - IMPLEMENTED (Phase 8, prototype)

Phase 8 adds a repeatable ingestion loop around the existing stages. It
reimplements nothing: each stage calls the same service the CLI scripts call.

```text
        WeatherProvider         LocalFile | OpenMeteo | RemoteHttp | Fixture
              |
              v
        IngestionService        [H] Phase 6: validate, normalise, manifest, register
              |
              v
        QualityGate             VALID | VALID_WITH_WARNINGS | REJECTED
              |
              v
        FreshnessTracker        FRESH | STALE | EXPIRED | UNKNOWN
              |
              v
        PipelineOrchestrator
              +--> PREPROCESS    [C] Phase 0
              +--> ANOMALY       [D] Phase 1
              +--> TRACK         [H2] Phase 2
              +--> GNN           [I] Phase 3  (optional)
              +--> DOWNSCALE     [J] Phase 4  (optional)
              +--> ALERT         [K] Phase 5
              |
              v
        Persist Results         <DATA_DIR>/realtime/runs/*.json
```

**Why a provider abstraction.** The pipeline must run against a provider nobody
has configured yet, and be testable with no network. A concrete provider baked
into the orchestrator would make both impossible, so everything downstream is
written against `WeatherProvider`.

**Three design decisions that are load-bearing:**

1. *Stale data is never silently treated as current.* Every run carries a
   freshness verdict, and an unknown timestamp yields `UNKNOWN`, not `FRESH`.
2. *A rejected quality verdict stops the run before anomaly detection.* The
   downstream stages are optimising code; given scrambled timestamps they return
   a confident, meaningless result rather than an error.
3. *A missing model is `SKIPPED`, never `FAILED`, and never faked.* No GNN or
   downscaling checkpoint means the stage says so and the run still completes.

**Idempotency.** `run_id = sha256(provider | window | newest timestamp | data
checksum)`. A repeated ingestion is recognised and short-circuited, so a
scheduled job firing twice does not double the event catalogue. The checksum is
of the *data*, not the request, so a window that gained an hour does not collide
with the previous one.

**Failure policy.** Ingestion is the only network stage, so it is the only one
that retries. `ProviderTimeoutError` and `ProviderUnavailableError` are retried
with exponential backoff; `ProviderResponseError` is not, because a malformed
payload will be malformed again. A partially-successful run is never marked
complete.

**Incremental processing.** `data/realtime/runs/state.json` records the last
processed and newest available timestamps. `new_steps` is `null` - not `0` -
when it cannot be determined, because guessing "nothing new" would silently drop
data. A full recomputation is required when `new_steps` is `null`, when the
window extends more than `context_steps` (default 24 h) beyond the last
processed time, or on `--force`.

**Scope.** Polling only, single-process file-backed state, no SLA, no failover,
and no demonstrated forecast skill. This does not make the system operational.
See [REALTIME_PIPELINE.md](REALTIME_PIPELINE.md).

### [Q] Model evaluation, explainability and uncertainty - IMPLEMENTED (Phase 9)

Phase 9 adds measurement around the models built in Phases 3 and 4. It
reimplements nothing: the field metrics, identical-frame comparison and
contingency tables all come from `validation/metrics.py`, so Phase 7 and Phase 9
cannot drift apart.

```text
        Phase 6 ingestion -> Phase 7 chronological split
                    |
        Phase 1 anomaly -> Phase 2 tracking
                    |
        Phase 3 GNN -> Phase 4 downscaling
                    |
          +---------+---------+---------+
          |         |         |         |
      Baseline  Explain.  Uncertainty  (no new
     [Q1][Q2]    [Q3]      [Q4]         capability)
          |         |         |
          +---------+---------+---------+
                    |
             EvaluationRun
                    |
             Dashboard / API
```

| Module | Responsibility |
| --- | --- |
| `evaluation/dataset.py` | Leak-checked chronological split, train-only scaler, content-derived id |
| `evaluation/baselines.py` | Persistence and constant velocity (persistence reuses Phase 7) |
| `evaluation/evaluator.py` | Measurement, per-metric verdicts, multi-horizon rollout |
| `evaluation/uncertainty.py` | Validation-residual bootstrap, measured coverage |
| `evaluation/explainability.py` | Typed evidence; never causal |
| `evaluation/downscaling.py` | Availability policy over the Phase 7 comparison |
| `evaluation/models.py` | `MetricResult`, `EvaluationRun`, `Provenance` |

**The status vocabulary is the architecture.** Every quantity is `MEASURED`,
`DERIVED`, `UNAVAILABLE`, `INSUFFICIENT_DATA` or `NOT_EVALUATED`, and the status
travels with the value. `MetricResult.unavailable()` *requires* a reason, and
`float(metric)` on an absent metric raises rather than yielding `0.0`. The
honest-reporting property is therefore structural, not a convention that a
future contributor might forget.

**Leakage is prevented structurally.** `assert_chronological` runs inside
`build_evaluation_dataset`, so a leaking split cannot be constructed. The
train-only scaler takes training indices explicitly, so fitting on the full
record is not an available mistake.

**Uncertainty is measured, not asserted.** Intervals come from validation
residuals; coverage is measured once on test and reported as `UNSUFFICIENT_DATA`
when the sample is too small. `is_calibrated` is a measurement, not a licence to
claim calibration.

**Evidence is never causal.** Explanations carry a fixed disclaimer, type each
item as `supporting` / `context` / `unavailable`, and emit an explicit
`unavailable` entry rather than a plausible-looking value. The test suite
searches generated strings for causal phrasing.

**Scope.** This measures the *pipeline* on reanalysis with models trained on
synthetic data. It is not a model fitted to weather, and it establishes no
forecast skill: no verified ground-truth warning catalogue exists. See
[MODEL_EVALUATION.md](MODEL_EVALUATION.md) and
[EXPLAINABILITY.md](EXPLAINABILITY.md).

### [R] Production hardening and operations - IMPLEMENTED (Phase 10)

Phase 10 adds no capability. It makes the existing system observable, refusable
and reproducible.

```text
      HTTP request
            |
   +--------v---------+   X-Request-ID, correlation, metrics,
   | RequestMiddleware |   standard error envelope
   +--------+---------+
            |
   +--------v---------+   filesystem boundary, extension allow-list,
   |   Router layer    |   input validation
   +--------+---------+
            |
   +--------v---------+   SHA-256 over checkpoints and manifests;
   | Integrity checks  |   a corrupt artefact is never loaded
   +--------+---------+
            |
   +--------v---------+   Phase 0-9 pipeline, unchanged
   |   Application     |
   +--------+---------+
            |
   +--------v---------+   structured redacting logs, system counters,
   |  Observability    |   per-stage timing
   +------------------+
```

| Module | Responsibility |
| --- | --- |
| `observability/logging.py` | Structured, redacting logs; request/run correlation |
| `observability/metrics.py` | System counters and duration histograms |
| `observability/timing.py` | Per-stage timing into both log and metrics |
| `api/middleware.py` | Request id, API metrics, error envelope |
| `api/routers/operations.py` | `/ready`, system metrics, integrity sweep |
| `security.py` | Path boundaries, filenames, safe subprocess |
| `integrity.py` | SHA-256 and metadata checks over artifacts |

**Three boundaries this phase refuses to blur:**

1. *Liveness is not readiness.* `/health` never consults the filesystem or a
   checkpoint, so a missing model cannot restart-loop a healthy process.
   `/ready` reports `READY` / `DEGRADED` / `NOT_READY`, and a missing model
   yields `DEGRADED`, never `READY`.
2. *System metrics are not model metrics.* The observability package contains
   no field whose name could be mistaken for a score, and a test asserts it. A
   request counter and an accuracy claim must not share a namespace.
3. *The error envelope is additive.* `detail` is retained alongside the new
   fields because Phase 0-9 clients read it; no status code changed.

**Message sanitisation follows authorship, not status class.** A message a
handler wrote for a caller is returned (redacted), so a 503 can still say "run
`scripts/train_gnn.py`". An exception nobody handled is replaced with a generic
message and a request id, because its text may contain a path or a credential.

**No scientific behaviour changed for speed.** A test asserts a pipeline run
produces identical stage statuses and messages with and without the
instrumentation. See [PRODUCTION_READINESS.md](PRODUCTION_READINESS.md),
[OBSERVABILITY.md](OBSERVABILITY.md) and [SECURITY.md](SECURITY.md).

## 4. Data contract summary

```text

dimensions : time, latitude, longitude

variables  : precipitation, temperature, pressure,

             relative_humidity, u_wind, v_wind   (float, CF-style units)

coordinates: time      -> datetime64, strictly increasing, >= 2 steps

             latitude  -> 1-D, finite, strictly monotonic, [-90, 90]

             longitude -> 1-D, finite, strictly monotonic, |lon| <= 360

derived    : wind_speed [m s-1], wind_direction [degree, meteorological]

provenance : source/dataset_type attributes + <name>.manifest.json

```

## 5. Testing strategy

| Area | Approach |

| --- | --- |

| Determinism | same seed => `xr.testing.assert_identical` |

| Provenance | manifest contents, checksum length, reproducibility with a fixed timestamp |

| Event injection | temporal/spatial localisation verified against a no-event run |

| Validation | one test per failure mode (missing variable/dim, non-monotonic or out-of-range coordinates, unsorted time, all-NaN warning) |

| Gap filling | no gap, 1-step gap, exactly-`max_gap`, larger-than-`max_gap`, leading, trailing, all-missing, invalid `max_gap` |

| Pipeline | immutability of the input, coordinate/variable preservation, report counters, determinism, auto-created output dirs, reloadable NetCDF |

| Downscaling | achieved resolution, exactness at source points, cubic accuracy versus nearest, registry behaviour, invalid method/resolution/input |

| API | health contract, OpenAPI paths, CORS, dataset listing/summary/validation, 400/404 handling, path-traversal rejection |
| Anomaly detection | climatology grouping, score bounds, threshold ladder, clustering determinism, event field semantics, empty/quiet dataset |
| Tracking | matching gates, trajectory ordering, speed/bearing/trend metrics, lifecycle, and a real tracking run feeding the tracks API |
| GNN | feature/target contract, graph construction, checkpoint round-trip, training loss decrease, inference decoding and clamping |
| Downscaling | achieved resolution, baseline exactness, checkpoint round-trip, chronological split, AI vs baseline comparison, event metrics |
| Alerts | model validation, severity ordering, deterministic scoring, per-rule triggering, duplicate prevention, lifecycle transitions, acknowledge, resolve, expiration, empty event store, missing track / GNN / downscaling model, malformed store, notification mock, API filters, 404, 422, dry-run isolation, simulation determinism, full pipeline |

Fixtures live in 	ests/conftest.py and use small grids (16 x 9 x 9) so the
whole suite runs in about a minute on CPU, with no network access. The Phase 5
alerting fixtures are in 	ests/alert_fixtures.py and build a **real** Phase 1 +
Phase 2 run, so the alert tests exercise the same code path the CLI does rather
than a hand-written alert object.

## 6. Deployment topology (Phase 0)

```text

docker compose

  ├── backend  (python:3.13-slim)  uvicorn ... :8000   + /health probe

  └── frontend (node:22-alpine)    vite dev server :5173

volumes: ./data -> /app/data, ./models -> /app/models

```

## 7. Key design decisions

| Decision | Rationale |

| --- | --- |

| Synthetic data as the Phase 0 source | Reproducible offline demos, no API keys, deterministic tests |

| NetCDF4 + Xarray | CF conventions, ubiquitous in meteorology, lazy/chunkable with Dask |

| Explicit NumPy gap filling | Clear, testable `max_gap` semantics without extra dependencies |

| Registry for downscaling methods | The AI model of Phase 4 drops in without touching call sites |

| `max_gap` leaves edges missing | Refuses to invent data at series boundaries |

| PyTorch Geometric for the GNN | torch 2.x support, spatio-temporal primitives, single extra dependency |

| Injected, known event | Gives the detection and tracking phases an objective ground truth |

| Rule-based tracking before the GNN | A deterministic, explainable baseline the GNN can later be measured against |

| Greedy one-to-one matching | Reproducible and simple; a global optimum is not needed at prototype scale |

| Delta prediction target | Keeps the loss on the movement, not the large constant coordinate offset |

| Chronological (never shuffled) split | Shuffling would leak the future into training and inflate every metric |

| Fixed normalisation divisors | A checkpoint stays readable and reproducible without the training set |

| JSON checkpoint, validated on load | Inspectable and diffable; integrity checks fail loudly instead of guessing |

| Plain PyTorch, not PyG, for the prototype | A few dozen lines of `index_add_` beat a heavy platform-specific dependency |
| Residuals on a bilinear baseline | The network learns only the sub-cell correction; one that learns nothing returns the baseline |
| Area-conserving synthetic reference | Makes the coarse field the exact area average of the target, so the task is well posed |
| Best-on-validation, early stopping off by default | Validation loss falls slowly here, so a patience window stopped training before convergence |
| Chronological split + training-only normalisation | Adjacent frames are nearly identical, so shuffling would leak the future |

| Settings via `pydantic-settings` | One configuration path for scripts, API and tests, no hardcoded paths |

### [J] AI downscaling - IMPLEMENTED (Phase 4, research prototype)

`extreme_weather.downscaling` extends the Phase 0 package. The classical
`nearest` and `cubic` baselines are **preserved unchanged**; the learned method is
*added* to the same registry, so `downscale(field, res, method="ai")` sits
alongside `method="nearest"` and no call site changes.

| Module | Responsibility |
| --- | --- |
| `reference.py` | Synthetic high-resolution target; **area-average conserving** - the coarse field is the exact area average of the reference (residual 1.8e-15) |
| `features.py` | 10 documented input channels from the coarse field and coordinates only; the target is never an input |
| `dataset.py` | **Chronological** 60/20/20 split; normalisation fitted on the **training block only** |
| `model.py` | Residual CNN: projection -> 3 residual blocks -> 1x1 conv -> **ReLU** -> bilinear upsample, added to a bilinear baseline (55,873 parameters) |
| `metrics.py` | MAE, RMSE, bias, correlation, POD, FAR, CSI, frequency bias |
| `inference.py` | AI and baseline paths, plus the extreme-event peak/centre/area metrics |
| `training.py` / `checkpoint.py` | Full-batch Adam, best-on-validation, `models/downscaling/best.pt` with integrity validation |
| `service.py` | Runs a method and produces the baseline-vs-AI comparison |

Resolutions are **detected from the data**, never hard-coded. The target is a
non-negative residual on a bilinear baseline, which guarantees non-negativity by
construction and bounds the damage when the model learns little.

> **The AI does not beat cubic interpolation** on the demo data (MAE 0.12598 vs
> 0.11826; it does beat `nearest` at 0.14363). Reported as measured in
> `docs/AI_DOWNSCALING.md`, which also explains why a finer grid is not
> automatically a more accurate one.

## 8. Phase 4 limitations (technical)

Inherited from Phase 0/1:

- Only synthetic data; no NWP/historical ingestion, no real-data validation.

- Small default domain; not a production forecast area.

- No alerting; no AI downscaling (only `nearest` / `cubic`).

- No authentication, no database, no background processing, no CI.

- Ingestion is eager (`load()` into memory) - Dask chunking is part of the stack

  but not yet used for real sources.

- Climatology, thresholds and the EFI-style score are prototype definitions.

Added by tracking (Phase 2):

- Linking is a fixed cost function: no learning, no uncertainty, no skill score.

- Greedy assignment, not a global optimum; no merge/split handling.

- Centroid-based positions, so a multi-centre system is tracked as its centroid.

- No prediction/extrapolation: a track ends when the event stops being matched.

- Matching gates are defaults, not tuned for any region or time resolution.

- The API serves the latest `tracks.json` only - no history, no re-tracking.

Added by the GNN (Phase 3):

- **Prototype, not a forecast.** Synthetic training data, no validation against

  observed weather, no skill score, not usable operationally.

- **Overfits the available data**: the demo run trains on 4 samples and its

  prediction is physically implausible (train loss 0.0014 vs test 64.56).

- Single-step and single-variable: no multi-step rollout, no ensemble, no

  uncertainty estimate, no coupling between variables.

- Delta targets are not scale-free; decoding clamps the predicted position but

  cannot make it plausible.

- Fixed normalisation and a static graph rebuilt per run: no streaming and no

  online updating.

- Prototype-scale model (~6.3k parameters, mean aggregation, no attention),

  chosen for CPU inspectability rather than capability.

- Event *linking* is untouched - the learned GNN tracker remains future work.

Added by downscaling (Phase 4):

- **The high-resolution target is synthetic**, generated from the coarse field.
  It is not observational truth, satellite or reanalysis, and every metric
  measures agreement with that generated target only.
- **The AI does not beat cubic interpolation** on point error. It beats
  `nearest` and preserves the event peak and centre best, but it over-predicts
  (bias +0.060) and raises FAR; that is the whole claim.
- **Higher resolution is not accuracy.** All three methods emit a finer grid.
- One variable (precipitation); no multi-step rollout, ensemble or uncertainty.
- The ReLU residual is monotone, so the model cannot suppress the baseline where
  the true field is lower.
- 43 training frames, one synthetic domain, no evidence of transfer.
- Unvalidated against observed storms; no benchmark against published methods.
- Diffusion/GAN super-resolution, real NWP ingestion and alerting are future
  work.
