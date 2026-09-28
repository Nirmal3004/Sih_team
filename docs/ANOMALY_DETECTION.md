# Extreme Weather Anomaly Detection (SIH26078, Phase 1)

> **Prototype.** Everything in this document runs on the synthetic Phase 0
> dataset with prototype thresholds. It is **not** an operational forecast, not an
> official warning system, and not a reproduction of any meteorological
> institution's product.

---

## 1. What "anomaly detection" means here

An **anomaly** is the difference between an observation and a *reference state*
that describes what is normal for that place and time. In this project the
reference state is a **climatology**: a per-grid-cell, per-calendar-group
summary of a historical (here: synthetic) dataset.

```text
reference dataset  ──►  climatology   (mean, std, count, quantile function)
                                  │
data / current field ────────────┼──►  anomaly score  (5 scores, see §3)
                                  │
                                  ▼
                      thresholds  ──►  extreme-cell mask
                                  │
                                  ▼
                      clustering  ──►  ExtremeWeatherEvent objects
                                  │
                                  ▼
                      events.json  ──►  GET /api/v1/events
```

A value is called *extreme* here when its score crosses a configured threshold
for its variable - never by comparing against a hard-coded number inside the
detector.

## 2. Climatology (reference distribution)

`extreme_weather.anomaly.climatology.build_climatology()`

```python
climatology = build_climatology(dataset, "precipitation", "month")
```

| Property | Value |
| --- | --- |
| Groupings | `month` (calendar month), `day_of_year` (1-366), `all` (pooled) |
| Statistics | `mean`, `std`, `count`, `min`, `max` |
| Quantile function | `quantiles` on 99 probability levels (0.01 ... 0.99) |
| Missing values | skipped (`skipna=True`); `count` records data availability |
| Short reference | groups below `min_samples` (default 10) use the pooled group |
| No coverage | `fallback="none"` yields NaN instead of silently pooling |

The returned object is a plain `xarray.Dataset`, so it can be inspected,
subset, chunked with Dask or written to NetCDF.

**It is not real climatology.** Every climatology carries
`source="synthetic (Phase 0/1 prototype reference distribution)"` and a comment
stating that it must not be used for operational decisions.

## 3. Anomaly scores

All scores live in `extreme_weather.anomaly.scores` and are registered in
`SCORE_FUNCTIONS`.

### 3.1 Absolute anomaly - `absolute`

```text
anomaly = value - reference_mean          [units of the variable]
```

Signed, directly interpretable in physical units, and the default detection
score because it keeps usable headroom on a short reference (see §6).

### 3.2 Standardized anomaly - `standardized`

```text
z = (value - reference_mean) / reference_std        [dimensionless]
```

A standard deviation at or below `min_std` (default `1e-6`) means "no spread",
where a z-score is undefined: the result is NaN by default
(`zero_variance="zero"` returns `0.0`). For precipitation the reference has a
large mass at zero, which makes `std` tiny and the z-score huge - a property of
the data, not a bug.

### 3.3 Percentile and extremeness - `percentile`, `extremeness`

The percentile `p` of a value is estimated by **interpolating the stored
empirical quantile function** of the reference:

```text
p = interpolation of q(p) at the value, clipped to [q_min, q_max]
```

so it saturates at the extreme stored levels (0.01 and 0.99 by default). The
bounded tail score is

```text
extremeness = clip((p - 0.5) / 0.5, 0, 1)
```

0 at or below the reference median, 1 beyond the highest reference quantile.

### 3.4 EFI-style score - `efi`

Inspired by the ECMWF Extreme Forecast Index, defined in this project as:

```text
EFI = Phi^-1( (p_fc - p_clim) / (1 - p_clim) )

p_fc   = percentile of the current value inside the reference distribution
p_clim = climatological percentile, configurable (default 0.9)
Phi^-1 = standard-normal quantile function (scipy.special.ndtri)
```

The ratio is clipped to `[0, 1 - 1e-6]` and the score to `[-3, 3]`, so it is
always finite:

| Score | Meaning |
| --- | --- |
| `<= -3` | at or below `p_clim` (ratio clipped at 0, so the score saturates) |
| `0` | just above `p_clim` |
| `+1` | ~84th percentile of the upper tail |
| `+2` | ~97.7th percentile of the upper tail |
| `>= +3` | beyond the stored reference quantiles |

**What this is not.** The operational ECMWF EFI compares the *probability
distribution of an ensemble forecast* with a climate-model climatology and
separates shift and spread effects. This project has no ensemble, no climate
model and no calibration, and the score saturates as soon as a value passes the
highest stored reference quantile (with a short reference the maximum reachable
value is `Phi^-1(1) ~= 1.28`). It is a *bounded, deterministic EFI-style
prototype*; the default detection uses the absolute anomaly instead.

The definition is also stored in every output: `EFI_FORMULA` in `scores.py`, the
`formula` attribute of the score field, and `metadata["score_formula"]` in
`events.json`.

## 4. Thresholds

`extreme_weather.anomaly.thresholds` - severity classes `NORMAL < WATCH <
MODERATE < SEVERE < EXTREME` (a Pydantic `StrEnum`) and a `ThresholdLadder`
mapping ascending score thresholds to severities:

```python
ThresholdLadder(
    name="precipitation_absolute",
    event_type=EventType.EXTREME_RAINFALL,
    variable="precipitation",
    score_name="absolute",
    steps=((10.0, WATCH), (25.0, MODERATE), (40.0, SEVERE), (50.0, EXTREME)),
)
```

* a score **equal** to a threshold already belongs to that severity;
* scores below the first threshold are `NORMAL`;
* ladders must be strictly ascending, must not contain `NORMAL`, and must end at
  `EXTREME`;
* thresholds are **per variable**, so precipitation (mm/h), wind speed (m/s) and
  temperature (K) can use completely different bands;
* a `ThresholdSet` groups ladders and validates unique names.

Shipped prototype ladders (`DEFAULT_THRESHOLD_LADDERS`, absolute scale):

| Variable | WATCH | MODERATE | SEVERE | EXTREME |
| --- | --- | --- | --- | --- |
| `precipitation` (mm/h) | 10 | 25 | 40 | 50 |
| `wind_speed` (m/s) | 5 | 10 | 15 | 20 |
| `temperature` (K) | 2 | 4 | 6 | 8 |

`EFI_THRESHOLD_LADDERS` and `STANDARDIZED_THRESHOLD_LADDERS` provide the same
structure on the other two score scales.

**These are prototype values.** They are not IMD (or any other official) warning
criteria, they are not calibrated against real data, and every ladder carries
`note=PROTOTYPE_THRESHOLD_WARNING`, which is stored in `events.json`.

## 5. Detection, clustering and the event schema

**Detection** (`detector.detect_extreme_cells`) turns a score field plus a ladder
into a boolean mask. The detection level is the first ladder threshold, so
`mask == (severity != NORMAL)` always holds, and NaN scores are never extreme.

**Clustering** (`clustering.cluster_extreme_events`) groups connected cells per
time step with `scipy.ndimage.label` (4- or 8-connectivity, no Python loop over
cells):

| Property | How it is computed |
| --- | --- |
| `cell_count` | number of labelled cells |
| `latitude`, `longitude` | intensity-weighted centroid, weight = `max(score - detection_level, 0) + 1e-6` (the floor keeps a constant-score cluster well defined) |
| `affected_radius_km` | largest haversine distance from the centroid to any member cell (Earth radius 6371.0088 km) |
| `intensity`, `max_value`, `mean_value` | peak / mean of the physical field inside the cluster |
| `anomaly_score` | peak score inside the cluster |
| `severity` | classification of the peak score with the ladder |
| `event_id` | `EVT-<TYPE>-<YYYYMMDDTHHMM>-<NNN>`, deterministic, numbered by descending peak score within the time step |

**Event model** (`schemas.event.ExtremeWeatherEvent`, Pydantic, `frozen`,
`extra="forbid"`): `event_id`, `event_type`, `timestamp` (valid time; naive
values are interpreted as UTC), `latitude` in [-90, 90], `longitude` in
[-180, 180], `intensity`, `anomaly_score`, `severity`, `affected_radius_km >= 0`,
`cell_count >= 1`, plus the optional `variable`, `units`, `max_value` and
`mean_value` that the pipeline actually produces. `duration` and `confidence`
are deliberately **absent**: they need event tracking, which is Phase 2+.

## 6. Why the default detection score is the absolute anomaly

Measured on the Phase 0 sample dataset (0.125 deg, 72 steps, one injected event)
against an event-free reference of the same configuration:

| Score | Background max | Event peak | Usable separation |
| --- | --- | --- | --- |
| `absolute` (mm/h) | ~7 | ~55 | yes, wide gap |
| `standardized` (z) | > 100 | ~115 | no (reference std is tiny) |
| `efi` | 1.28 (saturated) | 1.28 (saturated) | no, both clip at the ceiling |

With `absolute` and the prototype ladder the detector reports 8 events across the
10 time steps in which the injected event exceeds the WATCH threshold, with
intensity growing 15 -> 42 mm/h, radius 60 -> 101 km and severity
WATCH -> MODERATE -> SEVERE. The other scores are fully implemented and tested;
they are simply the wrong default for a 72-step synthetic reference.

## 7. Service, CLI and outputs

`extreme_weather.anomaly.service.AnomalyDetectionService` owns the business
logic (load -> climatology -> score -> detect -> cluster -> store) so the router
stays thin. `EventStore` is file backed; there is **no database** in Phase 1.

```bash
python scripts/detect_anomalies.py
python scripts/detect_anomalies.py --input data/processed/sample_history_processed.nc
python scripts/detect_anomalies.py --reference data/raw/sample_history.nc --score efi
python scripts/detect_anomalies.py --variables precipitation temperature --min-cells 3
python scripts/detect_anomalies.py --figure data/processed/anomaly_overview.png
```

Outputs (created automatically, raw data is never modified):

| File | Content |
| --- | --- |
| `data/processed/events.json` | stable schema (version, provenance, thresholds, events) |
| `data/processed/anomaly.nc` | `<var>_score`, `<var>_mask`, `<var>_climatology_mean` |
| `data/processed/anomaly_overview.png` | optional 4-panel figure (field, score, mask, centroids) |

API:

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/events` | list events; filters `event_type`, `severity`, `start_time`, `end_time`, `min_cell_count` |
| `GET /api/v1/events/{event_id}` | one event; 404 with the event count when unknown |

Without an analysis file the list endpoint returns `count: 0` and a `message`
explaining how to produce one. Nothing is ever fabricated.

## 8. Limitations

1. **Synthetic reference.** No real climatology, no real NWP/observed data, no
   reanalysis. Every statistic describes a synthetic sample.
2. **Short reference period.** 72 hourly steps is not a climatology. The
   `day_of_year` grouping is implemented but unusable at that length (each group
   holds a handful of samples and falls back to the pooled group).
3. **Deterministic only.** No ensemble, no uncertainty, no calibration and no
   reliability information. The EFI-style score is a prototype, not an
   operational index.
4. **Prototype thresholds.** Not official warning criteria and not calibrated.
5. **Three event types only** (`EXTREME_RAINFALL`, `EXTREME_WIND`,
   `EXTREME_TEMPERATURE`) and no physics-based classification (no CAPE, no
   shear, no moisture flux).
6. **No tracking yet.** Events are per time step; linking them into a storm life
   cycle is Phase 2+, and the GNN tracker is Phase 3+.
7. **Single file, no database.** The API serves the latest `events.json`; there
   is no history, no persistence layer and no authentication.
8. **Coarse domain.** The demo domain is 6 deg x 6 deg; performance on
   full-domain, multi-year data is untested.
9. **Verification is synthetic-only.** The integration test checks that the
   injected event is found at the right place and time - not skill against real
   extremes.

## 9. Why this is not an operational warning system

* the input is synthetic, so "normal" is whatever the generator produced;
* thresholds are prototype numbers, not the official criteria of any agency;
* there is no forecast model, no ensemble, no lead-time verification and no
  skill score against observed events;
* there is no alerting channel, no duty-of-care review and no fallback;
* a single 72-step synthetic window cannot support any climatological statement.

A real deployment would additionally need a multi-decade quality-controlled
reanalysis, verified warning criteria, probabilistic forecasts, verification
against reported events, and human-in-the-loop review.
