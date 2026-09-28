# 05 - Algorithm pipeline (SIH26078)

## Stage 1 — Ingestion and quality

A `WeatherProvider` returns a labelled Xarray `Dataset` plus provenance. Three
implementations:

| Provider | Source | Network | Nature |
| --- | --- | --- | --- |
| `fixture` | generated locally, seeded | none | **Synthetic** |
| `open-meteo` | ERA5 archive | yes | **Historical reanalysis** |
| `local` | a NetCDF file | none | depends on the file |

Freshness is assessed against configurable windows and recorded as `FRESH`,
`STALE`, `EXPIRED` or `UNKNOWN`. A retry policy with exponential backoff wraps
the fetch. The quality gate then checks the variable whitelist and rejects data
that would produce meaningless downstream results — a rejected dataset stops
the run, it does not continue with a warning.

The manifest records a SHA-256 of the payload, which is what makes
idempotency real rather than assumed: the same ingestion produces the same
identity, so it is skipped rather than repeated.

## Stage 2 — Preprocessing

`load → validate → missing → derived → save`.

Validation is a data contract: required dimensions, coordinate ranges, variable
presence, monotonic time. Gap filling is explicitly bounded (`max_gap`); an
unbounded fill would invent data. Derived variables (`wind_speed`,
`wind_direction`) are computed from components with proper unit handling.

## Stage 3 — Anomaly detection

1. **Climatology** — the reference distribution, grouped by month or
   day-of-year. Built from the **training split only** in any validation
   context; using the full record would leak the future into the past.
2. **Score** — `absolute`, `standardized`, `percentile`, `extremeness` or an
   EFI-style prototype.
3. **Thresholds** — a per-variable ladder producing severity categories.
4. **Grouping** — connected components over the exceedance mask, with a
   minimum-cell guard so a single noisy pixel is not an "event".

## Stage 4 — Event tracking

Events across consecutive time steps are matched with a cost function:

```text
cost = (w_d · d / d_max  +  w_i · intensity_mismatch) / (w_d + w_i)
```

Pairs are assigned greedily by increasing cost, with deterministic
tie-breaking. Defaults: 250 km, 120 km/h, 6 h; weights 0.7 distance /
0.3 intensity. Each track accumulates lifecycle states.

## Stage 5 — Graph construction

A track becomes a chain of nodes with temporal edges. Nodes within
`max_spatial_distance_km` of each other gain spatial edges. Samples are split
**chronologically** — train precedes validation precedes test — and the split
is asserted, not assumed.

## Stage 6 — GNN prototype

Message passing over the graph, then a regression head predicting the spatial
and intensity delta to the next state. **Prototype**: trained on synthetic
data, never validated against observed weather. On the available data it
overfits visibly (train loss 0.0014 against test 64.56), which is reported
rather than hidden. A finer grid is not a more accurate grid.

## Stage 7 — AI downscaling

A residual CNN refines a ~12 km field onto a ~5 km grid. **Prototype**: on the
available data it **loses to cubic interpolation on MAE**, which is recorded in
`docs/AI_DOWNSCALING.md`. The baseline is always computed alongside, so the
comparison is real.

## Stage 8 — Alerting

A rule-based risk score combines intensity, proximity, speed and persistence
against reference values, and a rule engine maps the score to a severity
through configured thresholds. Alerts have a lifecycle: `NEW → ACTIVE →
ACKNOWLEDGED → RESOLVED`, with an `EXPIRED` terminal state and time-bucket
deduplication.

**The severities are this project's own categories, not official IMD or WMO
warnings.** Notifications are recorded to a local append-only log; nothing is
sent to an external service.

## Stage 9 — Orchestration

One run id, derived from the data rather than from the clock, so an identical
ingestion is recognisably the same run. Stages report `SUCCESS`, `WARNING`,
`SKIPPED` or `FAILED`. Optional model stages report `SKIPPED` with the reason;
they never invent a prediction.

## Stage 10 — Evaluation

Every quantity is tagged `MEASURED`, `DERIVED`, `UNAVAILABLE` or
`INSUFFICIENT_DATA`, with the evidence that produced it. Baselines
(constant-velocity, nearest-neighbour, cubic) are always computed alongside, so
a model is only called better when the measurement says so on that metric.
Uncertainty is reported as intervals with a coverage check, not as a single
number.
