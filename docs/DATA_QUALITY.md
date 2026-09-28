# Data quality control (Phase 6)

Real data fails in ways the deterministic synthetic generator never does. This
document lists every check the ingestion layer applies before a dataset is
allowed into the pipeline, and what each one does when it fires.

The governing principle: **checks report, they never fix.** A quality layer that
silently clipped, interpolated or dropped data would leave a manifest claiming
the file was processed when it had in fact been altered. Everything here is
recorded instead.

## Errors vs warnings

| Severity | Meaning                                                            |
| -------- | ------------------------------------------------------------------ |
| `errors` | The dataset cannot be trusted for processing. **Blocks.**            |
| `warnings`| The dataset is usable but suspicious. **Recorded, does not block.**  |

`DataQualityReport.valid` is `True` when `errors` is empty. Warnings never make
a dataset invalid — a real reanalysis subset with 12 % missing values is a
legitimate dataset with a caveat, not a broken one.

## The checks

### Structure

| Check | Severity | Notes |
| ----- | -------- | ----- |
| `time`, `latitude`, `longitude` present | error | Without them the dataset cannot be gridded. |
| At least one data variable | error | An empty dataset analyses nothing. |
| Unexpected dimensions | warning | Carried through; unused by Phase 1-5. |

### Time

| Check | Severity | Notes |
| ----- | -------- | ----- |
| Time coordinate non-empty | error | |
| Strictly increasing | error | A genuinely out-of-order axis blocks. |
| Duplicate timestamps | warning | Counted; the normalizer keeps the first occurrence. |

Duplicates are reported as a warning rather than *also* as a blocking
"not strictly increasing" error: the normalizer removes them, so telling an
operator their file is unusable would be wrong about a trivially fixable
problem. Only a genuinely out-of-order axis blocks.

### Coordinates

| Check | Severity | Notes |
| ----- | -------- | ----- |
| Latitude within `[-90, 90]` | error | |
| Longitude within `[-360, 360]` | error | `0-360` is a *legal* convention and is not rejected. |
| Coordinate values finite | error | NaN/Inf in a coordinate breaks every spatial join. |
| Coordinate non-empty | error | |
| Duplicate coordinate values | warning | Counted. |

### Values

| Check | Severity | Notes |
| ----- | -------- | ----- |
| Variable entirely NaN | error | Nothing to analyse. |
| NaN fraction ≥ 10 % | warning | Missing-value filling will dominate the result. |
| Values outside the plausible range | warning | Reported **as-is**; never clipped. |
| Same, with `strict_ranges=True` | error | Opt-in for a strict validation run. |

Plausible ranges are sanity bounds, not scientific thresholds:

| Variable            | Plausible range    |
| ------------------- | ------------------ |
| `precipitation`     | 0 – 500 mm h-1     |
| `temperature`       | 180 – 350 K        |
| `pressure`          | 85 000 – 110 000 Pa |
| `relative_humidity` | 0 – 100 %          |
| `u_wind`, `v_wind`  | -150 – 150 m s-1   |

A value outside them is *reported and passed through*. This project has no
business deciding that a 600 mm/h rainfall rate is impossible — it might be, and
a real event is exactly the case where the check must not get in the way.

### Metadata

| Check | Severity | Notes |
| ----- | -------- | ----- |
| `units` attribute present on a standard variable | warning | The normalizer assumes the standard unit and says so. |
| Contract variables present | (informational) | Listed in `missing_standard_variables`; **not invented**. |

## The report

```json
{
  "valid": true,
  "errors": [],
  "warnings": ["Variable 'precipitation' is 12.5% NaN - ..."],
  "error_count": 0,
  "warning_count": 1,
  "missing_fraction": 0.125,
  "time_steps": 72,
  "spatial_points": 9,
  "variables": ["precipitation", "pressure", "relative_humidity", "temperature", "u_wind", "v_wind"],
  "per_variable_nan": { "precipitation": 0.125, "temperature": 0.0 },
  "duplicate_timestamps": 0,
  "duplicate_coordinates": 0,
  "unexpected_dimensions": [],
  "missing_standard_variables": [],
  "generated_at": "2024-06-01T12:00:00+00:00"
}
```

`check_quality()` **never raises** for a bad dataset — returning a report is the
entire point. `require_valid()` is the explicit opt-in for callers that want an
exception, and its message lists every error at once so an operator does not
have to fix them one at a time.

## Where it runs

| Point | When |
| ----- | ---- |
| `IngestionService.ingest` | Twice: on the file *as delivered*, and again after normalisation. |
| `DatasetRegistry.validate` | On demand, for a stored dataset. |
| `POST /api/v1/data/validate` | Read-only. Writes nothing. |
| `GET /api/v1/data/datasets/{id}/quality` | Re-runs the check against the current file. |
| `POST /api/v1/data/datasets/{id}/process` | Phase 0 validates again before preprocessing. |

Running the check **twice** during ingestion is deliberate: the first pass
describes the file as it arrived, which is what an operator needs when
something looks wrong; the second describes the data that will actually be
analysed, which is what the manifest records.

## The manifest embeds the summary

Every ingested dataset stores its quality summary in the manifest, so a
downstream reader can see whether a dataset was clean *without* re-reading and
re-checking the file:

```json
"quality_summary": {
  "valid": true,
  "errors": [],
  "warnings": [],
  "error_count": 0,
  "warning_count": 0,
  "missing_fraction": 0.0
}
```

That is also why a quality summary never contains an accuracy, skill or
verification figure: this layer measures *data quality*, not model performance,
and conflating the two would let a clean file look like a validated model.
