# Phase 7 baseline (recorded before any Phase 7 change)

This file records the state of the repository **before** any Phase 7 work, as
required by the Phase 7 procedure. No test was modified, skipped or deleted
before this measurement.

## Test suite

```bash
python -m pytest -q
```

| Metric      | Value |
| ----------- | ----- |
| Total       | 1105  |
| Passed      | 1105  |
| Failed      | 0     |
| Skipped     | 0     |
| Xfailed     | 0     |
| Duration    | 145.26 s |

Breakdown of the pre-existing phases:

| Block                    | Tests |
| ------------------------ | ----- |
| Phase 0-5 (pre-Phase 6)  | 872   |
| Phase 6 (ingestion)      | 233   |
| **Total**                | **1105** |

## Frontend

```bash
cd frontend && npm run build
```

Result: **exit 0**, `tsc --noEmit` clean, 0 TypeScript errors, 0 JSX errors,
Vite build completed in 4.72 s.

## Docker

```bash
docker compose config
```

Result: **exit 0**, configuration parses cleanly.

## Environment

| Property        | Value                |
| --------------- | -------------------- |
| Python          | >= 3.11              |
| Package version | 0.1.0                |
| Phase           | 6 (real / historical data ingestion) |

## What Phase 6 left in place that Phase 7 builds on

* `extreme_weather.ingestion` - real/historical ingestion, quality control,
  manifests with provenance, dataset registry.
* Real ERA5 reanalysis ingestion works end to end (Phase 6 validated this).
* Phase 1 anomaly detection, Phase 2 tracking, Phase 3 GNN, Phase 4 downscaling
  and Phase 5 alerting all exist and are covered by tests.
* `docs/REAL_DATA_VALIDATION.md` - the Phase 6 validation write-up, which
  established that **no ground truth is available**.

## The gap Phase 7 addresses

Phase 6 proved the pipeline *can consume* real data. It did not measure whether
anything it produces is any good, because:

* the Phase 1 thresholds are prototype values chosen for synthetic data;
* the climatology is a reference distribution over a few days, not a
  climatology;
* no train/validation/test split existed, so no model result meant anything;
* no baseline was ever compared against, so an "improvement" was unmeasurable.

Phase 7 closes that gap. The headline expectation is set correctly in advance:
**the honest outcome may well be that the AI components do not beat their
baselines**, and the deliverable is a report that says so rather than one that
says otherwise.
