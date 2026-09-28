# Real data validation (Phase 7)

How this project measured what it can and cannot demonstrate, on **real
historical data**, under a method that cannot leak the test period into
training.

The headline expectation was set before the work started: **the honest outcome
may well be that the AI components do not beat their baselines.** This document
reports what actually happened.

---

## 1. Dataset source

| Property | Value |
| -------- | ----- |
| Source | ERA5 reanalysis via the Open-Meteo archive API |
| Attribution | ECMWF ERA5, Copernicus Climate Change Service |
| Licence | ECMWF ERA5 licence (Copernicus) - free for non-commercial use |
| Period | 2024-06-17 00:00 UTC .. 2024-06-30 23:00 UTC (336 hourly steps) |
| Grid | 5 x 5 points over Karnataka, India |
| Resolution | 0.5053 deg (~56 km) |
| Variables | precipitation, temperature, pressure, relative humidity, u_wind, v_wind |
| Missing data | 0.0 % |
| Credentials | none required |

### Why ERA5 is a *reanalysis*, not an observation

ERA5 is real: it is built from genuine measurements - station reports, radiosondes,
satellite retrievals, buoys, aircraft - fused with a physical model through a
data-assimilation system, at 31 km resolution, every hour, from 1940 to the
present.

That makes it **a reconstruction of the past, not a direct reading of it**, and
it is emphatically not a forecast. Three consequences run through this document:

1. ERA5 has **model error** even where the observations are dense.
2. ERA5 describes the **past only**. Fetching it now tells you nothing about the
   present or the future.
3. Nothing in this project ingests a **live** feed, and a downloaded reanalysis
   must never be described as "real-time".

---

## 2. Train / validation / test methodology

The split is **strictly chronological**. Weather data is a time series, and
shuffling it before splitting produces a model that has seen next week's rain
while learning about today's - after which every number it reports is
meaningless. There is no shuffle option and no random seed anywhere in
`extreme_weather.validation.splits`, because a random split of a time series is
always wrong.

| Split | Steps | Period |
| ----- | ----- | ------ |
| Train | 202 | 2024-06-17 00:00 .. 2024-06-25 09:00 |
| Validation | 67 | 2024-06-25 10:00 .. 2024-06-28 04:00 |
| Test | 67 | 2024-06-28 05:00 .. 2024-06-30 23:00 |

The invariant, asserted by `assert_no_leakage` on every split:

```text
max(train_time) < min(validation_time) < min(test_time)
```

with no timestamp in two splits and none dropped.

### Where leakage is actually prevented

Three statistics could each leak the test period, and each is computed from the
**training slice only**:

1. **The climatology.** The Phase 1 service builds it from a reference
   *dataset*, and the reference passed in is the training split.
2. **The thresholds.** `calibrate_thresholds` takes the training slice; the
   calibration period is recorded in the report so the boundary is checkable.
3. **The downscaling target construction.** Built from the training split.

The test split is scored **once**. Model checkpoints are selected on validation
loss, never on test.

---

## 3. Climatology methodology

A climatology is a reference distribution over a long period. Fourteen days is
not that, and the report says so rather than filling the gap:

| Grouping | Buckets sufficient | Usable | Message |
| -------- | ------------------ | ------ | ------- |
| month | 1 / 12 | No | "Insufficient historical coverage for monthly climatology." |
| day_of_year | 0 / 366 | No | "Insufficient historical coverage for day-of-year climatology." |
| hour_of_day | 0 / 24 | No | "Insufficient historical coverage for hour-of-day climatology: 14 distinct day(s), 30 required." |

This refusal is the point. A day-of-year climatology built from a fortnight and
quietly filled from a pooled monthly mean would produce **smooth, confident,
wrong** numbers - and the confidence is the danger. A real climatology needs
decades; this dataset has a fortnight, and the report measures the gap instead of
hiding it.

**No ground truth needed for this**: coverage is a property of the record, not a
comparison against an event catalogue.

---

## 4. Threshold calibration

The Phase 1 thresholds shipped with the project are prototype values chosen to
make the synthetic demo fire. On real data they are arbitrary. Phase 7 replaces
them with thresholds **derived from the training period of the dataset being
analysed**.

| Variable | Samples | WATCH | MODERATE | SEVERE | EXTREME |
| -------- | ------- | ----- | -------- | ------ | ------- |
| precipitation | 5050 | 4.9 | 10.5 | 15.0 | 15.0 mm h-1 |
| temperature | 5050 | 31.5 | 32.5 | 32.8 | 32.8 C |
| u_wind | 5050 | 7.69 | 9.39 | 10.3 | 10.3 m s-1 |
| v_wind | 5050 | 3.69 | 5.30 | 5.77 | 5.77 m s-1 |

These are the 99th, 99.9th, 99.99th and 100th percentiles of the training
distribution, mapped onto the project's own severity ladder. Calibration is
**deterministic**: the same training data and percentiles always produce the
same thresholds.

### What calibration is *not*

These thresholds are **statistically calibrated, not meteorologically
validated**. Nothing here compares them to an IMD or WMO criterion, because this
project has no access to one. Every calibration report carries:

> "Thresholds are statistically calibrated to the training period of this
> dataset. They are NOT validated against any official or meteorological warning
> criterion, and no such comparison was possible."

Note the tied upper steps (15.0 mm h-1 twice): the field is zero most of the
time, so the top percentiles coincide. The code nudges tied steps apart to keep
every severity reachable, and records that it did so.

---

## 5. GNN evaluation

The baseline is **persistence**: `predicted_next = current_state`. A GNN that
cannot beat "nothing moves" is adding noise, not signal, and the report has to be
able to say so.

**Result: not computed.** The validation period produced one track with two
graph nodes, and neither node fell in the test split with a known next state. The
report states:

> "The test split contains no node with a known next state, so no metric was
> computed."

This is the correct outcome. Training on the test split to manufacture a number
would have produced a result that meant nothing, and a `verdict` of "beats
persistence" from two samples would be the single most misleading line this
project could print. **No GNN metric is claimed, in either direction.**

The mechanism is tested (`test_a_model_that_moves_the_wrong_way_loses_to_persistence`,
`test_a_model_that_predicts_nothing_moves_ties_persistence`), so the comparison
is verified to be *capable* of reporting a loss - it simply had nothing to
measure here.

Run it yourself with a longer record:

```bash
python scripts/evaluate_gnn_real.py \
    --tracks data/validation/reports/phase7_era5_tracks.json --epochs 40
```

---

## 6. Downscaling evaluation

Compared on **identical frames**: nearest, cubic and AI.

**Result: only the interpolation baselines were scored.** No trained AI
downscaling checkpoint exists in this environment, so the AI is **omitted
rather than imitated** by a baseline wearing its name. The report says:

> "No trained AI downscaling checkpoint exists, so only the interpolation
> baselines were scored. No AI result is claimed."

Two further honesty constraints are enforced by the code and asserted by tests:

* **POD/FAR are not computed.** No defensible event threshold exists for a
  synthetic reference target, and a POD against an arbitrary cut would be a
  number with no meaning.
* **The reference is the Phase 4 synthetic high-resolution target**, generated
  from the coarse field. It is *not* observational truth, so these numbers
  measure interpolation of a known construction, not forecast skill.

The comparison is verified to work in both directions by test: a biased AI field
is correctly reported as losing to an exact cubic baseline, and an accurate one
as winning.

---

## 7. Alert evaluation

| Metric | Value |
| ------ | ----- |
| Alerts on the validation split | 2 |
| Severities | 1 WARNING, 1 ADVISORY |
| Trigger components | events, tracking |
| Alert-event matching skill | **not computed** - no ground truth |

The severities are **this project's own categories**, not official IMD or WMO
warnings. The risk score is a transparent rule-based triage aid with per-factor
explanations, not a validated disaster-risk index and **not a probability of
occurrence**.

---

## 8. Ground-truth limitations

**Ground truth unavailable - metric not computed.**

This is a fact about the project, not a gap in the code. No reference event
catalogue ships here, so:

| Metric | Status |
| ------ | ------ |
| POD | not computed |
| FAR | not computed |
| CSI | not computed |
| Precision / Recall / F1 | not computed |
| Accuracy | not computed |
| Any skill score | not computed |

Every metric field is `null` with the explicit note attached. **None is `0.0`**,
because a zero reads as "the method has zero skill", which is a completely
different - and false - claim.

`extreme_weather.validation.ground_truth` exists so that a user who *does* have
a trusted catalogue can use it. It requires stated provenance: a file of
timestamps with no source and no matching criteria cannot be treated as truth,
because its own errors would then be scored as the model's. Matching is by
event type, a time window and a haversine distance, with the criteria recorded
in the report.

Optional ground-truth support lives in `data/reference/`.

---

## 9. Reproducibility

```bash
# 1. Fetch and ingest a bounded ERA5 subset, then run every stage.
python scripts/run_phase7_validation.py --days 14 --end-date 2024-06-30

# 2. Read the report.
cat data/validation/reports/phase7_summary.md

# 3. Reproduce without re-downloading.
python scripts/run_phase7_validation.py --days 14 --end-date 2024-06-30 \
    --reuse-existing
```

Every run writes `data/validation/provenance/<run_id>.json` recording the
dataset's SHA-256, the split boundaries, the configuration, each stage's
outcome, and the Python/NumPy/Xarray versions. Downloads are bounded: a request
over 120 days or without an explicit bounding box is refused outright.

Artifacts:

```text
data/validation/
    raw/          the ingested source, in the archive's own units
    processed/    the Phase 0 output and per-split files
    manifests/    the Phase 6 provenance manifest
    splits/       the train/validation/test boundaries
    reports/      every JSON and Markdown report
    provenance/   one record per run
```

---

## 10. Known limitations

1. **Research prototype.** Not production-ready, not operational, not a
   guaranteed early warning, not 100 % accurate, not an exact prediction engine,
   and not verified against any issued warning.
2. **No ground truth**, so no detection skill is reported - by design.
3. **Fourteen days is not a climatology.** Monthly, day-of-year and
   hour-of-day are all reported as insufficient.
4. **Thresholds are statistically calibrated, never meteorologically
   validated.**
5. **Alert severities are project categories**, not official warnings, and the
   risk score is not a probability.
6. **The GNN could not be evaluated** on this record: too few test nodes.
7. **The AI downscaler was not evaluated**: no trained checkpoint exists here.
8. **The downscaling reference is synthetic**, so it measures interpolation of a
   known construction rather than forecast skill.
9. **ERA5 is reanalysis, not observation** - and it is historical, never live.
10. **No claim of disaster reduction, lives saved or economic benefit.** None of
    those was measured, and none is implied anywhere in this project.
11. **No authentication.** The API is unauthenticated, as it has been through
    Phases 0-6.

