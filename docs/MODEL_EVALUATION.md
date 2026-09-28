# Model Evaluation Card (SIH26078, Phase 9 - IMPLEMENTED)

> **Status: IMPLEMENTED.** This card describes working code in
> `src/extreme_weather/evaluation/`.
>
> **Research prototype.** Nothing here establishes forecast skill, disaster
> reduction, or any accuracy claim. No verified ground-truth warning catalogue
> exists for this project, so none is asserted.

## How to read this card

Every quantity is tagged with its epistemic status, and the tag is part of the
result, not commentary on it:

| Tag | Meaning |
| --- | --- |
| **MEASURED** | Computed on a held-out split by comparing against something. |
| **DERIVED** | Computed from an estimated quantity (for example a bootstrap width). |
| **UNAVAILABLE** | Not computed. **Never zero.** Carries the reason. |
| **INSUFFICIENT_DATA** | Attempted; the sample was too small to mean anything. |
| **NOT EVALUATED** | Deliberately out of scope, with the reason recorded. |

`MetricResult` in `evaluation/models.py` enforces this structurally: a value and
a status travel together, and an absent metric **cannot** be coerced to a float.

---

## 1. Model purpose

| Model | Purpose | Not for |
| --- | --- | --- |
| Phase 3 GNN | Predict the next state (position + intensity) of a tracked event from the event graph | Operational forecasting; it is a one-step extrapolation prototype |
| Phase 4 AI downscaler | Refine a coarse field onto a finer grid | Claiming the finer grid is more accurate; it is a different field |

## 2. Training data

| Model | Trained on | Origin |
| --- | --- | --- |
| GNN | Synthetic NWP-style event tracks (Phase 2 output) | `SYNTHETIC` |
| Downscaler | Synthetic coarse/fine field pairs | `SYNTHETIC` |

**Neither model has been trained on ERA5 or on any observational record.** The
Phase 7 evaluation runs the unchanged models over historical reanalysis; that
measures the *pipeline* on real data, not a model trained on it. Any skill
figure from that combination is a statement about transfer, not about a model
fitted to weather.

## 3. Evaluation data

| Item | Value |
| --- | --- |
| Source | ERA5 reanalysis via the Phase 6 ingestion path (`HISTORICAL`) |
| Nature | Historical reanalysis: observations fused with a model, describing the past. **Not live, not a forecast.** |
| Period | As recorded in the run's `provenance.dataset_checksum` / `split_boundaries` |
| Checksum | SHA-256 of the dataset content, recorded per run |
| Grid | 5x5, ~0.1 deg, as produced by the Phase 7 builder |

The dataset identity is **content-derived**: the same bytes always produce the
same `dataset_id`, and different data never collides under one id.

## 4. Temporal split

```text
TRAIN (60%)  ->  VALIDATION (20%)  ->  TEST (20%)
```

Enforced by `assert_chronological`, which is called **during construction**, so
a leaking split cannot be built in the first place:

* blocks are consecutive index ranges, so no step is dropped or double-counted;
* `max(train) < min(validation) < min(test)`;
* explicit set-overlap check as defence in depth;
* the time axis must itself be strictly increasing.

**Learned statistics use training data only.** `fit_train_only_scaler` takes the
training indices explicitly, so fitting on the full record is not an available
mistake. A test asserts that perturbing only the test period leaves the fitted
constants unchanged.

## 5. Features

GNN node features are the 16 values in
`extreme_weather.gnn.features.NODE_FEATURE_NAMES`: scaled latitude/longitude,
scaled intensity, anomaly score, scaled speed, `sin`/`cos` of bearing, a
stationary flag, scaled cell count, `sin`/`cos` of hour of day, and an
event-type one-hot block.

## 6. Target

`[delta_latitude, delta_longitude, delta_intensity]` to the **next hourly
state**. The last point of a track has no target and is excluded - predicting it
would score the model against a state it was shown.

## 7. Baseline models

| Baseline | Formula | Why it exists |
| --- | --- | --- |
| **Persistence** | `next = current` | The minimum bar. A model that cannot beat this adds noise. |
| **Constant velocity** | `next_position = current + v_last * dt`, `next_intensity = current` | Persistence is weak for anything that moves. Beating *this* shows the model learned more than "events drift". |

Intensity is deliberately **not** trend-extrapolated in the constant-velocity
baseline. A trend model for intensity would smuggle a third modelling choice
into the comparison; the documentation says so explicitly.

Both definitions are published in the API response, so a reader never has to
infer what a baseline number means.

## 8. Metrics

### GNN (per sample, on the test split)

| Metric | Unit |
| --- | --- |
| `latitude_mae_deg`, `longitude_mae_deg` | deg |
| `latitude_rmse_deg`, `longitude_rmse_deg` | deg |
| `displacement_error_km_mean` / `_median` / `_std` / `_p90` | km (haversine) |
| `intensity_mae`, `intensity_rmse` | source variable unit |
| `combined_loss` | deg + source unit |

Reported as mean, median, RMSE, standard deviation and p90 where meaningful.

### Downscaling (on identical test frames)

| Metric | Availability |
| --- | --- |
| MAE, RMSE, bias, correlation | **MEASURED** when a high-resolution reference exists |
| POD / FAR / CSI | **UNAVAILABLE** unless an explicit event threshold *and* a real reference exist |
| Peak intensity, peak preservation ratio, wet-area ratio, structure correlation | **MEASURED** only where the frames support them |

### Ground-truth detection skill

**UNAVAILABLE.** No verified ground-truth warning catalogue exists for this
project. POD/FAR/CSI against an invented event catalogue would be a number with
no meaning, so they are not computed - and are not reported as zero.

## 9. Uncertainty method

**Validation-residual bootstrap** (`evaluation/uncertainty.py`).

Chosen over a seeded ensemble because an ensemble needs several training runs per
evaluation, which would make the suite slow and machine-dependent. The
residual bootstrap needs one trained model and the validation predictions it
already produces.

* Residuals are `truth - prediction` on the **validation** split.
* Empirical quantiles of the residuals give the interval, which is **added to**
  the point prediction so it lives in the same space.
* The bootstrap is seeded, so the same seed gives the same intervals.
* Intervals need at least 20 validation residuals; below that, the quantiles are
  dominated by individual samples and **no interval is reported**.

### What this method is not

It is **not a calibrated probability** and **not a Bayesian posterior**. It is an
empirical error distribution, valid only if validation and test residuals are
exchangeable - an assumption this project **cannot verify**, because verifying it
would require using the test split. The word "confidence" is avoided
throughout.

## 10. Calibration

`evaluate_coverage` measures whether the nominal interval actually covers the
test residuals. The test split is read **once**, after the width is fixed from
validation; nothing feeds back, so this is a measurement and not a tuning step.

| Nominal | Reported as |
| --- | --- |
| 0.80 | Empirical coverage per output, interval width, sample count |
| 0.90 | Same |

Below 30 test residuals, coverage is **`INSUFFICIENT_DATA`** - a fraction over a
sample that small is sampling noise, and reporting one would be dishonest.

`is_calibrated` is a **measurement**, not an assertion: it returns whether the
numbers say the interval is honest, and it does not license any stronger claim.

## 11. Explainability method

Evidence is assembled per event from what the pipeline recorded and what models
consumed. Every item is typed:

| Type | Meaning |
| --- | --- |
| `supporting` | A model consumed it as input. |
| `context` | A recorded fact for a human reader; no model consumed it. |
| `unavailable` | The evidence was sought and does not exist. |

**No causal claim is made anywhere.** A fixed, non-optional disclaimer is
attached to every explanation, and the test suite searches every generated
string for causal phrasing.

Feature sensitivity (`sensitivity_report`) perturbs one input at a time and
reports the change in the model's output. It is labelled **input sensitivity**
and is explicitly *not* causal importance: a model can be sensitive to an
irrelevant feature and insensitive to a relevant one.

## 12. Known failure modes

| Failure mode | Mitigation |
| --- | --- |
| An unavailable model is treated as a good or bad result | `NOT_AVAILABLE` with a reason; no inference from the baselines |
| A rejected/NaN prediction is scored as low error | Non-finite model output removes the model from the comparison entirely |
| Interval interpreted as a probability | Fixed caveat on every interval; "confidence" never used |
| A horizon with no data silently omitted | Reported `INSUFFICIENT_DATA` |
| One-step and rollout errors averaged together | Never pooled; reported separately, with the reason pooling would mislead |
| A missing previous step makes constant-velocity look perfect | Reported `INSUFFICIENT_DATA`, explicitly "rather than being reported as zero error" |
| A "not a model input" perturbation reads as "does not matter" | Marked `perturbed: false` with the reason |
| A stale run's `run_id` collision | Derived from dataset, seed, checkpoint and parts |

## 13. Data limitations

* **No ground truth.** The single largest limitation. Nothing here is an
  accuracy or skill measurement.
* **The models are synthetic-trained.** Evaluated on reanalysis, they measure
  transfer, not fit.
* **ERA5 is reanalysis**, not observations and not a forecast.
* **The GNN produces one-step extrapolations** whose error compounds in any
  rollout; a 3-step number is a 3-model-error number.
* **Alert risk scores are rule-based triage aids**, not validated indices.
* **Anomaly scores are on the detector's own scale**, not comparable across
  detectors or versions.
* **No percentile or standardized-anomaly evidence is reported**, because no
  climatological reference period long enough to compute one exists.

## 14. Reproducibility

Every run records:

| Field | Purpose |
| --- | --- |
| `seed` | Reproduces the bootstrap |
| `dataset_id` + `dataset_checksum` | Identifies the data |
| `split_boundaries` | Where each block starts and ends |
| `model_checkpoints` | Which checkpoint produced which number |
| `code_fingerprint` | SHA-256 over the evaluation source files, so a result is traceable to its code |
| `configuration` | Every parameter, including the train-only scaler constants |
| `environment` | Platform facts |
| `package_version` | Version identifier |
| `run_id` | Deterministic hash of dataset, seed, checkpoint and parts |

**Verified:** running the same evaluation twice with the same inputs produces
**byte-identical** records, apart from the wall-clock timestamps. This is
asserted by a test, not claimed.

## 15. Non-operational disclaimer

**This is a research prototype built for SIH26078.** It is not an operational
emergency-warning system. It has no demonstrated forecast skill, no verified
accuracy claims, no availability guarantee, and no skill score, because no
verified ground-truth warning catalogue exists. The alert severities are project
prototype categories, NOT official IMD or WMO warnings. Nothing here may be used
for emergency decisions.
