# AI Downscaling Prototype (SIH26078, Phase 4)

> **Research prototype.** Trained on synthetic data against a **synthetic**
> high-resolution reference. Not an operational forecast, not validated against
> observations, and not suitable for any emergency decision.
>
> **Higher resolution does not imply higher accuracy.** The metrics below measure
> agreement with a generated target. Section 12 reports the measured comparison
> honestly, including where the AI loses to a baseline.

---

## 1. The downscaling problem

Global NWP models run on grids of roughly 12 km. Many hazards - orographic
rainfall, urban flooding, convective cells - vary on scales far below that, so a
12 km field is smooth where reality is not. **Downscaling** refines a coarse
field onto a finer grid.

Two ways to do it:

* **Interpolation** (Phase 0, preserved unchanged): nearest-neighbour and cubic
  resampling. They produce a fine grid that contains **no new information** - they
  only interpolate between the coarse points.
* **Learning** (Phase 4, this phase): infer the sub-coarse-cell structure that the
  coarse field is consistent with, using patterns learned from data.

The honest framing: interpolation cannot invent structure, and a learned model can
only invent the structure it has seen before. Neither creates information that was
not in the input.

## 2. Why 12 km → 5 km in this project

The project generates a synthetic NWP-style field at **0.125 deg** and the
configured target is **0.05 deg**. Nothing is hard-coded: the resolutions are
**detected from the coordinate arrays** by `detect_resolution()` (median absolute
spacing), and every reported number comes from that. Changing the generator changes
the reported resolutions automatically.

| Quantity | Value (this dataset) |
| --- | --- |
| Input resolution | 0.125 deg (~13.9 km) |
| Output resolution | 0.05 deg (~5.6 km) |
| Scale factor | 2.5 |
| Input grid | 49 x 49 |
| Output grid | 121 x 121 |

## 3. The existing interpolation baseline (preserved)

Phase 0 shipped `nearest` and `cubic` behind a registry
(`extreme_weather.downscaling.registry`). **Neither was modified.** The learned
method is *added* to the same registry, so the public call is unchanged:

```python
from extreme_weather.downscaling import available_methods, downscale

available_methods()                       # ('ai', 'cubic', 'nearest')
downscale(field, 0.05)                    # nearest
downscale(field, 0.05, method="cubic")     # cubic
downscale(field, 0.05, method="ai")        # Phase 4, loads models/downscaling/best.pt
```

## 4. The AI approach

```text
coarse field (12 km)
    ↓  synthetic high-resolution reference (5 km)   ← training target
    ↓  10 documented input features
    ↓  per-channel normalisation (training block only)
    ↓  feature projection → 3 residual conv blocks → 1x1 output conv
    ↓  ReLU residual, bilinearly upsampled to the target grid
    ↓  AI_output = bilinear baseline + residual
high-resolution field (5 km)
```

**Residual learning.** The network predicts a *correction* to a bilinear
upsample, not the field. The bilinear baseline already satisfies the coarse-scale
constraint, so the network only has to learn the sub-cell part - a small,
zero-centred quantity that is far easier to fit. It also bounds the damage: a
network that learns nothing returns the baseline exactly.

**Non-negativity.** Precipitation cannot be negative, so the residual passes
through **ReLU** and the output is `baseline + ReLU(residual) >= baseline`. This is
a documented design constraint, not a silent clip, and it is enforced by tests.
The cost is stated plainly: the network cannot learn to *suppress* the baseline
where the true field is lower. A sigmoid-gated residual would allow that and is
future work.

## 5. The synthetic high-resolution target

A learned model needs a target at the fine resolution. This project has no
observational archive, so the target is **generated**:

```text
1. background  = bilinear upsample of the coarse field      (coarse-scale signal)
2. structure   = two octaves of smooth noise, scaled by |background|
3. correction  = minimum-norm solution of  A·C = (coarse - A·candidate)
                so that block_average(reference) == coarse EXACTLY
```

Step 3 is what makes the task well-posed. The overlap matrices `W` define the
area-average operator `A = W @ fine`; the correction is
`C = Wᵀ(WWᵀ)⁻¹ error (WWᵀ)⁻¹Wᵀ`, exact to floating point (measured residual
**1.8e-15**).

> **The target is SYNTHETIC. It is not observational truth, not satellite data,
> not reanalysis.** It exists so a model has *something* with known fine-scale
> structure to be measured against. Every metric in this phase describes agreement
> with this generated target and nothing else.

## 6. Input features (10 channels)

| # | Feature | Unit | Purpose |
| --- | --- | --- | --- |
| 0 | `coarse_precipitation` | scaled | the primary signal |
| 1 | `latitude_scaled` | 1 | latitude / 90, in [-1, 1] |
| 2 | `longitude_scaled` | 1 | longitude / 180, in [-1, 1] |
| 3-6 | `north/south/east/west_neighbour` | scaled | the local trend |
| 7 | `latitude_gradient` | scaled | north - south |
| 8 | `longitude_gradient` | scaled | east - west |
| 9 | `coarse_std` | scaled | 3x3 local standard deviation |

Coordinates let the network learn that precipitation is not spatially uniform;
without them a convolution is translation-invariant and cannot. The neighbour and
gradient channels distinguish a cell on the edge of a storm from one in its core -
the most useful cue for where the fine field should bulge. The local standard
deviation flags how much fine structure a neighbourhood could hide.

Neighbour channels use edge replication, so nothing is NaN at the boundary. **No
feature reads the high-resolution target**, which is verified by a test.

## 7. Target

```text
target = max(reference - bilinear_baseline, 0)
```

The residual, matching what the network predicts. Both terms are non-negative by
construction, consistent with the ReLU output.

## 8. Model architecture

```text
input [B, 10, H, W]   (coarse grid, H=W=49 here)
  → 1x1 conv projection to 32 channels, ReLU
  → 3 x residual block: 3x3 conv, ReLU, 3x3 conv, + skip
  → 1x1 conv to 1 channel
  → ReLU                       ← the non-negativity constraint
  → bilinear upsample to (H_t, W_t)
  → + bilinear baseline
output [B, 1, 121, 121]
```

**55,873 parameters.** Runs on CPU in seconds. PyTorch is already a project
dependency. **No** Lightning, PyTorch Geometric, diffusion or GAN framework is
used, and **no dependency was added in this phase**.

The scale factor is not assumed: the network runs on the coarse grid and upsamples
in the forward pass, so any target shape works without retraining (tested with
several target sizes).

## 9. Normalization

Per-channel mean and standard deviation, computed **on the training block only**
and stored in the checkpoint.

This is the single most important anti-leakage rule in the project. Fitting on the
whole dataset would let test-set statistics inform training and bias every reported
number. `test_normalization_differs_when_the_test_block_is_included` pins the
behaviour: including the test block measurably changes the statistics.

## 10. Chronological split

```text
  earliest ................................. latest
  |---------------- TRAIN ----------------|
                                |--- VALIDATION ---|
                                              |-- TEST --|
```

60/20/20 by **time**, never shuffled. Adjacent weather frames are nearly
identical, so a shuffled split would leak the answer into training and produce
excellent-looking meaningless numbers. Tests assert
`max(train) <= min(validation) <= min(test)` in both index and timestamp order,
that the blocks partition the samples without overlap, and that the feature
statistics come from the training block alone.

## 11. Loss

```text
loss = MSE(predicted_residual, target_residual)
```

MSE because the target is a residual: small, zero-centred and unbounded, so a
squared error is well behaved and needs no weighting. MAE and RMSE are both
**reported** so the two are visible, but the optimised loss stays the single
explainable MSE. No gradient-L1, adversarial or spatial-gradient term was added -
none is justified by an experiment here, and inventing one to move a metric would
be dishonest.

## 12. Measured baseline comparison

All three methods scored on the **same** test frames, against the synthetic
reference. Values as produced by `POST /api/v1/downscaling/evaluate`:

| Method | MAE | RMSE | Bias | Corr. | POD | FAR |
| --- | --- | --- | --- | --- | --- | --- |
| nearest | 0.14363 | 0.29655 | -0.00864 | 0.9236 | 0.9215 | 0.1170 |
| cubic | 0.11826 | 0.25179 | -0.01131 | 0.9467 | 0.9678 | 0.0978 |
| **ai** | 0.12598 | 0.25365 | **+0.05959** | **0.9491** | **0.9917** | 0.1621 |

**Read this honestly:**

* The AI **beats `nearest`** on MAE, RMSE and correlation.
* The AI **loses to `cubic`** on MAE and RMSE - it does not win the point-error
  comparison.
* The AI has the **highest correlation** (0.9491) and the **highest POD** (0.9917),
  but also the **highest FAR** (0.1621) and by far the **largest |bias|**
  (0.0596 vs 0.0086 and 0.0113). It detects the event better and pays for it
  with over-prediction and false alarms. POD alone must never be read alone.

**The claim this supports:** on this synthetic dataset the learned residual sits
between the two baselines on point error, while being less biased and more
sensitive to the event. **The claim this does not support:** that AI downscaling
beats cubic interpolation. It does not, on these numbers.

Where it *is* ahead is event-specific: peak preservation, event-centre
displacement and structure correlation, at the cost of a positive bias and more
false alarms.

Why it does not win on point error: with 72 frames and 43 for training, against a
structured synthetic target, a smooth interpolant is a hard baseline to beat. The
model is small and trained for seconds. This is reported as measured; no test-set
tuning was done to change the outcome.

## 13. Extreme-event evaluation

The project injects a **known** event, so its peak, centre and area can be
compared against the reference:

| Method | Peak | Ref. peak | Peak ratio | Centre shift | Area ratio | Struct. corr. |
| --- | --- | --- | --- | --- | --- | --- |
| nearest | 8.712 | 13.168 | 0.662 | 1.487 km | 1.048 | 0.8932 |
| cubic | 9.093 | 13.168 | 0.691 | 0.670 km | 1.083 | 0.9184 |
| **ai** | 10.617 | 13.168 | **0.806** | **0.476 km** | 1.198 | **0.9249** |

* The AI preserves the **peak best** (ratio 0.806, closest to 1.0).
* Both baselines **severely damp the peak** (0.66-0.69 of the reference 13.17
  mm h-1) - the classic behaviour of interpolation on a sharp maximum.
* The AI has the **smallest centre displacement** (0.476 km vs 0.670 and 1.487 km)
  and the **best structure correlation** (0.9249). This is a measured distance on
  a synthetic field; **no claim of exact localisation is made**, and it would be
  meaningless against real storms.
* The AI **over-estimates the wet area** (ratio 1.198 vs 1.048 and 1.083), which
  is the same over-prediction its positive bias shows.

## 14. Checkpointing

`models/downscaling/best.pt` (PyTorch format - the weights are convolutional
tensors, unlike the Phase 3 GNN's small floats). It stores the model state, the
model configuration, the feature names, the input and output resolutions, the grid
shapes, the variable, the normalisation statistics, the training seed, the
training metadata, the best validation loss and the prototype disclaimer.

`load_checkpoint` validates everything and raises `DownscalingCheckpointError` for
a missing file, an unreadable file, a missing key, a schema-version mismatch, a
**feature-contract mismatch**, or parameters that do not match the architecture.
Loading uses `weights_only=True`.


## 15. API

| Endpoint | Purpose | Codes |
| --- | --- | --- |
| `GET /api/v1/downscaling/status` | availability, version, resolutions | 200 always |
| `GET /api/v1/downscaling/model` | architecture, features, normalisation | 200 always |
| `POST /api/v1/downscaling/run` | downscale one frame | 200 / 404 / 422 / 503 |
| `POST /api/v1/downscaling/evaluate` | compare methods on the test set | 200 / 422 / 503 |

`/status` and `/model` never fail: a missing model is `available: false` with an
explanatory message, so a dashboard renders "not trained yet". **No endpoint
returns a raw array** - a downscaled field is tens of thousands of values, so
responses carry metadata, statistics and file references.

## 16. Frontend

`DownscalingPanel` shows the model status, the detected input/output resolutions
with grid sizes, a method selector (nearest / cubic / AI), the measured
MAE / RMSE / Bias / Correlation / POD / FAR table, and the extreme-event table.
It deliberately shows **no accuracy percentage** and states that higher resolution
does not imply higher accuracy.

## 17. Limitations

1. **The target is synthetic.** Not observational truth, not satellite, not
   reanalysis. Every metric describes agreement with a generated field.
2. **The AI does not beat `cubic`** on point error (section 12). It beats
   `nearest`, has the highest correlation, detects the event best and preserves
   its peak and centre best - but it over-predicts (bias +0.060) and raises FAR
   to 0.162. That is the whole claim.
3. **Higher resolution is not accuracy.** All three methods produce a finer grid;
   only the reference comparison says anything about correctness.
4. **One variable.** Only precipitation. Temperature and wind are not modelled.
5. **No multi-step rollout, no ensemble, no uncertainty estimate**, no coupling
   between variables.
6. **Monotone residual.** ReLU means the model cannot suppress the baseline where
   the true field is lower - a known limitation of this design.
7. **Tiny training set** - 43 frames - and a single synthetic domain. Nothing here
   transfers to another region or season.
8. **Fixed normalisation**; no per-region adaptation.
9. **Static, per-run graph**; no streaming, no online updating.
10. **Unvalidated.** No comparison against observed storms, no independent
    verification data, no benchmark against published downscaling methods.
11. **Diffusion and GAN super-resolution are future work**, as is real NWP
    ingestion.

## 18. Synthetic-data disclaimer

The coarse field, the high-resolution reference, the features, the targets, the
losses and every metric above come from **synthetic data** produced by
`scripts/generate_sample_data.py` with a fixed seed.

Nothing here describes real weather, and nothing here has been compared with a
real storm. **This is a research prototype, not an operational forecast. It must
not be used to issue a warning or guide emergency decisions.** The same notice is
printed by the CLI, stored in every checkpoint, returned in every API response and
rendered in the dashboard.

Real deployment would require observational or NWP high-resolution truth, a real
climatology, diverse extreme-weather cases, and validation across unseen weather
regimes. None of that exists here.


