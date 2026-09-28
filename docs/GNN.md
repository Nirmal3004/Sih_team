# GNN Event Evolution Prototype (SIH26078, Phase 3)

> **Research prototype — not an operational forecast.** The model is trained on
> *synthetic* data produced by this project's deterministic generator. It has not
> been validated against observed weather, has no skill score, and must not be
> used for any emergency decision. The live demo produces a physically
> implausible displacement (see §16); that is stated plainly rather than hidden.

---

## 1. Why a GNN

Phase 2 produced trajectories, but nothing that *explains* or *extrapolates*
them. A storm's next position depends on more than its own last two points: a
weaker cell nearby may merge into it, a stronger one may suppress it, and the
diurnal cycle matters. Modelling that with a per-track time series throws the
neighbourhood away.

A graph makes the neighbourhood explicit: each event state is a node, relations
are edges, and a message-passing layer lets a node read its surroundings. The
graph formulation also matches the physics of the problem - a trajectory is a
path through a field of interacting systems - and it scales to heterogeneous
inputs (grid cells, events, tracks) without restructuring.

**What this prototype is not:** it is not the full spatio-temporal GNN planned
for a later phase, and it does not replace the Phase 2 rule-based tracker. It
predicts *evolution* of an already-tracked event; the *linking* is still Phase 2's
deterministic cost function.

## 2. Graph definition

The input is `data/processed/tracks.json` (Phase 2 output). Construction is a
pure function of those tracks: nodes are ordered by `(start_time, track_id)` and
every candidate pair is evaluated in that fixed order. No randomness, no hashing,
no dependence on input order - rebuilding from the same tracks yields identical
arrays.

## 3. Node definition

**One node = one track point = one event state at one valid time.**

For a track of *n* points the graph gets *n* nodes. The last point of each track
carries no target, because its future is genuinely unknown; the other *n - 1*
carry the delta to their own next point.

## 4. Node features (16 values)

| # | Feature | Unit | Meaning |
| --- | --- | --- | --- |
| 0 | `latitude` | deg | Centroid latitude of the event. |
| 1 | `longitude` | deg | Centroid longitude. |
| 2 | `latitude_scaled` | 1 | `latitude / 90` - keeps geography in `[-1, 1]`. |
| 3 | `longitude_scaled` | 1 | `longitude / 180`. |
| 4 | `intensity_scaled` | 1 | `intensity / 100` (mm h-1, m s-1 or K). |
| 5 | `anomaly_score` | 1 | Phase 1 anomaly score of the detection. |
| 6 | `speed_scaled` | 1 | `speed_kmh / 120` (Phase 2 speed). |
| 7 | `bearing_sin` | 1 | `sin(bearing)`; undefined bearing -> 0. |
| 8 | `bearing_cos` | 1 | `cos(bearing)`; undefined bearing -> 1. |
| 9 | `is_stationary` | 1 | 1.0 when the event did not move (`CALM`). |
| 10 | `cell_count_scaled` | 1 | `log1p(cell_count) / log1p(100)`, capped to `[0, 1]`; missing -> 0. |
| 11 | `hour_sin` | 1 | `sin(2*pi*hour/24)`. |
| 12 | `hour_cos` | 1 | `cos(2*pi*hour/24)`. |
| 13 | `is_extreme_rainfall` | 1 | Event type one-hot (3 columns). |
| 14 | `is_extreme_wind` | 1 | " |
| 15 | `is_extreme_temp` | 1 | " |

Two encoding choices matter:

* **`sin`/`cos` instead of raw angles.** 359 deg and 1 deg are neighbours, not
  opposite ends of a range. The same argument applies to hour-of-day across
  midnight. Both pairs lie exactly on the unit circle (tested).
* **Scaled copies of coordinates and speed.** Raw degrees (77 deg) and mm h-1
  would otherwise dominate the first layer's activations. The scaled columns are
  O(1); the raw ones are retained because they are interpretable in the
  checkpoint.

An unknown event type yields all-zero one-hot columns - it is represented, never
silently mapped onto a real type.

**No feature reads the future.** Every value is computable from the node's own
timestamp and its incoming edge. The next point's position and intensity appear
only as the *label*.

## 5. Edge definition

Two kinds, both **directed**:

**Temporal edges (always built).** `point[i] -> point[i+1]` inside a track: the
real trajectory step. A track of *n* points contributes *n - 1*.

**Spatial edges (optional, configurable).** Two points sharing a **timestamp**
but belonging to **different** tracks, joined when their great-circle distance
is at or below `max_spatial_distance_km` (default 150 km). This expresses "two
events were near each other at this hour" - a storm approaching a weaker one.
Arbitrarily distant events are never connected, and the threshold is validated to
be finite and non-negative.

Three guarantees, all tested: **no self-loops**, **no duplicate edges**, and
**no spatial edge inside a track** (that pair is already temporal; a second edge
would double-count the step in the aggregation).

## 6. Edge features (6 values)

| # | Feature | Unit | Meaning |
| --- | --- | --- | --- |
| 0 | `distance_scaled` | 1 | `distance_km / 250`. |
| 1 | `bearing_sin` | 1 | `sin` of the source->target bearing. |
| 2 | `bearing_cos` | 1 | `cos` of the source->target bearing. |
| 3 | `time_delta_scaled` | 1 | `hours / 6`. |
| 4 | `is_temporal` | 1 | 1.0 within a track, else 0.0. |
| 5 | `is_spatial` | 1 | 0.0 within a track, 1.0 otherwise. |

Only the **source** state and the pair geometry are used; the destination's own
features arrive through message passing. The two indicator columns let the model
distinguish the edge kinds, which matters because a temporal edge is a real
trajectory step while a spatial edge only means proximity.

## 7. Message passing

For node `i` with neighbours `j`:

```text
message  m_ij = W_msg . concat(h_j, e_ij)          # neighbour state + edge
aggregate      = mean over incoming edges of m_ij  # isolated nodes: skipped
update    h_i' = ReLU( W_self . h_i + aggregate )
```

This is the message-passing scheme of Gilmer et al. (2017) in its simplest form.

* **Why mean, not sum.** A node's in-degree varies - one step, or several
  concurrent neighbours. A sum would make the magnitude of the prediction depend
  on how crowded the region is; a mean does not. The test
  `test_aggregation_is_a_mean_not_a_sum` pins this: duplicating a message must not
  change the output.
* **Isolated nodes** keep their own representation (`update(h_i)`) instead of
  receiving the average of nothing, which would be zero and would silently
  weaken them.
* **Self-transform.** The self term means a node retains its own state rather than
  being overwritten by its neighbourhood.
* **One scatter per layer.** All edges are processed in a single
  `index_add_`, so a layer is O(edges) with no Python loop.

There is no attention, no edge weighting and no recurrence - deliberately, to
keep the model inspectable and CPU-cheap.

## 8. Prediction target

The head regresses **changes, not absolute coordinates**:

```text
predicted  [ delta_latitude, delta_longitude, delta_intensity ]
decoded    next_latitude  = current_latitude  + delta_latitude
           next_longitude = current_longitude + delta_longitude
           next_intensity = current_intensity + delta_intensity
```

**Why deltas.** An absolute latitude near 13 deg would dominate a squared error:
being wrong by 0.01 deg is 13 units of error against a 169-unit mean, so the
optimisation would spend its capacity learning the offset instead of the motion.
A delta is small and zero-centred, so the loss measures exactly the thing being
predicted.

**The cost of that choice.** Deltas are not scale-free either, and the model can
still emit an absurd one. Decoding therefore **clamps** the result to
`[-90, 90] x [-180, 180]`; a latitude of 213 deg is a bug, not a forecast. The
raw delta is reported unchanged alongside it so the caller always sees what the
model actually produced.

## 9. Normalization

Fixed divisors, **not statistics fitted on the training data**:
intensity / 100, speed / 120, cell count through `log1p`, edge distance / 250 km,
edge time / 6 h.

The trade-off is deliberate. Z-scoring would be more flexible, but it would make
a checkpoint unreadable without the training set and would let the input
distribution shift silently at inference. Fixed constants are reproducible,
inspectable and stored in the checkpoint, so inference transforms inputs exactly
as training did.

## 10. Chronological split

Weather events evolve forward in time. Shuffled samples would let the model train
on a storm's *later* states and be scored on its *earlier* ones - temporal
leakage, producing an optimistic number that means nothing for forecasting and
that no metric would reveal.

Samples are therefore **never shuffled for splitting**. Nodes with a target are
ordered by the timestamp of the **current** node and cut into three contiguous
blocks at fractional points of the observed time range:

```text
  earliest .................................................. latest
  |------------------ TRAIN -----------------|
                                 |--- VALIDATION ---|
                                               |-- TEST --|
```

Default 60 / 20 / 20, cut by *time* rather than sample count so each block is a
distinct period. The invariant the tests assert is
`max(train) <= min(validation) <= min(test)`, that the blocks partition the
samples without overlap, and that no training sample shares a timestamp with the
test block. Fractions must be non-negative and sum to 1, with a non-empty train
and test.

The training *shuffle* inside an epoch is a different thing and is seeded
(`torch.randperm` with an explicit generator), so it is reproducible.

## 11. Training

```text
data/processed/tracks.json
   -> build_dataset  (graph + chronological split)
   -> training.train (train block only; best on validation)
   -> models/gnn/best.json
```

* **Full-batch** gradient step per epoch. The graph is small (tens of nodes);
  mini-batching would add machinery with no benefit.
* **Adam**, learning rate 0.01, 400 epochs, seed 20260601 by default.
* **Validation after every epoch**; the parameters with the lowest validation
  loss are snapshotted and restored at the end.
* **The test block is never used** to update or select anything. It is evaluated
  once at the end as a diagnostic. A test asserts that passing the test block to
  `train()` does not change a single weight.
* A run with no validation block has nothing to select on and keeps the final
  weights, which is stated rather than silently "best"ing something.

## 12. Loss function

```text
position_error  = MSE(delta_lat_pred, delta_lat_true) + MSE(delta_lon_pred, delta_lon_true)
intensity_error = MSE(delta_intensity_pred, delta_intensity_true)

loss = position_weight * position_error + intensity_weight * intensity_error
```

The two position terms are **summed** (not averaged) so the term is the squared
Euclidean displacement error in degrees - the quantity of physical interest.

Position and intensity are weighted **separately** because they are not
comparable numbers: a millidegree of latitude and one unit of mm h-1 would
otherwise let whichever term is numerically larger dictate the optimisation.
Defaults are position 1.0 and intensity 0.5, expressing that *where* a storm goes
matters more than how strong it is for trajectory extrapolation. Both are
configurable through `ModelConfig` and the CLI.

## 13. Checkpointing

`models/gnn/best.json` - a single JSON document:

| Key | Content |
| --- | --- |
| `schema_version` | Checkpoint layout version. |
| `model_version` | Architecture version (`gnn-prototype-1.0`). |
| `model_config` | Every hyper-parameter, including the seed. |
| `node_feature_names` / `edge_feature_names` / `target_names` | The feature contract. |
| `feature_config` | The scaling constants. |
| `training` | Seed, epochs, learning rate, sample counts, losses, best epoch. |
| `parameters` | Flat `{name: [floats]}` of the model state. |
| `disclaimer` | The prototype notice, stored with the artefact. |

**Why JSON, not `torch.save`:** the checkpoint stays human-inspectable, diffable
and loadable without a torch deserialisation (which is not a safe format to read
from an untrusted file). The weights are a few thousand floats.

**Integrity.** `load_checkpoint` validates before returning and raises
`CheckpointError` for: a missing file, invalid JSON, a non-object document, a
missing key, a `schema_version` mismatch, a **feature-contract mismatch** (an old
checkpoint can never be silently loaded into changed features), an empty
parameter set, a wrong parameter count, and parameters that do not match the
architecture.

Because JSON flattens tensors, the loader restores the original shapes from the
reference model and treats a length mismatch as an error rather than reshaping
silently. Inference fails clearly when the checkpoint does not exist.

## 14. Inference

`GNNService.predict(track_id)` rebuilds the graph from the *stored* tracks and
runs the model over it, so the node inherits its real neighbours. Only the **last**
point of a track is predicted: earlier points already have observations, so
predicting them would score the model against data it was shown.

Errors are distinct and catchable, each mapping to one HTTP status:

| Exception | Cause | HTTP |
| --- | --- | --- |
| `GNNModelUnavailableError` | no valid checkpoint | 503 |
| `GNNUnknownTrackError` | track id not in `tracks.json` | 404 |
| `GNNInvalidTrackError` | no usable tracking file / invalid input | 422 |
| `GNNInferenceError` | model ran but produced nothing usable | 500 |

Nothing is fabricated: without a checkpoint or a track, the answer is an error.

## 15. API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/gnn/status` | availability, version, losses, training metadata |
| `GET /api/v1/gnn/model` | architecture, features, targets, hyper-parameters |
| `POST /api/v1/gnn/predict` | prototype next-state prediction for one track |

`/status` **never fails**: a missing model is `available: false` with an
explanatory message, so a dashboard can render "not trained yet" rather than an
error. `/model` also works untrained, flagging `available: false`. Every response
carries the prototype `notice` field, so a consumer cannot render a prediction
without also being handed its status.

```bash
python scripts/build_graph_dataset.py
python scripts/train_gnn.py --epochs 500 --learning-rate 0.02
python scripts/run_gnn_inference.py --track-id TRK-EXTREME_RAINFALL-20240602T0200-001
python scripts/run_gnn_inference.py --list-tracks
```

## 16. Limitations

1. **The demo prediction is physically implausible.** Trained on the available
   data (1 track, 7 samples, 4 for training), the model outputs
   `delta_lat = +0.97 deg` in one hour - about 109 km/h northward, faster than
   most tropical cyclones. Train loss reaches 0.0014 while validation is 5.88 and
   test 64.56: a textbook overfit on four samples. This is reported, not hidden,
   and it is the honest result of a prototype on synthetic data.
2. **No skill score.** The reported losses are a regression error on synthetic
   trajectories. They are not CRPS, not RMSE against observations, and not a
   forecasting verification of any kind. No accuracy percentage is shown anywhere
   in the API or the dashboard, because none exists.
3. **Tiny training set.** The pipeline produces one track. Meaningful training
   needs a multi-year, multi-storm archive.
4. **Single-step, one variable.** One next state, one variable per track, no
   multi-step rollout and no ensemble or uncertainty estimate.
5. **Delta targets are not scale-free.** A model can emit an absurd displacement;
   decoding clamps the position, which prevents an invalid coordinate but does
   **not** make the value plausible.
6. **Fixed normalisation.** Not adapted to a new region or variable, which limits
   transfer.
7. **Static graph.** Built once per run from a file; no streaming, no online
   updating, no re-linking.
8. **Phase 2 linking is unchanged.** The graph refines nothing about how events
   were matched; a full GNN *tracker* (learned association) remains future work.
9. **Not validated.** No comparison against observed storms, no independent test
   data, no verification of any kind.
10. **Prototype-scale model.** ~6.3k parameters, 2 layers, mean aggregation, no
    attention - chosen for CPU inspectability, not capability.

## 17. Synthetic-data disclaimer

The graph, the features, the targets, the losses and the prediction above are
all derived from **synthetic data** generated by `scripts/generate_sample_data.py`
with a fixed seed. A deterministic generator produced that storm, and a
four-sample model learned to reproduce it.

Nothing here describes real weather, and nothing here has been compared to a real
storm. **This is a research prototype, not an operational forecast, and it must
not be used to issue a warning or guide emergency decisions.** The same notice is
printed by the CLI, stored in every checkpoint, returned in every API response and
rendered in the dashboard.
