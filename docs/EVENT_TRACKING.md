# Extreme Weather Event Tracking (SIH26078, Phase 2)

> **Prototype.** Tracking links *synthetic* detections produced by the Phase 1
> anomaly detector. It is not an operational forecast, not a real-time warning
> product, and it is not verified against observed weather.

---

## 1. The tracking problem

Phase 1 detects events **per time step**: at 02:00 an `EXTREME_RAINFALL` blob sits
at 13.00 N, 77.00 E, and at 03:00 another blob is detected at 13.00 N, 76.99 E.
Nothing in the Phase 1 output says these are *the same storm*.

Tracking answers that question and, once the answer is known, measures the
motion:

```text
Phase 1 events (per step)
        ↓  sort chronologically, group by valid time
        ↓  match step t to step t+1   (distance, type, gap, intensity)
        ↓  extend open tracks / open new ones
EventTrack: trajectory, speed, bearing, duration, lifecycle
```

The tracker is deliberately **rule-based and explainable**. A GNN tracker is
**FUTURE (Phase 3)** and will refine the linking decision made here.

## 2. Event matching

For a previous event `a` at time `t` and a candidate `b` at `t' > t`:

```text
distance_km = haversine(centroid_a, centroid_b)
gap_hours   = t' - t                      (must be strictly forward)
speed_kmh   = distance_km / gap_hours

cost = ( distance_weight * (distance_km / max_distance_km)
       + intensity_weight * intensity_mismatch(a, b) ) / (distance_weight + intensity_weight)

intensity_mismatch(a, b) = 1 - min(I_a, I_b) / max(I_a, I_b)      [0, 1]
                          = 0 when either intensity is <= 0
```

A pair is **rejected** (cost = infinity) when:

1. the event types differ;
2. `gap_hours` is `<= 0` or `> max_time_gap_hours`;
3. `distance_km` is `> max_distance_km`;
4. `speed_kmh` is `> max_speed_kmh` - a physically impossible jump.

Assignment is **greedy over increasing cost**, with ties broken by
`(cost, previous_event_id, next_event_id)`. It is strictly **one-to-one**: a
detection can extend at most one track per step, and a track can be extended by
at most one detection per step. No randomness, no hashing, no dependence on input
order. Greedy is not a global optimum - stated explicitly so the trade-off is
visible; for the one-to-one assignment problem it is a standard, reproducible
approximation.

## 3. Matching configuration

`MatchingConfig` (Pydantic, frozen, validated) holds every knob:

| Parameter | Default | Meaning |
| --- | --- | --- |
| `max_distance_km` | 250.0 | reject pairs further apart than this |
| `max_speed_kmh` | 120.0 | reject physically impossible movement |
| `max_time_gap_hours` | 6.0 | reject pairs separated by more time |
| `distance_weight` | 0.7 | weight of the normalised distance term |
| `intensity_weight` | 0.3 | weight of the intensity-similarity term |

Override from the CLI (`--max-distance-km`, `--max-speed-kmh`,
`--max-time-gap-hours`, `--distance-weight`, `--intensity-weight`) or in code
(`TrackingService(config=MatchingConfig(...))`). The active configuration is
written into `tracks.json` under `metadata.matching`, so any result can be
reproduced.

## 4. Distance calculation

Haversine great-circle distance on a sphere of radius **6371.0088 km**:

```text
a = sin²(Δφ/2) + cos φ1 · cos φ2 · sin²(Δλ/2)
d = 2R · asin(√a)
```

One degree of latitude is 111.19 km; longitude degrees shrink with `cos φ`.
Non-finite input returns `0.0` instead of raising.

## 5. Speed calculation

```text
speed_kmh = distance_km / gap_hours
```

`movement_speed_kmh` returns `0.0` for a missing, zero, negative or non-finite
duration - it never divides by zero. A stationary event therefore reports
`speed = 0`, not `NaN` and not an exception.

## 6. Bearing

```text
y = sin Δλ · cos φ2
x = cos φ1 · sin φ2 - sin φ1 · cos φ2 · cos Δλ
bearing = (degrees(atan2(y, x)) + 360) mod 360
```

Degrees clockwise from **true north**: 0 = N, 90 = E, 180 = S, 270 = W. For
coincident points the direction is mathematically undefined, so the function
returns `None` instead of an arbitrary value.

## 7. Compass direction

A 16-point rose (`N, NNE, NE, ENE, E, ESE, SE, SSE, S, SSW, SW, WSW, W, WNW, NW,
NNW`) with 22.5° sectors centred on each label. A missing/undefined bearing maps
to **`CALM`**, which is also what the first point of every track reports (there is
no previous position yet).

## 8. Intensity evolution

```text
intensity_trend = STRENGTHENING | WEAKENING | STABLE
relative_change = (I_last - I_first) / max(|I_first|, |I_last|)
```

`STABLE` when `|relative_change| <= 0.05` (5 %), or when an intensity is missing
or non-finite. This measures the *physical* field (mm/h, m/s, K) of the track, not
the anomaly score.

## 9. Track lifecycle

```text
step t1  event A            → new track (single point)
step t2  event A' matched   → track grows
step t3  event A'' matched  → track grows
...                          track keeps growing
```

The status is decided **when the run finishes**, which makes it deterministic:

| Status | Rule |
| --- | --- |
| `NEW` | exactly one point, and that point is in the last processed step |
| `ACTIVE` | at least two points, last point in the last processed step |
| `ENDED` | last point *before* the last processed step (it stopped being matched) |

Consequences: an event that appears and disappears inside the window becomes its
own `NEW` track; a distant event never joins an existing track (the distance and
speed gates reject it); a storm present in the final step is `ACTIVE` even though
it may die the next hour - "still moving when the window ended".

## 10. Track identifiers

```text
TRK-<EVENT_TYPE>-<YYYYMMDDTHHMM>-<NNN>
```

`<EVENT_TYPE>` is the event type, `<YYYYMMDDTHHMM>` the **start time** of the
track in UTC, and `<NNN>` a 1-based counter per `(event type, start time)`,
assigned in the deterministic order in which the tracker opens tracks. Ids
contain no randomness and no state from previous runs.

## 11. Determinism

* events are sorted by `(timestamp, event_id)` before anything else, so the input
  order cannot change the result;
* matching iterates over a list sorted by `(cost, previous_id, next_id)`;
* tracks are sorted by `(start_time, event_type, track_id)`;
* metrics are pure functions of the coordinates and timestamps.

Re-running the CLI on the same `events.json` therefore produces byte-identical
`tracks.json` apart from the `generated_at` stamp. This is covered by tests for
both input order and repeated runs.

## 12. Storage

```text
data/processed/events.json   # Phase 1 input (written by detect_anomalies.py)
data/processed/tracks.json   # Phase 2 output (written by track_events.py)
```

`tracks.json` carries `schema_version`, run metadata (matching configuration,
lifecycle rule, disclaimer), counters and the tracks. `TrackStore` provides
`save()`, `load()`, `get()`, `list(**filters)`, `clear()` and `exists()`. A
missing file is **not** an error: `load()` returns `None` and the API answers
`count: 0` with an explanatory message. Tracking only ever writes to the
processed directory - `data/raw` is never modified (verified by a test and by a
SHA-256 check of the demo pipeline).

## 13. API

| Endpoint | Purpose |
| --- | --- |
| `GET /api/v1/tracks` | list tracks; filters `event_type`, `status`, `start_time`, `end_time` |
| `GET /api/v1/tracks/{track_id}` | one track (404 with the track count when unknown) |
| `GET /api/v1/tracks/{track_id}/trajectory` | ordered positions for the map layer |

Filter semantics: `start_time` keeps tracks that **end** at or after it;
`end_time` keeps tracks that **start** at or before it. Invalid values are
rejected with 422. The endpoints read `tracks.json`; they never fabricate a
track.

```bash
python scripts/track_events.py
python scripts/track_events.py --max-distance-km 150 --max-speed-kmh 80
python scripts/track_events.py --events data/processed/events.json --no-write
```

## 14. Limitations

1. **Synthetic input.** Tracks are built from synthetic detections; the geometry
   is as good as the generator, not of a real storm.
2. **No learning.** Linking is a fixed cost function; there is no uncertainty, no
   ensemble and no skill score.
3. **Greedy, not optimal.** The one-to-one assignment is greedy over cost; a
   global optimum (e.g. Hungarian) is not used.
4. **Centroid-based.** A track follows the intensity-weighted centroid of the
   detections, so split/merge events and double vortices can confuse it.
5. **No extrapolation.** The tracker never predicts; a track simply ends.
6. **No merging/splitting logic.** Two storms that merge stay two tracks; one
   storm that splits stays one track.
7. **Single run, single file.** No history, no database, no re-tracking of older
   windows, no authentication.
8. **Thresholds are prototype values** (Phase 1) and the matching gates are
   defaults, not tuned for any region or time resolution.

## 15. Synthetic-data disclaimer

The track above is a *prototype* artefact. `TRK-EXTREME_RAINFALL-20240602T0200-001`
with a duration of 7 h and a maximum speed of 0.62 km/h describes a blob that a
deterministic generator moved slightly around 13 N, 77 E - it is **not** a
measured monsoon depression, not a forecast, and it must never be used to issue a
warning. The same disclaimer is embedded in `tracks.json`
(`metadata.data_disclaimer`) and printed by the CLI.
