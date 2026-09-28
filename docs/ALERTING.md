# DeepWeather AI (SIH26078) - Alerting and Emergency Response (Phase 5)

> **Status: IMPLEMENTED (Phase 5, prototype).**
>
> **This is not an operational system.** Every severity name in this document is a
> *project prototype category*. None of them is an IMD category, a WMO category, or
> any official warning. The data is synthetic, the scores are rule-based, and
> nothing here may be used to make an emergency decision.

---

## 1. What Phase 5 adds

Phase 5 turns the Phase 0-4 pipeline output into **structured alerts with a
lifecycle**, exposes them over REST, and renders them on the dashboard:

```text
  Phase 1 events  ─┐
  Phase 2 tracks  ─┼─> risk engine ─> alert rules ─> dedup ─> alert store
  Phase 3 GNN    ─┤   (deterministic,  (R1..R7)   (event +      │
  Phase 4 down.  ─┘    explainable)                level +        │
                                                          bucket)   │
        ┌─────────────┼─────────────┐                             │
        v             v             v                             v
   REST API      Dashboard     Notification                data/processed/
                              adapters (local mock)       alerts.json
```

### The honesty contract

Four rules shaped the whole implementation:

1. **An alert must say why it exists.** Every alert carries a
   `RiskAssessment` with a `contributing_factors` list and a plain-language
   `explanation`. The score is a plain weighted sum a reader can redo by hand.
2. **An alert must say what it did not use.** `source_components` records the
   stages that actually contributed. A run with no GNN checkpoint reports
   `["events", "tracking"]` - it does not silently imply a prediction.
3. **A missing model is not an error.** The GNN, the downscaler and the event and
   track stores are all optional. A missing one degrades the run and is reported
   in `degraded`; it never crashes the API.
4. **No claim is stronger than the evidence.** There is no accuracy percentage,
   no "guaranteed prediction", no "exact evacuation zone" anywhere in the model,
   the API, the CLI or the dashboard.

---

## 2. Architecture

| Module | Responsibility |
| --- | --- |
| `alerts/severity.py` | The severity ladder and the configurable score cut points. |
| `alerts/models.py` | `Alert`, `RiskAssessment`, `ContributingFactor`, `AlertStoreResult`, deterministic ids. |
| `alerts/engine.py` | The deterministic 0-100 risk score and its explanation. |
| `alerts/rules.py` | The seven configurable rules (R1-R7) and their outcomes. |
| `alerts/service.py` | Orchestration: assemble context, score, apply rules, deduplicate, persist, transition. |
| `alerts/store.py` | The file-backed `alerts.json` store. |
| `alerts/notifications.py` | The `NotificationProvider` protocol, the local mock and the two stubs. |
| `alerts/simulation.py` | The deterministic lifecycle demonstration. |
| `alerts/schemas.py` | FastAPI request/response models. |
| `api/routers/alerts.py` | The REST endpoints. |

### File layout

```text
data/processed/events.json          # Phase 1  (input)
data/processed/tracks.json          # Phase 2  (input)
models/gnn/best.json                # Phase 3  (optional input)
models/downscaling/best.pt          # Phase 4  (optional input)
data/processed/alerts.json          # Phase 5  (output)
data/processed/notifications.json   # Phase 5  (output, local log only)
```

Nothing is ever written to `data/raw`.

---

## 3. The alert model

`Alert` is a validated Pydantic model (`extra="forbid"`), so an invalid status,
coordinate or timestamp is rejected at the boundary rather than reaching the
store. Grouped by purpose:

| Group | Fields |
| --- | --- |
| Identity | `alert_id`, `event_id`, `track_id`, `event_type`, `deduplication_key` |
| Classification | `severity`, `priority`, `confidence_label`, `risk` |
| Text | `title`, `message` |
| Timing | `created_at`, `updated_at`, `valid_from`, `valid_until`, `acknowledged_at`, `resolved_at`, `resolved_reason` |
| Position | `current_lat/lon`, `predicted_lat/lon`, `affected_radius_km` |
| Motion/intensity | `intensity`, `predicted_intensity`, `movement_speed_kmh`, `movement_direction` |
| Provenance | `source_components` |
| Lifecycle | `status` |
| Honesty | `notice` |

Key semantics:

* `current_*` are **observed** values copied from the last track point. They are
  never predicted.
* `predicted_*` are the Phase 3 GNN output, or `None` when the model was
  unavailable. Absence is explicit, never a silent zero.
* `acknowledged_at` / `resolved_at` are the real timestamps recorded when the
  operator acted. They are never back-dated or invented.
* `source_components` is the provenance list (`events`, `tracking`, `gnn`,
  `downscaling`).

### Statuses

`AlertStatus` is an ordered `StrEnum`: `NEW`, `ACTIVE`, `ACKNOWLEDGED`,
`RESOLVED`, `EXPIRED`. `RESOLVED` and `EXPIRED` are terminal
(`.is_terminal`), and `NEW`/`ACTIVE`/`ACKNOWLEDGED` are "open" (`.is_open`).


---

## 4. Severity

`AlertSeverity` is an ordered `StrEnum`:

| Severity | Rank | Meaning in this prototype |
| --- | --- | --- |
| `INFO` | 0 | Recorded for context; no operator action implied. |
| `WATCH` | 1 | Something notable, below the action threshold. |
| `ADVISORY` | 2 | Worth a look, no action implied. |
| `WARNING` | 3 | An operator should look at it. |
| `SEVERE_WARNING` | 4 | An operator should act. |
| `EXTREME` | 5 | The highest band this prototype can assign. |

> **These names are this project's own.** They are *not* IMD categories, *not*
> WMO categories, and are not calibrated against real weather. The string
> `SEVERITY_DISCLAIMER` says so and is attached to every payload that reports a
> severity.

The mapping from the Phase 1 detection severity onto the alert ladder is
explicit (`AlertSeverity.from_event_severity`), not implicit:

| Phase 1 `Severity` | Phase 5 `AlertSeverity` |
| --- | --- |
| `NORMAL` | `INFO` |
| `WATCH` | `WATCH` |
| `MODERATE` | `ADVISORY` |
| `SEVERE` | `WARNING` |
| `EXTREME` | `SEVERE_WARNING` |

### Configurable cut points

`SeverityThresholds` holds the score bounds that select a level (defaults 25 /
45 / 70). It validates `0 <= warning < severe_warning < extreme <= 100`, so an
invalid ladder fails loudly rather than silently mis-classifying alerts.

---

## 5. Risk calculation

### Design rules

* **No black box.** The score is `sum(weight * normalized_value)` over nine named
  factors whose weights total exactly 100, so the result lands directly on the
  0-100 scale the severity ladder expects.
* **No randomness.** No sampling, no clock read, no model inference. The same
  inputs always give the same score - which is what makes the tests and the
  `--dry-run` mode meaningful.
* **No probability.** The score is a triage aid with no probability meaning.

### The nine factors

| Factor | Weight | Normalised from | Scores 0 when |
| --- | --- | --- | --- |
| `anomaly_severity` | 22 | Phase 1 severity rank / 4 | severity is `NORMAL` |
| `intensity` | 14 | `intensity / reference(type)` | intensity is 0 |
| `intensity_trend` | 12 | 1.0 strengthening, 0.4 stable, 0.0 weakening | trend is weakening |
| `movement_speed_kmh` | 8 | `speed / 60 km/h` | the event did not move |
| `predicted_intensity_change` | 12 | `delta / reference(type)` | **no GNN prediction** |
| `predicted_movement` | 6 | `predicted_speed / 60 km/h` | **no GNN prediction** |
| `affected_radius` | 10 | `radius / 150 km` | radius is 0 |
| `persistence` | 10 | `duration / 12 h` | **event is not tracked** |
| `downscaled_threat` | 6 | `threat / reference(type)` | **no downscaled field** |

The reference values live in `RiskConfig` and are configurable. They are
*display/reference constants of the prototype ladder*, not physical danger
thresholds.

### Missing inputs are visible, not silent

When a factor cannot be evaluated its `points` is `0.0` **and** its `detail`
says why (`"GNN prediction unavailable - factor contributes nothing"`). The
explanation additionally carries an explicit line, for example:

```text
WARNING priority (score 33.37/100) because:
- Observed intensity: 41.75 against a reference of 50.00 (+11.69)
- Anomaly severity: Phase 1 severity MODERATE (+11.00)
- Affected radius: 88.14 km against a reference of 150.00 km (+5.88)
- Intensity trend: no track history available; trend treated as neutral (+4.80)
- no GNN prediction available, so no forward-looking intensity was used
- no AI-downscaled threat field available, so no ~5 km threat information was used
Rule-based prototype triage score. It is NOT a validated disaster-risk index,
NOT calibrated against impact, and NOT a probability of occurrence.
```

Note the "no track history" wording: an untracked event has no first intensity,
so the trend is reported as **neutral** rather than inferred from a placeholder
zero. Inferring a trend from a value that was never observed would be inventing
evidence.

`RiskAssessment.total_points` is asserted equal to `score` in the test suite, so
there is no hidden term and no unexplained rounding gap.

### `confidence_label`

This describes **input completeness**, never correctness:

| Label | Rule |
| --- | --- |
| `HIGH` | `gnn` **and** `downscaling` both contributed |
| `MEDIUM` | exactly one of them contributed |
| `LOW` | detection and tracking only |



---

## 6. Alert rules

Seven rules live in `ALERT_RULES`, evaluated in registry order. Each is a pure
function of a `RuleContext` returning a `RuleOutcome`, so it is unit-testable in
isolation and the decision is auditable.

| ID | Name | Effect | Key thresholds (`RuleConfig`) |
| --- | --- | --- | --- |
| **R1** | `extreme_anomaly_creates_alert` | Creates an alert at or above the trigger severity; suppresses anything weaker. | `min_trigger_severity` (default `WATCH`) |
| **R2** | `severe_strengthening_elevates` | Raises the **operator priority** of a severe, strengthening event. Never lowers it. | `elevate_for_severities` |
| **R3** | `persistent_track_maintains` | Keeps a long-tracked event `ACTIVE`, preserving `ACKNOWLEDGED` if a human already acted. | `persistence_hours` (3.0) |
| **R4** | `predicted_increase_raises_level` | Raises the level when the Phase 3 prediction shows growth. **Inert without a prediction.** | `predicted_increase_delta` |
| **R5** | `weakening_downgrades_or_resolves` | Downgrades a weakening event, or resolves it below the threshold. | `resolve_intensity_threshold`, `downgrade_on_weakening` |
| **R6** | `missing_event_expires` | Expires an **existing** alert whose event has gone. Never overrides an operator decision. | `expire_after_hours` (6.0) |
| **R7** | `duplicate_suppressed` | Detects an open alert with the same dedup key so the service refreshes it instead of duplicating. | `deduplication_window_minutes` |

Two design points worth calling out:

* **R4 is inert without a prediction.** A missing GNN checkpoint must never be
  read as "the event will intensify".
* **R6 only applies to an alert that already exists.** The first evaluation of a
  historical dataset is not staleness, so a newly created alert always starts
  open regardless of how old the underlying event is.

---

## 7. Deduplication

The dedup key is:

```text
event_id + "|" + alert_level + "|" + time_bucket
```

where `time_bucket` is epoch-aligned (`int(ts)//60 // minutes * minutes`). The
alert id is derived from that key (`ALT-<event>-<LEVEL>-<bucket>`), so it is
stable across runs, contains no random component, and re-running the generator
over the same data reproduces the same ids.

Effect: a second run **inside the window refreshes the existing alert**
(`updated_count` goes up, `created_count` stays 0); a run in a **later bucket**
creates a genuinely new alert. That is the intended behaviour - a genuinely
escalated threat deserves a new alert - and it is why the key includes the
level.

Verified by `test_rerunning_in_the_same_window_updates_instead_of_duplicating`
and `test_dry_run_does_not_modify_an_existing_store` (byte-for-byte comparison
of both stores across a dry run).

---

## 8. Lifecycle

```text
  NEW ──> ACTIVE ──> ACKNOWLEDGED ──> RESOLVED   (terminal)
   │        │             │
   └────────┴─────────────┴──────> EXPIRED      (terminal)
```

The legal transitions are declared in `ALLOWED_TRANSITIONS` and enforced by
`AlertService._check_transition`. An illegal transition raises
`AlertTransitionError` (HTTP 422) rather than silently succeeding - a dashboard
that quietly accepts "resolve an expired alert" is worse than one that refuses.

`acknowledge_alert`, `resolve_alert` and `promote_alert` are idempotent for
their own target state (re-acknowledging keeps the original timestamp);
acknowledging a `RESOLVED` alert is still refused.

`update_alert` refuses to change `status`, `alert_id`, `created_at`,
`acknowledged_at` or `resolved_at`, so the transition check cannot be bypassed.
`expire_alerts` never expires an `ACKNOWLEDGED` alert, because acknowledgement
is an operator decision rather than a stale-record artefact.

---

## 9. Store

`data/processed/alerts.json` - a versioned JSON envelope mirroring the Phase 1
and Phase 2 stores. No database; the state stays inspectable with `cat`.

| Failure | Behaviour |
| --- | --- |
| Missing file | `load()` returns `None`; the API returns an empty list with a message, not a 500. |
| Empty (0-byte) file | Treated as an empty store (`metadata["empty_file"]`), not an error. |
| Malformed JSON | Raises `AlertStoreError`; the router maps it to HTTP 422. |
| Non-object payload | Raises `AlertStoreError`. |
| Malformed records | Skipped; counted in `metadata["skipped_records"]`. One bad alert cannot hide the good ones. |
| Duplicate `alert_id` | Collapsed to the first occurrence, counted as skipped. |
| Write failure | Raises `AlertStoreError` instead of leaking an `OSError`. |

---

## 10. Notification adapters

```python
class NotificationProvider(Protocol):
    def send(self, alert: Alert) -> NotificationRecord: ...
    def supports(self) -> bool: ...
    def name(self) -> str: ...
```

| Provider | Enabled by default | Behaviour |
| --- | --- | --- |
| `MockNotificationProvider` | **yes** | Appends to `data/processed/notifications.json`; `delivered=True`. |
| `EmailNotificationProvider` | no | Stub. `supports() -> False`; reports non-delivery. |
| `WebhookNotificationProvider` | no | Stub. `supports() -> False`; reports non-delivery. |

The stubs exist so the adapter architecture is real, but they never claim a
delivery and never read a credential. **No API key, token, password or webhook
secret is read, stored or logged anywhere in this project.**

Notifications are best-effort by construction: `AlertService.notify` catches a
provider failure and records it as a non-delivery, so a broken channel can never
break alert generation. If the local log write itself fails, the record is
marked `delivered=False` with the reason - the alert itself is untouched.

The notification body includes the full risk explanation, so a logged
notification is self-contained.



---

## 11. API

```text
GET  /api/v1/alerts                      # list + filters
GET  /api/v1/alerts/{alert_id}           # one alert + lifecycle timeline
GET  /api/v1/alerts/summary              # dashboard counters
POST /api/v1/alerts/generate             # run the generation pipeline
POST /api/v1/alerts/{alert_id}/acknowledge
POST /api/v1/alerts/{alert_id}/resolve
POST /api/v1/alerts/simulate             # deterministic demo
GET  /api/v1/notifications               # local notification log
```

Filters on `GET /api/v1/alerts`: `status`, `severity`, `event_type`,
`start_time`, `end_time`, `priority`.

### Status codes

| Code | When |
| --- | --- |
| 200 | Success. |
| 404 | Unknown alert id, or no alert file exists. |
| 422 | Invalid filter value, invalid body, illegal lifecycle transition, or a corrupt store file. |
| 503 | `POST /generate` while `ALERT_GENERATION_ENABLED=false` - a genuinely unavailable capability, **not** a fake success. |

Every request model uses `extra="forbid"`, so a typo in a body is a 422 rather
than a silently ignored field. Every response carries the prototype `notice`.

`/summary` is declared **before** `/{alert_id}` so the literal path is not
captured by the parameter.

### Running it

```bash
python scripts/generate_alerts.py               # write alerts + notifications
python scripts/generate_alerts.py --dry-run     # write NOTHING
python scripts/generate_alerts.py --use-gnn     # enrich with the Phase 3 model
python scripts/generate_alerts.py --no-notify   # skip the adapters
python scripts/simulate_alerts.py               # lifecycle demonstration
```

`--dry-run` is verified by `test_dry_run_does_not_modify_an_existing_store`,
which compares both files byte-for-byte before and after.

---

## 12. Dashboard

`frontend/src/components/AlertsPanel.tsx` renders:

* **Top counters** - Active alerts, Severe alerts, Events tracked, Predicted
  threats, Acknowledged, Resolved, Expired.
* **Alert table** - severity, event type, current location, predicted location,
  intensity, trend, speed, direction, status, created time. Selecting a row
  opens the details.
* `AlertDetails.tsx` - the full alert, its component provenance and the
  acknowledge/resolve buttons (disabled in terminal states).
* `RiskSummary.tsx` - the score meter, the reason list, and a per-factor table
  whose points sum to the score. Inactive factors are listed separately.
* `AlertTimeline.tsx` - the lifecycle stages with their real timestamps.
* `AlertMap.tsx` - a **separate** Leaflet panel with the current position
  (filled circle), the affected radius (dashed outline), the predicted position
  (outlined circle, only when a prediction exists) and the move between them.

`WeatherMap.tsx` is deliberately **untouched**: adding overlays inside the proven
Phase 0 map risked breaking it, so the alert layers live in their own component
on the same stack.

The dashboard never shows an accuracy percentage, never calls a severity an
official warning, and never auto-acknowledges or auto-resolves anything.

---

## 13. Simulation

`python scripts/simulate_alerts.py` (or `POST /api/v1/alerts/simulate`) walks a
synthetic event through:

```text
detected -> tracked -> predicted -> downscaled
         -> alert created -> acknowledged -> resolution
```

It reuses the **real** service, store, rules and notification adapters, so a demo
shows what the pipeline actually does. It is fully deterministic: the timeline is
anchored to the constant `SIMULATION_START`, never the wall clock, so the same
inputs always yield the same steps, ids and scores.

Honesty properties, all asserted in `tests/test_alert_simulation.py`:

* Every response is labelled `SIMULATION - SYNTHETIC DATA` and carries a
  disclaimer.
* A stage that did not happen is reported as such with `timestamp: null` - a
  timestamp is never invented.
* `--dry-run` writes nothing.
* The dashboard's "Simulate lifecycle" button always passes `dry_run: true`, so
  a demo click can never mutate the real store.
* With no events it returns a single `no_input` step explaining which script to
  run, rather than an empty 200.

---

## 14. Configuration

Only settings the code actually consumes:

| Variable | Default | Effect |
| --- | --- | --- |
| `ALERTS_FILENAME` | `alerts.json` | Alert file under `<DATA_DIR>/processed`. |
| `NOTIFICATIONS_FILENAME` | `notifications.json` | Local notification log. |
| `ALERT_GENERATION_ENABLED` | `true` | `false` makes `POST /generate` return 503; stored alerts are still served. |
| `ALERT_DEFAULT_VALIDITY_MINUTES` | `360` | How long a new alert is current. |
| `ALERT_DEDUPLICATION_WINDOW_MINUTES` | `60` | Dedup bucket size. |

Everything else (factor weights, reference values, severity thresholds, rule
thresholds) is a validated `RiskConfig` / `RuleConfig` / `SeverityThresholds`
dataclass passed in code or on the CLI - there are no environment variables for
them, and none were invented.

---

## 15. Failure handling

Every optional input degrades instead of raising:

| Missing / broken | Result |
| --- | --- |
| `events.json` | 0 alerts; `degraded` says to run `detect_anomalies.py`. |
| `tracks.json` | Detection-only alerts, `source_components = ["events"]`. |
| GNN checkpoint | `predicted_*` is `None`; the two prediction factors score 0 and say so. |
| Downscaling checkpoint / field | The downscaled factor scores 0 and says so. |
| GNN service raising | Caught, recorded in `degraded`; alerts still produced. |
| Malformed optional model output | Same path - the value is treated as absent. |
| Corrupt `alerts.json` | HTTP 422 with the parse error. |
| Unwritable notification log | The record is marked `delivered=False`; the alert is unaffected. |
| Broken notification provider | Caught by `notify`; recorded as a provider failure. |

`source_components` is the authoritative record of what was used, so a
degraded run is visible rather than merely tolerated.



---

## 16. Security

* **No secrets anywhere.** No API key, token, password or webhook secret is
  hard-coded, read, stored or logged. The email and webhook adapters read no
  credential at all.
* **No new dependencies.** Phase 5 adds no Python package and no npm package.
* **Secrets are not in the Docker or compose files.** The alert and notification
  paths are mounted data directories, already covered by the existing volumes.
* **Request validation is strict.** `extra="forbid"` on every request model;
  path parameters are bound by FastAPI.
* **Known gaps (deliberate, Phase 6+):** there is **no authentication and no
  authorisation** on the acknowledge/resolve endpoints. They are open in this
  prototype, exactly as the read endpoints are. Adding an identity layer is
  future work and is called out here rather than being quietly assumed.

---

## 17. Limitations

Stated plainly, because a prototype that hides these is worse than useless:

* **The score is not a risk index.** It is a weighted sum of nine normalised
  factors. It is not calibrated against impact, casualties, exposure or any
  outcome, and it is not comparable across regions, seasons or event types.
* **The reference values are arbitrary ladder constants.** "50 mm/h", "150 km",
  "60 km/h" and "12 h" were chosen to make the demo readable. They have no
  meteorological justification and are not thresholds anyone should act on.
* **The severities are not official categories.** Renaming them to match a real
  scheme would require real calibration; they are not interchangeable.
* **One event type path is thin.** The intensity reference is per event type,
  but the whole demo dataset is extreme rainfall, so the wind and temperature
  paths are structurally present and empirically untested.
* **Only the newest detection of a track is alerted on.** Earlier points of a
  long track are treated as superseded. This avoids re-issuing the same threat
  every hour, but it means the alert history is not a complete event log.
* **No persistence beyond a JSON file.** No database, no history retention, no
  concurrency control. Two simultaneous `generate` calls can race on the file.
* **No real escalation.** There is no on-call rotation, no paging policy, no
  SLA and no duty officer. An alert sitting `ACKNOWLEDGED` is only as good as
  someone noticing it.
* **The simulation demonstrates the plumbing, not the science.** It proves the
  lifecycle works end to end; it says nothing about whether any of it would be
  right on real weather.
* **Downscaling does not improve accuracy.** Carried over from Phase 4: on the
  demo data the AI downscaler beats nearest-neighbour but **loses to cubic
  interpolation** on MAE. A ~5 km grid is not a more accurate grid, and the
  alerting layer inherits that.

---

## 18. Synthetic-data disclaimer

Every number in this layer comes from the project's deterministic synthetic
generator (`scripts/generate_sample_data.py`). The data is **not** observed
weather, **not** a reanalysis, and **not** derived from any real event. The
simulation attaches a *generated* downscaled threat value to synthetic events so
the demo can exercise that code path; it is a number attached to a fictional
event, not a model output.

---

## 19. Non-operational disclaimer

**This system is not operational and must not be used for emergency decisions.**

The anomaly detection, event tracking, GNN prediction and AI downscaling it
builds on are research prototypes trained and evaluated on synthetic data. The
alert severities are project-invented categories. The risk score is a rule-based
triage aid. None of it has been validated against observed weather, verified
against outcomes, or reviewed by a forecast authority. There is no
authentication, no audit trail, no availability guarantee and no support model.

For real warnings, use the official issuing authority for your region.

