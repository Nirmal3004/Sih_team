# Real-time / Near-Real-Time Pipeline (Phase 8 - IMPLEMENTED)

> **Status: IMPLEMENTED.** This document describes working code in
> `src/extreme_weather/realtime/`, the `/api/v1/pipeline` router and
> `scripts/run_realtime_pipeline.py`.
>
> **This is a research prototype.** It is not an operational
> emergency-warning system, it has no SLA, and it demonstrates no forecast
> skill. Nothing in this phase should be used for emergency decisions.

## 1. Architecture

Phase 8 wraps the existing Phases 0-7 components in an orchestrator. It
reimplements none of them: every stage calls the same service the CLI scripts
call, so a near-real-time run and a manual run produce the same result by
construction.

```text
        WeatherProvider            LocalFile | OpenMeteo | RemoteHttp | Fixture
              |
              v
        IngestionService           Phase 6: validate, normalise, manifest, register
              |
              v
        QualityGate                VALID | VALID_WITH_WARNINGS | REJECTED
              |
              v
        FreshnessTracker           FRESH | STALE | EXPIRED | UNKNOWN
              |
              v
        PipelineOrchestrator
              |
              +--> PREPROCESS     Phase 0
              +--> ANOMALY        Phase 1
              +--> TRACK          Phase 2
              +--> GNN            Phase 3  (optional)
              +--> DOWNSCALE      Phase 4  (optional)
              +--> ALERT          Phase 5
              |
              v
        Persist Results            data/realtime/runs/*.json
```

| Module | Responsibility |
| --- | --- |
| `realtime/providers.py` | The provider abstraction and four implementations |
| `realtime/freshness.py` | Age classification and thresholds |
| `realtime/quality_gate.py` | Structural checks before any processing |
| `realtime/retry.py` | Retry policy with exponential backoff |
| `realtime/idempotency.py` | Deterministic run identity |
| `realtime/orchestrator.py` | Stage sequencing and failure policy |
| `realtime/scheduler.py` | One-shot and interval execution |
| `realtime/models.py` | `PipelineRun`, `StageResult`, `IngestionRecord` |
| `realtime/logging.py` | Structured stage logging and redaction |
| `realtime/exceptions.py` | Typed errors mapped to HTTP status codes |

## 2. Provider abstraction

```python
class WeatherProvider(ABC):
    def available(self) -> bool: ...
    def fetch(self, request: WeatherRequest) -> ProviderFetch: ...
```

| Provider | Purpose | Credentials |
| --- | --- | --- |
| `local` | Replay / air-gapped runs from a file on disk | none |
| `open-meteo` | ERA5 reanalysis, the Phase 6 path reused unchanged | none |
| `remote-http` | A deployment's own NWP gateway | optional, via env var |
| `fixture` | Deterministic synthetic data for tests and demos | none |

Every provider:

* returns data **in the source's own units** - Phase 6's normaliser is the
  single place that converts;
* records a `ProviderFetch` with a SHA-256 content checksum, used for
  idempotency;
* raises a *specific* exception, so a timeout is retryable and a malformed
  response is not;
* **never invents data**. A provider that cannot deliver raises.

`available()` never raises and never touches the network, so a status endpoint
can poll it cheaply without consuming a provider's rate limit.

### Remote safety

* only `https` is accepted, so a misconfigured endpoint cannot become a
  `file://` read;
* every request carries a timeout;
* responses are capped at 256 MB;
* a token is read from the environment and sent as an **`Authorization` header**,
  never as a query parameter, because a URL can end up in a log.

## 3. Ingestion lifecycle

```text
fetch -> retry (transient only) -> checksum -> identity -> freshness
      -> quality gate -> preprocess -> anomaly -> track -> gnn? -> downscale? -> alert
```

Ingestion is the only stage that touches the network, so it is the only stage
that retries. Every attempt is logged; a run that took three tries says so
rather than looking first-try lucky.

## 4. Freshness calculation

`age = now - newest timestamp in the data`, classified against configurable
windows (default: fresh <= 3 h, stale <= 12 h).

| Status | Meaning | Usable as current? |
| --- | --- | --- |
| `FRESH` | inside the fresh window | yes |
| `STALE` | past fresh, inside expiry | **no** - usable but flagged |
| `EXPIRED` | past the expiry window | no - must not be presented as current |
| `UNKNOWN` | timestamp unparseable, or in the future | no |

Two deliberate choices:

* **`UNKNOWN` is not `FRESH`.** A missing or future timestamp means a clock or
  timezone problem; treating it as current is the worst possible outcome.
* **`STALE` is not "usable as current".** `is_usable_as_current` is true only
  for `FRESH`, so a permissive default cannot be inherited by accident.

By default the pipeline **processes** stale and expired data, because refusing
would make backfills impossible - but the status is always recorded, and
`allow_stale=False` stops an expired run before any processing.

## 5. Quality gates

| Check | Severity | Why |
| --- | --- | --- |
| `spatial_dimensions` | blocking | No grid means no location |
| `coordinates_present` | blocking | No coordinates means no location |
| `time_present` | blocking | No time axis means no event |
| `temporal_coverage` | blocking | Zero steps covers nothing |
| `time_monotonic` | blocking | Reordered time breaks tracking and lag models |
| `no_duplicate_times` | advisory | A repeat can double-count an event |
| `variables_present` | blocking | A missing field makes analysis wrong |
| `units_valid` | advisory | Missing units are common in real files |
| `finite_values` | advisory | Gaps are real, but 95% missing is not usable |

Monotonic means **non-decreasing**: an equal pair of timestamps is duplication,
not disorder, and is reported by the advisory check with the right severity.

A `REJECTED` verdict **stops the run before anomaly detection**. This is the
point of the gate: the downstream stages are optimising code, and given
scrambled timestamps they will return a confident, meaningless result rather
than an error.

## 6. Incremental processing

State lives in `data/realtime/runs/state.json`:

```json
{
  "last_processed": "2024-06-30T09:00:00+00:00",
  "newest_available": "2024-06-30T12:00:00+00:00",
  "new_steps": 3,
  "context_steps": 24
}
```

`new_steps` is `null` - not `0` - when it cannot be determined, because
guessing "nothing new" would silently drop data. In that case the caller
recomputes.

**A full recomputation is required when:**

* `new_steps` is `null` (no prior state, or unparseable timestamps);
* the requested window extends more than `context_steps` beyond the last
  processed time, because anomaly/climatology statistics and GNN graph features
  need history that a new-only window cannot supply;
* `--force` is given, or the fingerprint has not been seen.

`context_steps` (default 24 h) is the history retained for climatology,
tracking continuity and downscaling input.

## 7. Retry strategy

| Failure | Retryable | Reason |
| --- | --- | --- |
| `ProviderTimeoutError` | yes | Transient |
| `ProviderUnavailableError` | yes | Transient |
| `ProviderResponseError` | **no** | A bad payload will be bad again |

Default: 3 attempts, base delay 1 s, factor 2, capped at 60 s
(1 s, 2 s, 4 s...). Exhaustion raises; a partial run is **never** marked
complete.

## 8. Idempotency

The `run_id` is a SHA-256 over:

```text
provider | time window | newest data timestamp | content checksum
```

The same inputs always produce the same `run_id`, so a second identical
ingestion is recognised and short-circuited: the `ingest` stage reports the
skip reason and every later stage is marked `SKIPPED`. Nothing is silently
discarded - the rerun is recorded, and it records *why* it was skipped.

`--force` / `force: true` overrides this deliberately.

Note the checksum is of the **data**, not the request: asking for the same
window twice yields the same identity, while a window that now contains a new
hour does not collide with the previous one.

## 9. Pipeline stages

| Stage | Optional | Fails the run? |
| --- | --- | --- |
| `ingest` | no | yes |
| `validate` | no | yes, on `REJECTED` |
| `preprocess` | no | yes |
| `anomaly` | no | yes |
| `track` | no | yes |
| `gnn` | **yes** | `SKIPPED` when no checkpoint |
| `downscale` | **yes** | `SKIPPED` when no checkpoint |
| `alert` | no | yes |

An absent model is `SKIPPED`, never `FAILED`, and **no prediction is
fabricated**. When a model is present it is actually run; the stage reports
what it did, including "available but nothing could be predicted".

**Phase 0 preprocessing requires `u_wind`/`v_wind`** because it always derives
wind speed. Most weather providers do not serve wind, so when the components
are absent the dataset passes through unchanged and the run records a warning.
Skipping gap-filling and derived wind fields is a real limitation and is
reported as one.

## 10. API

| Method | Path | Returns |
| --- | --- | --- |
| `GET` | `/api/v1/pipeline/status` | state, freshness, stage summary |
| `GET` | `/api/v1/pipeline/runs` | run history, newest first |
| `GET` | `/api/v1/pipeline/runs/{run_id}` | one run in full |
| `POST` | `/api/v1/pipeline/run` | execute now (`?dry_run=true`) |
| `GET` | `/api/v1/ingestion/status` | latest ingestion and provenance |

Status codes carry meaning:

* **400** - invalid provider configuration (bad name, missing `--url`,
  non-https endpoint)
* **404** - unknown run id
* **422** - invalid request, or the data was rejected by the quality gate.
  *Retrying will not help.*
* **503** - a provider or dependency is unavailable. *Retrying might help.*

The request body is **required** for `POST /pipeline/run`: a state-changing
endpoint should not fire on an empty request.

Runs are executed synchronously so the response describes a run that has
actually finished. For long windows use the CLI's `--interval` mode.

## 11. CLI

```bash
python scripts/run_realtime_pipeline.py --help
python scripts/run_realtime_pipeline.py --dry-run
python scripts/run_realtime_pipeline.py --once
python scripts/run_realtime_pipeline.py --interval-minutes 60 --provider open-meteo
python scripts/run_realtime_pipeline.py --once --provider local --file data/raw/x.nc
```

Options: `--once`, `--dry-run`, `--interval-minutes`, `--max-runs`,
`--provider`, `--start`, `--end`, `--hours`, `--variables`, `--lat`, `--lon`,
`--file`, `--url`, `--retries`, `--retry-delay`, `--timeout`, `--fresh-hours`,
`--stale-hours`, `--context-steps`, `--force`, `--no-stale`, `--json`,
`--quiet`, `--verbose`.

Exit codes: `0` success, `1` run failed, `2` configuration error, `130`
interrupted.

A dry run validates the configuration, checks provider availability, prints the
planned stages and writes **no production artifacts at all**.

## 12. Configuration

All optional, all environment-driven:

| Variable | Default | Purpose |
| --- | --- | --- |
| `DATA_DIR` | `data` | Run records and incremental state |
| `WEATHER_API_TOKEN` | unset | Bearer token for `remote-http` |
| `LOG_LEVEL` | `INFO` | Log verbosity |

Freshness windows, retry counts and context length are CLI/API arguments
rather than settings, because they are per-deployment policy, not application
identity.

## 13. Failure handling

| Situation | Result |
| --- | --- |
| Provider unreachable after retries | `ingest` `FAILED`, run `FAILED` |
| Malformed response | `ingest` `FAILED` immediately, no retry |
| Quality gate rejects | `validate` `FAILED`, no later stage runs |
| Preprocessing fails | `preprocess` `FAILED` |
| No GNN checkpoint | `gnn` `SKIPPED` with the reason |
| No downscaling checkpoint | `downscale` `SKIPPED` with the reason |
| Alerting fails | `alert` `FAILED`; earlier stages are kept |
| Identical rerun | `ingest` `SUCCESS` (skip noted), later stages `SKIPPED` |

A failed persistence write logs an error but does not fail the run: losing the
log of a run is bad, losing the run is worse.

## 14. Logging

```text
[INGEST     ] provider=fixture steps=7 checksum=533e8326dcca attempts=1
[INGEST     ] freshness=STALE - Data is 4.0 h old, past the 3 h freshness window.
[VALIDATE   ] status=VALID checks=9 blocking_failures=0
[PREPROCESS ] steps=7 vars=2
[ANOMALY    ] events=0
[TRACK      ] tracks=0 multi_point=0
[GNN        ] SKIPPED: No trained GNN checkpoint is available.
[DOWNSCALE  ] SKIPPED: No trained AI downscaling checkpoint is available.
[ALERT      ] alerts=16
```

Every message is passed through `redact()`, which masks anything
credential-shaped. This is a backstop: providers already avoid credentials in
URLs, but a third-party error string can contain anything.

## 15. Testing

`tests/phase8_support.py` provides `StubProvider` and `FlakyProvider` so the
suite runs with **no network and no API key**. A suite that reaches a live
provider fails intermittently and cannot prove anything about failure handling.

| Module | Covers |
| --- | --- |
| `test_phase8_providers.py` | valid/malformed/timeout/retry/missing vars/invalid units, URL safety |
| `test_phase8_freshness.py` | fresh, stale, expired, unknown, boundaries, configurability |
| `test_phase8_quality.py` | coordinates, duplicates, variables, NaN-heavy, coverage |
| `test_phase8_pipeline.py` | success, rejection, retry, optional models, idempotency, dry run |
| `test_phase8_api.py` | status, run, history, detail, 400, 404, 422, 503 |
| `test_phase8_scheduler_cli.py` | retry policy, redaction, scheduler, CLI |

## 16. Limitations

* **No operational reliability.** No SLA, no failover, no multi-region
  redundancy, no on-call model.
* **No demonstrated forecast skill.** No skill score has been measured against
  ground truth; no ground-truth catalogue is available.
* **Polling only.** There is no push ingest and no websocket; the scheduler polls
  on an interval.
* **Single-process state.** Run history and incremental state are files. Two
  API workers would race.
* **No backfill scheduler.** Missed ticks are not caught up on restart, to avoid
  a burst of identical ingests.
* **ERA5 is historical reanalysis**, not live data and not a forecast.
* **Synthetic by default.** The `fixture` provider is deterministic synthetic
  data, labelled as such in its attributes, notes and logs.
* **Phase 0 preprocessing is skipped** when a provider does not serve wind.

## 17. Security

* Credentials are read from environment variables, never from Git.
* Tokens are sent as headers, never as query parameters.
* URLs are redacted in manifests, logs and API responses (userinfo stripped,
  credential-shaped query values masked).
* Only `https` endpoints are accepted.
* Every remote call has a timeout and a response size cap.
* No credential ever appears in a log line: `redact()` is applied to every
  message.

## 18. Non-operational disclaimer

**This is a research prototype built for SIH26078.** It is not an operational
emergency-warning system. It has no demonstrated forecast skill, no verified
accuracy claims and no availability guarantee. The alert severities are
project prototype categories, NOT official IMD or WMO warnings. Nothing here
may be used for emergency decisions.
