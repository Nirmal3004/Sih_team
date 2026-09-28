# Observability (SIH26078, Phase 10 - IMPLEMENTED)

> **Status: IMPLEMENTED.** `src/extreme_weather/observability/`,
> `src/extreme_weather/api/middleware.py`.
>
> **Research prototype.** Observability says whether the system is working. It
> says nothing about whether the weather prediction is right, and no metric in
> this document is a model-quality metric.

## The boundary this module defends

| It measures | It never measures |
| --- | --- |
| Request volume and latency | Accuracy |
| Error counts by status | Precision / recall / F1 / AUC |
| Pipeline runs and durations | Forecast skill |
| Events, tracks and alerts produced | Disaster reduction |

The two live in separate modules on purpose. Mixing a request counter with a
model-quality claim is the most common way a dashboard ends up asserting a
number nobody measured, so `extreme_weather.observability` contains no field
whose name could be mistaken for a score, and a test asserts that.

## Log format

Two renderings of the same record. `LOG_FORMAT=text` (default) for a terminal:

```text
2026-01-01 12:00:00 INFO    [extreme_weather.api] request completed method=GET
    path=/api/v1/events status=200 duration_ms=12.4 request_id=req-8f3a1c9d
```

`LOG_FORMAT=json` for a shipper - one object per line:

```json
{"ts":"2026-01-01T12:00:00.123456+00:00","level":"INFO",
 "logger":"extreme_weather.api","component":"extreme_weather.api",
 "event":"request.completed","message":"request completed method=GET",
 "method":"GET","status":200,"duration_ms":12.4,
 "request_id":"req-8f3a1c9d","run_id":"run-ec0398bacf331738"}
```

| Field | Meaning |
| --- | --- |
| `ts` | UTC, ISO-8601, microsecond precision |
| `level` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `component` | The logger name, e.g. `api`, `pipeline` |
| `event` | Stable dotted name, e.g. `pipeline.stage.completed` |
| `message` | Human-readable, redacted |
| `request_id` | Correlation for one HTTP request |
| `run_id` | Correlation for one pipeline run |
| `error_type`, `error` | Present on failure; the traceback is not inlined |

`event` is the field to alert or aggregate on. Prose changes; the dotted name
does not.

## Log levels

| Level | Used for |
| --- | --- |
| `DEBUG` | Diagnostics; off by default |
| `INFO` | A request completed, a stage finished, a run started/ended |
| `WARNING` | A request was rejected, a component is degraded, an optional model is absent |
| `ERROR` | An unhandled failure; the traceback goes here and only here |

`LOG_LEVEL` sets the threshold. Invalid values fall back to `INFO` rather than
crashing a container at startup.

## Request correlation

Every response carries `X-Request-ID`.

* **Not supplied** -> a fresh `req-<16 hex>` is generated. `uuid4`, not a
  counter: a guessable request id is an enumeration vector.
* **Supplied** -> honoured only if it matches `[A-Za-z0-9_.:-]{1,128}`. This is
  a **security control**, not a formality: reflecting an arbitrary client
  string into a response header is a header-injection vector.
* The id is bound to a `ContextVar`, so every log line emitted while handling
  that request carries it, and concurrent requests cannot see each other's.

`X-Run-ID` is also accepted, so a pipeline run can be correlated with the
requests that triggered it.

## Run correlation

`bind_context(run_id=...)` attaches a run id to every record emitted inside the
block. The Phase 8 orchestrator binds its derived `run_id`, so all eight stage
lines of a run carry the same identifier.

## Operational metrics

Served at `GET /api/v1/operations/metrics`. Counters are labelled, and a
declared counter reads as **0** rather than "no data" - the two are different
states and a dashboard must be able to tell them apart.

| Counter | Meaning |
| --- | --- |
| `api_requests_total{method}` | Requests served |
| `api_errors_total{method,status}` | Responses >= 400 |
| `pipeline_runs_total` | Pipeline runs started |
| `pipeline_runs_succeeded_total` | Runs that completed without a failed stage |
| `pipeline_runs_failed_total` | Runs that failed |
| `events_detected_total` | Phase 1 events produced |
| `tracks_generated_total` | Phase 2 tracks produced |
| `alerts_generated_total` | Phase 5 alerts produced |
| `gnn_inference_attempts_total` | GNN inference attempts |
| `downscaling_inference_attempts_total` | Downscaling inference attempts |
| `integrity_checks_total` / `integrity_failures_total` | Artifact integrity |

Histograms: `api_request_duration_seconds{method}`,
`pipeline_stage_duration_seconds{stage}`, `pipeline_run_duration_seconds`.
Buckets resolve a 5 ms API call from a 5-minute batch job:

```text
0.005 0.01 0.025 0.05 0.1 0.25 0.5 1 2.5 5 10 30 60 300
```

## Stage timing

`StageTimer` records each stage's duration to both the log and the histogram,
and **records a failed stage too** - a stage's duration is exactly the thing
worth knowing when it fails.

The run record already carried per-stage durations; this adds the live view, so
a slow stage is visible in a running log rather than only after the fact.

**No scientific behaviour was changed for speed.** A test asserts that a
pipeline run produces byte-identical stage statuses and messages with and
without the instrumentation.

## Secret handling

Every message and every structured field passes through `redact()` before
emission:

| Pattern | Result |
| --- | --- |
| `api_key=VALUE` | `api_key=***` |
| `Authorization: Bearer TOKEN` | `Authorization: ***` |
| `?apikey=VALUE&x=1` | `?apikey=***&x=1` |
| a field **named** `api_key` | `***` regardless of its value |

Values over 512 characters are truncated. This is a backstop: providers and URL
builders already avoid credentials, but a third-party exception string can
contain anything, and a log file outlives the process.

## Error envelope

Every error response uses one shape:

```json
{
  "error": "not_found",
  "message": "No detected event with id 'X'.",
  "detail": "No detected event with id 'X'.",
  "request_id": "req-8f3a1c9d"
}
```

`detail` is retained as an alias of `message` because Phase 0-9 clients and
tests read it; removing it would be a breaking change to a published contract.
The envelope is **additive** and no status code changed.

A **500 never** exposes a stack trace, a filesystem path or an internal
exception message. The detail goes to the log; the client gets the request id.

A **503 is not masked**: "no trained checkpoint; run `scripts/train_gnn.py`" is
actionable guidance, and hiding it would leave the caller with nothing to act
on. The rule is authorship - a message a handler wrote for a caller is
returned; an exception nobody handled is not.

## Troubleshooting

**A request cannot be found in the logs.**
Take `X-Request-ID` from the response and grep for it. If it is absent, the
request did not reach this service.

```bash
curl -i localhost:8000/api/v1/events | grep -i x-request-id
grep 'req-8f3a1c9d' app.log
```

**A stage is slow.**
`GET /api/v1/operations/metrics` -> `histograms` ->
`pipeline_stage_duration_seconds{stage=gnn}`. Compare `avg_seconds` with
`buckets` to see whether it is one slow run or a uniformly slow stage.

**A run is failing but the API is green.**
Pipeline failures are recorded on the run, not the API. Read the run record in
`data/realtime/runs/`, or `GET /api/v1/pipeline/runs/{run_id}`, and look at
`errors`.

**Readiness is `NOT_READY`.**
`GET /ready` -> `components`. A `directories` entry with state `UNAVAILABLE`
means `DATA_DIR` or `MODEL_DIR` is missing or unwritable; `notes` says which.

**A model will not load.**
`GET /api/v1/operations/integrity` distinguishes `MISSING` (no checkpoint - the
normal state here), `INCOMPATIBLE` (intact but the wrong model) and `CORRUPT`
(hash mismatch or unloadable). A `CORRUPT` artifact is never used.

**Logs look like prose.**
`LOG_FORMAT=text` is the default and is for humans. Set `LOG_FORMAT=json` for a
shipper or a `jq` pipeline.

## Measured on this environment

| Observation | Value |
| --- | --- |
| `import extreme_weather.api.main` | ~25-36 s (dominated by the PyTorch import) |
| `create_app()` alone | 0.76 s |
| Route count | 57 |
| Typical API latency | 1-50 ms (see the `duration_ms` field) |

The import cost is a PyTorch characteristic on this platform, not a project
defect; it is paid once per process and does not affect per-request latency.
