# Production Readiness (SIH26078, Phase 10 - IMPLEMENTED)

> **Status: HARDENED, NOT PRODUCTION-READY.**
>
> **This system is a research prototype and is not an operational
> disaster-warning system.** Passing the checks in this document means the
> software behaves predictably and fails visibly. It does **not** mean the system
> is ready to run unattended, and it establishes **no** forecast skill,
> accuracy, disaster-reduction or economic benefit.

## What "hardened" means here, and what it does not

Phase 10 made four things true that were not true before:

1. **Failures are visible.** Health, readiness, structured logs, request
   correlation and system metrics.
2. **Bad input is refused.** Path boundaries, extension allow-lists, input
   validation, sanitised errors.
3. **Bad artifacts are refused.** SHA-256 integrity on checkpoints and
   manifests; a corrupt model is never loaded.
4. **The deployment is reproducible.** One validation command runs every check.

What remains untrue:

* there is **no authentication** (see [SECURITY.md](SECURITY.md));
* there is **no demonstrated forecast skill** and no ground truth;
* there is **no HA, no failover, no multi-worker safety**;
* there is **no SLA**.

## Deployment

### Container (recommended)

```bash
docker compose up --build
# frontend http://localhost:5173   backend http://localhost:8000/health
```

Both services run as a **non-root** user (uid 1001). `./data` and `./models`
are bind-mounted, so the host directories must be writable by that uid:

```bash
# Linux
sudo chown -R 1001:1001 data models
# macOS / Windows (Docker Desktop): ownership is mapped automatically
```

The frontend waits for the backend to report **healthy**, not merely started.

### Local development

```bash
python -m uvicorn extreme_weather.api.main:app --reload   # backend
cd frontend && npm run dev                                  # dashboard
```

Safe defaults mean no environment file is required for local work.

## Configuration

Every setting is an environment variable read by
`extreme_weather.config.Settings`. `.env.example` documents each one.

| Variable | Default | Purpose |
| --- | --- | --- |
| `APP_ENV` | `development` | `production` enables production-aware behaviour |
| `LOG_LEVEL` | `INFO` | Threshold; an invalid value falls back to `INFO` |
| `LOG_FORMAT` | `text` | `json` for a log shipper |
| `DATA_DIR` | `data` | Datasets, runs, reports |
| `MODEL_DIR` | `models` | Checkpoints |
| `CORS_ORIGINS` | localhost:5173,127.0.0.1:5173 | Explicit allow-list |
| `ALLOW_CORS_WILDCARD` | `false` | Must be opted into |
| `ENFORCE_DATA_DIR_BOUNDARY` | `true` | Refuse paths outside `DATA_DIR` |
| `VERIFY_ARTIFACT_INTEGRITY` | `true` | Hash artifacts before use |
| `MIN_FREE_DISK_BYTES` | `268435456` | Readiness space threshold (256 MiB) |
| `API_REQUEST_ID_HEADER` | `X-Request-ID` | Correlation header |
| `API_MAX_DETAIL_CHARS` | `2000` | Longest error message returned |
| `WEATHER_API_TOKEN` | unset | Remote-provider bearer token |

**Secrets belong in the environment**, never in the repository, the compose
file or an image layer. See [SECURITY.md](SECURITY.md).

## Health and readiness

Two probes, deliberately different.

| Endpoint | Question | Consults models? | Used by |
| --- | --- | --- | --- |
| `GET /health` | Is the process running? | **No** | Container restart policy |
| `GET /ready` | Should it receive traffic? | Yes | Traffic gating, dashboards |

`/health` never consults the filesystem or a checkpoint, because a liveness
probe that fails on a missing model would restart-loop a perfectly healthy
process.

`/ready` returns an explicit state:

| State | Meaning | `ready` |
| --- | --- | --- |
| `READY` | Configuration valid, directories writable, every component known | `true` |
| `DEGRADED` | Serving, but something optional is missing | `true` |
| `NOT_READY` | Configuration invalid, or a required directory is unwritable | `false` |

**A missing model never yields `READY`.** For this project that is the *normal*
state - no model ships trained - so `DEGRADED` is the expected reading and
`NOT_READY` is reserved for a real inability to serve.

## Logging and metrics

See [OBSERVABILITY.md](OBSERVABILITY.md) for formats, the request-id contract
and the full metric list. In short:

* `X-Request-ID` on every response; `X-Run-ID` accepted for correlation.
* Every log line is redacted; secrets are masked before emission.
* System metrics only. **No accuracy, precision, recall or forecast-skill
  figure is produced anywhere**, because none has been measured.

## Integrity checking

`GET /api/v1/operations/integrity` reports each artifact as `VALID`,
`INCOMPATIBLE`, `CORRUPT` or `MISSING`. A `CORRUPT` artifact is never used, and
`INCOMPATIBLE` (intact but the wrong model version) is reported separately
because it is a different problem.

## Performance

Measured on this environment:

| Observation | Value |
| --- | --- |
| `import extreme_weather.api.main` | ~25-36 s (PyTorch import) |
| `create_app()` | 0.76 s |
| Routes | 57 |
| Typical API latency | 1-50 ms |
| Import cost effect | Once per process; none per request |

The import cost is a PyTorch characteristic on this platform, not a project
defect.

**No scientific behaviour was changed for speed.** A test asserts a pipeline
run yields identical stage statuses and messages with and without the
instrumentation. A regression here would mean a model had been quietly tuned to
make a benchmark look better, which is the failure mode this project has
refused throughout.

## Backup and recovery

| Artefact | Location | Recoverable? |
| --- | --- | --- |
| Datasets | `DATA_DIR` | **Yes** - re-fetch, or restore from backup |
| Manifests | `DATA_DIR/manifests` | Yes, with the data |
| Model checkpoints | `MODEL_DIR` | **Only if backed up** - they are not reproducible by re-running inference |
| Pipeline run records | `DATA_DIR/realtime/runs` | Yes - JSON |
| Evaluation runs | `DATA_DIR/evaluation/runs` | Yes - JSON, carry seed and checksums |
| Alerts | `DATA_DIR/processed` | Yes, but re-generation is not idempotent across time |

Practical notes:

* **Checkpoints are the only irreplaceable artefact.** A trained GNN or
  downscaler cannot be reconstructed by re-running the pipeline. Back up
  `MODEL_DIR` and record each checkpoint's SHA-256; `/api/v1/operations/integrity`
  reports it.
* Run and evaluation records are already self-describing: each carries a seed,
  dataset checksum, split boundaries and a code fingerprint, so a stored result
  can be traced to the data and code that produced it.
* There is **no database**, so backup is a filesystem copy. There is also no
  transactional consistency across files: a copy taken mid-run may capture a
  partially written record.

## Reproducibility

```bash
python scripts/validate_deployment.py            # everything
python scripts/validate_deployment.py --offline  # without network tests
python scripts/validate_deployment.py --fast     # config, API, CLI, compose
python scripts/validate_deployment.py --json     # machine-readable
```

| Exit | Meaning |
| --- | --- |
| 0 | Every check passed |
| 1 | A check failed |
| 2 | A required tool is missing |

**On network-dependent tests.** One provider test calls the real Open-Meteo
archive. It is never skipped, xfailed or weakened. `--offline` **deselects** it
and prints a banner saying the run was a subset - a documented, reproducible
subset, not a passing full suite. The default run includes it.

**On the live server probe.** The API is checked twice: in process over ASGI,
which isolates the application, and over a real socket against a uvicorn
subprocess, which proves the deployment works. That second probe once failed
with `ECONNRESET` ("the socket connection was closed unexpectedly"). The cause
was a start-up race in the validator, not a backend fault: importing the app
costs ~15-36 s on this platform because PyTorch dominates, so a client that
requested immediately saw a refused socket and reported it as a crash. The
probe now allocates its port from the OS, captures stdout and stderr, awaits
`/health` before issuing any other request, and always terminates the process.
A server that dies during start-up is reported with its log, so the failure
says *why* instead of timing out.

## Known limitations

1. **No authentication or authorisation.** A reverse proxy providing it is a
   prerequisite for any non-trusted network.
2. **Single process.** Run history and incremental state are files; two workers
   would race. There is no distributed lock.
3. **No HA, failover or backup scheduling.** One container, one volume.
4. **No SLA.** Nothing here is monitored or paged.
5. **Polling, not push.** The Phase 8 scheduler polls; there is no streaming
   ingest.
6. **No demonstrated forecast skill.** No verified ground-truth warning
   catalogue exists, so no accuracy or skill figure is reported. The models are
   trained on synthetic data and evaluated on reanalysis.
7. **ERA5 is reanalysis** - historical, not live, not a forecast.
8. **Metrics are process-local.** They are lost on restart and are not shared
   between workers.
9. **Import cost is high** on this platform (PyTorch).
10. **CORS is a browser control.** It does not authenticate a non-browser
    client.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Container exits immediately | Host `data`/`models` not writable by uid 1001 |
| Frontend loads, API calls fail | `CORS_ORIGINS` must list the browser origin |
| `NOT_READY` | `GET /ready` -> `components` -> `directories`; then `notes` |
| `DEGRADED` on a fresh install | Expected: no trained checkpoint ships |
| A model will not load | `GET /api/v1/operations/integrity` -> `CORRUPT` vs `INCOMPATIBLE` |
| Cannot trace a request | `X-Request-ID` from the response, then grep the log |
| A run failed but the API is green | Read `data/realtime/runs/*.json`; pipeline failures are not API errors |

## Non-operational disclaimer

**This system is a research prototype and is not an operational
disaster-warning system.** It has no demonstrated forecast skill, no verified
accuracy, no availability guarantee and no operational capability. The alert
severities are project prototype categories, NOT official IMD or WMO warnings.
Nothing in this repository may be used for emergency decisions.
