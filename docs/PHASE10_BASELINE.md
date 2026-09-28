# Phase 10 baseline and regression record

Recorded before and after the Phase 10 work, per the Phase 10 stop conditions.
No pre-existing test was modified, weakened, deleted, skipped or xfailed.

## Pre-Phase-10 baseline (recorded before any Phase 10 change)

```bash
python -m pytest -q
```

| Metric   | Value |
| -------- | ----- |
| Total    | 1643  |
| Passed   | 1643  |
| Failed   | 0     |
| Skipped  | 0     |
| Xfailed  | 0     |
| Duration | 332.03 s |

## Phase 10 test modules

| Module | Covers |
| ------ | ------ |
| `tests/test_phase10_core.py` | configuration, structured logging, redaction, path traversal, filenames, subprocess, integrity (including deliberate corruption) |
| `tests/test_phase10_operations.py` | liveness contract, readiness states, request correlation, error envelope, metrics collection |
| `tests/test_phase10_timing.py` | stage timing, metric names, instrumentation does not change results, reproducibility |
| `tests/test_phase10_validation.py` | the deployment validator: OS-assigned ports, readiness gating, server-log capture on failure, guaranteed process cleanup |

## Bugs found and fixed during Phase 10

Each was found by a test or a validation check, not by inspection alone.

| # | Bug | Severity |
| --- | --- | --- |
| 1 | `ComponentStatus.usable` referenced instead of `.available`, making `/ready` raise | High |
| 2 | Redaction passed only field *values*, so a field named `api_key` leaked | High |
| 3 | The error envelope **replaced** `detail`, breaking 9 existing API tests | High |
| 4 | All 5xx messages masked, hiding actionable 503 guidance ("run `train_gnn.py`") | Medium |
| 5 | `safe_path` silently reinterpreted an absolute path as root-relative instead of refusing it | Medium |
| 6 | Validation script truncated `--help` output to its last 25 lines, but argparse prints `usage:` first, so every CLI reported "no usage line" | Medium |
| 7 | Validation probe used `/api/v1/ingestion/datasets`; the router is mounted at `/api/v1/data` | Low |
| 8 | `run_id` referenced in `run()` where no such parameter exists | Low |
| 9 | `COMPATIBLE` imported as an integrity state that does not exist | Low |
| 10 | A `pytest.skip` for unavailable symlinks; replaced with a capability branch so the suite reports 0 skipped | Low |
| 11 | The validation script's live probe requested the API immediately after spawning uvicorn, so a refused socket surfaced as `ECONNRESET` and was misread as a backend fault | High |

## The `ECONNRESET` failure: root cause

The CI-style validation failed with `ECONNRESET` / "the socket connection was
closed unexpectedly". It was diagnosed rather than retried away.

**It was a start-up race in the validation harness, not a backend
regression.** Reproduced deliberately with both patterns against the same
code:

| Pattern | Result |
| --- | --- |
| Spawn uvicorn, then request immediately | fails after ~3.3 s: `ConnectError [WinError 10061]` (the Windows form of the reported reset) |
| Spawn uvicorn, poll `/health` until it answers, then request | succeeds after ~15.3 s; every endpoint returns 200 |

The cause is import cost, not the API. `import extreme_weather.api.main`
takes ~15-36 s on this platform because PyTorch dominates, while a client that
does not wait gives up in ~3 s and sees a socket that is not listening yet.
The server log for the "failing" run shows
`Application startup complete` followed by 200s - it was starting the whole
time.

Ruled out as causes: the backend (every endpoint returns 200 both in-process
and over TCP), the readiness middleware (`/ready` answers `DEGRADED`, which is
correct for a checkout with no trained checkpoint), port management (the probe
now takes an OS-assigned port), and process lifecycle (cleanup is now
asserted).

The fix, in `scripts/validate_deployment.py`:

* the port is assigned by the OS, never hard-coded;
* the server's stdout **and** stderr are captured to a file;
* `/health` is polled until it answers, and only then is `/ready` consulted,
  so no request is ever issued against a half-started process;
* a server that dies during start-up is reported immediately **with its log**,
  so the failure says why instead of timing out;
* the process is terminated on exit and killed if it ignores that, so
  validation never leaves a listener behind.

The in-process ASGI probe is kept alongside it: the live probe proves the
deployment works, the in-process one is deterministic and isolates the
application. `tests/test_phase10_validation.py` pins all of the above.

## Network-dependent tests

One Phase 8 provider test calls the real Open-Meteo archive. It is **never**
skipped, xfailed or weakened. `scripts/validate_deployment.py --offline`
**deselects** it and prints a banner stating the run was a subset. The default
validation run includes it.
