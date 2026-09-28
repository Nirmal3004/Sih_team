# 06 - Validation results (SIH26078)

> The authoritative, machine-generated record is
> [`docs/FINAL_VALIDATION.md`](../FINAL_VALIDATION.md), produced by
> `python scripts/generate_final_validation.py` from actual command output.
> This page is the summary; the generated file is the evidence.

## Test suite

| Metric | Value |
| --- | --- |
| Total tests | **1825** |
| Passed | **1825** |
| Failed | 0 |
| Skipped | 0 |
| Xfailed | 0 |
| Errors | 0 |
| Phase 0–10 regressions | 0 |
| Phase 11 tests | 44 |

No test was skipped, xfailed or weakened in Phase 11. One pre-Phase-10 test
that used `pytest.skip` for unavailable symlinks was replaced with a capability
branch in Phase 10, which is why the skipped count is zero rather than
non-zero.

## What the Phase 11 tests cover

| Area | What is asserted |
| --- | --- |
| System status endpoint | 200 response, identity, readiness consistency with `/ready`, disclaimer always present, no accuracy key anywhere in the payload |
| Unavailable ≠ zero | Every count is `null` (not `0`) when nothing is stored; a corrupt store is unknown, not zero |
| Degraded state | A missing model is `UNAVAILABLE` with a reason and is never `READY`; alerting reflects its configuration switch |
| Safe serialization | No environment secret, no stack trace, no absolute data path, survives a corrupt run record |
| Configuration audit | Every documented env var is a real `Settings` field; `.env.example` commits no secret; no machine-specific absolute path |
| Security | Hostile path, overlong query, credential-shaped strings, error envelope hides internals, no wildcard CORS |
| Demo command | Dry run writes nothing; every required field is reported; `unavailable` prints instead of `0`; contradictory flags rejected |
| Reproducibility | Volatile fields stripped, identical records pass, differing records name the field, `NOT_APPLICABLE` is never a pass |
| Regression (2 bugs) | Orchestrator stores follow its own settings; the orchestrator persists `tracks.json` |

## Frontend, Docker and API

| Check | Result |
| --- | --- |
| `npm run build` | PASS — `tsc --noEmit` clean, vite built 159 modules |
| `docker compose config` | PASS — 2 services rendered |
| `GET /health` | 200 |
| `GET /ready` | 200, state reported explicitly |
| `GET /api/v1/system/status` | 200 |
| Deployment validation | 6/6 checks, 19 endpoints probed in-process *and* over TCP |
| Reproducibility | 11 passed, 0 failed, 0 not applicable |

## Demo pipeline

```bash
python scripts/run_sih_demo.py --dry-run          # PASS
python scripts/run_sih_demo.py --reuse-existing   # PASS
```

Both exit 0. The evaluation stage reports `UNAVAILABLE` with its reason in both
runs — that is the documented, correct outcome, not a failure.

## Three real defects found and fixed in Phase 11

The first two were found by the reproducibility check and the third by the
validation-record generator — none by inspection. All three are now covered by
regression tests.

| # | Defect | Severity | Symptom |
| --- | --- | --- | --- |
| 1 | `PipelineOrchestrator` accepted a `settings` argument but built its event, track and alert stores through services that defaulted to the process-wide `get_settings()` | High | A run pointed at a scratch directory wrote into the *repository's* data directory. `settings` was silently ignored for persistence. |
| 2 | `TrackingService.run()` defaults to `write_output=False`, and the orchestrator called it without that flag | High | Tracks were computed and reported in the run record, then discarded. `tracks.json` was never written, so `GET /api/v1/tracks` stayed empty after a successful run. |
| 3 | The reproducibility report right-aligned its verdict inside brackets (`[           PASS]`), so the validation generator matching `"[PASS]"` found nothing | Medium | `docs/FINAL_VALIDATION.md` reported "0 passed" for reproducibility — a number that reads exactly like a pass. |

The first two are the kind of defect a mocked test suite would have missed, and
both were caught precisely because the Phase 11 tooling ran the *real* pipeline
into a *real* isolated directory. The third is the most embarrassing of the
three, and it is the reason this project's own rule — that a validation record
must be generated from real output — exists.

## What is not validated

Stated plainly, because a validation record that implies more than it checked
is the failure mode this project exists to avoid:

- **No forecast skill.** No verified ground-truth warning catalogue was
  available, so no accuracy, precision, recall or F1 is reported. The
  evaluation reports these as unavailable.
- **No meteorological validation** of the GNN or downscaling models. They are
  trained on synthetic data and evaluated on reanalysis, which is not the same
  as being right.
- **No load, soak or concurrency testing.** The live probe checks a handful of
  sequential requests, not throughput.
- **No multi-process or HA test.** Run state is file-based; a second worker
  would race, and that is documented rather than tested.
- **GRIB is untested in this environment.** The optional backend is not
  installed; only its missing-dependency error path is exercised.
