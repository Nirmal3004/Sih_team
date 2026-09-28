# Final validation (SIH26078, Phase 11)

> **Generated file.** Every number below was produced by running the
> command shown against this repository. Nothing here is typed by hand.
>
> Regenerate with `python scripts/generate_final_validation.py`.

**This project is a research prototype.** It demonstrates no forecast
skill, and no accuracy, lives-saved or economic-benefit figure has been
measured. Where a quantity could not be computed it is reported as
unavailable, never as zero.

Generated: 2026-09-27 12:58:53 UTC

## Summary

**9/9 checks passed.**

| Check | Result | Detail |
| --- | --- | --- |
| pytest suite | **PASS** | 1825 passed (total 1825) in 254s |
| Phase 11 tests | **PASS** | 44 test(s) in tests/test_phase11_system.py |
| frontend build | **PASS** | tsc --noEmit + vite build completed |
| docker compose config | **PASS** | rendered successfully (3 service(s)) |
| API health / readiness | **PASS** | /health 200, /ready 200 (READY), /api/v1/system/status 200 |
| reproducibility | **PASS** | 11 passed, 0 failed, 0 not applicable |
| demo pipeline (dry run) | **PASS** | exit 0; 7 stage(s) OK, 1 unavailable (documented, not failures) |
| demo pipeline (live) | **PASS** | exit 0; 7 stage(s) OK, 1 unavailable (documented, not failures) |
| deployment validation | **PASS** | 8 check(s) passed |

## Test counts

| Metric | Value |
| --- | --- |
| Total tests | 1825 |
| Passed | 1825 |
| Failed | 0 |
| Skipped | 0 |
| Xfailed | 0 |
| Errors | 0 |

Skipped and xfailed are reported as they are. A non-zero count in either column would mean a test had been suppressed, which this project does not do.

## Reproducibility

11 passed, 0 failed, 0 not applicable.

Two independent pipeline runs with identical seeded inputs produce identical event ids, track ids, alert ids, graph structure, provenance and run ids. Timestamps and durations are excluded because they are intentionally volatile.

## Notes

No check failed. A `UNAVAILABLE` demo stage (for example the evaluation, when no ground-truth reference exists) is a documented outcome, not a failure, and is reported as such rather than being converted into a zero.
