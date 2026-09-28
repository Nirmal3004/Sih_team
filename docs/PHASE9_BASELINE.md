# Phase 9 baseline and regression record

Recorded before and after the Phase 9 work, per the Phase 9 stop conditions. No
pre-existing test was modified, weakened, skipped or xfailed to make Phase 9
pass.

## Pre-Phase-9 baseline (recorded before any Phase 9 change)

```bash
python -m pytest -q
```

| Metric   | Value |
| -------- | ----- |
| Total    | 1505  |
| Passed   | 1505  |
| Failed   | 0     |
| Skipped  | 0     |
| Xfailed  | 0     |
| Duration | 418.93 s |

That 1505 is the sum of the original 1105 tests
([PHASE7_BASELINE.md](PHASE7_BASELINE.md)), 185 Phase 7 tests and 215 Phase 8
tests.

## Phase 9 test modules

| Module | Covers |
| ------ | ------ |
| `tests/test_phase9_dataset.py` | chronological split, no overlap, no future leakage, train-only statistics, provenance, deterministic id |
| `tests/test_phase9_baselines.py` | persistence, constant velocity, GNN scoring, insufficient targets, multi-step rollout |
| `tests/test_phase9_uncertainty.py` | interval generation, bootstrap reproducibility, coverage, insufficient samples |
| `tests/test_phase9_explainability.py` | event evidence, missing evidence, no fabricated evidence, sensitivity, no causal language |
| `tests/test_phase9_downscaling.py` | identical test frames, baseline comparison, unavailable AI checkpoint, no test leakage |
| `tests/test_phase9_api.py` | status, summary, GNN, downscaling, uncertainty, explanation, run, run detail, 404, 422, dry run, CLI |
| `tests/phase9_support.py` | shared doubles and fixtures (not a test module) |

**Total Phase 9 tests: 138.**

## Final result

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
| Duration | 449.05 s |

1505 preserved + 138 new. **No Phase 0-8 regression.**

## Honest outcome on the real ERA5 record

The Phase 9 evaluation was run against the real ERA5 dataset produced by Phase 7
(`data/validation/processed/phase7_era5_processed.nc`, 336 hourly steps,
2024-06-17 to 2024-06-30) and its real tracked events
(`data/validation/reports/phase7_era5_tracks.json`).

The result is:

```text
run eval-029d6b32d4a44181  status=INSUFFICIENT_DATA
  dataset=eval-ceba449fc24696a7  checksum=ceba449fc24696a7
  [gnn]          NOT_AVAILABLE
  [uncertainty]  NOT_AVAILABLE
  [downscaling]  NOT_AVAILABLE
```

**This is the correct and expected outcome, not a failure.** The cause was
measured directly: the real ERA5 track graph contains **2 nodes spanning a
single timestamp**, of which 1 has a next-state target. A chronological
60/20/20 split of that cannot allocate a non-empty test block, so there is no
held-out next-state data on which any method - model or baseline - could be
scored.

This matches Phase 7's own recorded finding ("GNN score: not computed on this
record - too few test nodes"). The evaluation is reported as
`INSUFFICIENT_DATA` with that reason attached, and **not** as a zero.

The measurement machinery is exercised end to end by the deterministic
self-test (`--model fixture --sample-count 60`), which produces
`PARTIAL`/`SUCCESS` with measured baselines, uncertainty intervals and measured
coverage. The self-test is labelled `SYNTHETIC` in every output so it is never
mistaken for a result about real weather.
