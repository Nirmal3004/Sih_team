# 02 - Solution overview (SIH26078)

## One sentence

A reproducible, end-to-end pipeline that ingests weather data, detects extreme
events, tracks them, applies graph and convolutional prototypes, and serves
honest, provenance-carrying results through an API and dashboard.

## What it actually does

| Stage | Input → Output | Prototype? |
| --- | --- | --- |
| Ingestion | provider payload → Xarray `Dataset` + manifest with SHA-256 | No |
| Quality gate | freshness, variable whitelist, retry | No |
| Preprocessing | validate → gap fill (max_gap) → derived wind fields | No |
| Anomaly detection | climatology + score + threshold ladder → events | No |
| Event tracking | events → trajectories with lifecycle | No |
| Graph construction | tracks → nodes/edges, chronological split | No |
| GNN | graph → next-state prediction for a tracked event | **Yes** |
| AI downscaling | ~12 km field → ~5 km field | **Yes** |
| Alerting | events + tracks → risk score, rule evaluation, lifecycle | **Yes** (severities are project categories) |
| Orchestration | provider → run record, idempotent, stage-level status | No |
| Evaluation | runs → MEASURED / UNAVAILABLE, with provenance | No |
| Operations | health, readiness, metrics, integrity | No |

## Three design commitments

**1. Unavailable is `null`, never `0`.**

"No trained checkpoint" and "a checkpoint that scored zero" are different
facts. Every count, metric and status in the API and the dashboard distinguishes
them, and a quantity that could not be measured carries the reason it could
not. This is enforced by tests, not by convention.

**2. Reuse over reimplementation.**

The demo command, the API and the individual CLIs all call the same services.
`scripts/run_sih_demo.py` drives `PipelineOrchestrator` rather than
re-implementing eight stages, so the demonstration cannot drift away from the
product. Two real defects were found precisely because of this: the
orchestrator was writing its artifacts to the wrong directory, and it was
discarding the tracks it had just computed.

**3. Every result is traceable.**

Each stored record carries a seed, a dataset checksum, split boundaries and a
code fingerprint. `scripts/verify_reproducibility.py` runs the pipeline twice
and proves that identical inputs produce identical artifacts, excluding only
the fields that are genuinely meant to differ.

## What you can see in five minutes

```bash
pip install -r requirements.txt
python scripts/generate_sample_data.py
python scripts/run_sih_demo.py --dry-run     # writes nothing
python scripts/run_sih_demo.py               # the real run
python -m uvicorn extreme_weather.api.main:app --port 8000
curl localhost:8000/api/v1/system/status    # the single best request
```

The last one tells you what the system is, what data it holds, which models
actually load, and how much it has produced — with `null` wherever a quantity
is unknown.

## Where this stops

At a working, honest foundation. It is not an operational warning system, and
`07_limitations.md` lists every reason in detail.
