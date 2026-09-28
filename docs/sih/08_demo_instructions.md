# 08 - Demo instructions (SIH26078)

## Before you start

You need Python 3.11+ and about 2 GB of disk. **No API key. No network
access.** The default demo runs entirely on locally generated data.

```bash
git clone <repository> && cd deepweather-ai
python -m venv .venv
# Windows:  .venv\Scripts\activate
# POSIX:    source .venv/bin/activate
pip install -r requirements.txt
python scripts/generate_sample_data.py
```

Generating the sample data takes under a minute and creates
`data/raw/sample_history.nc`, a synthetic dataset with **one deliberately
injected extreme event**. That injection is what gives the detection, tracking
and alerting stages something to show.

## The three commands that matter

```bash
# 1. Prove the pipeline works without writing anything
python scripts/run_sih_demo.py --dry-run

# 2. Run the real demonstration
python scripts/run_sih_demo.py

# 3. See the whole system in one request
python -m uvicorn extreme_weather.api.main:app --port 8000
#   then, in another terminal:
curl localhost:8000/api/v1/system/status
```

## What the judge should see

`--dry-run` first, because it is the honest opening: every stage runs, nothing
is written, and the summary is identical in structure to the real run.

Then the real run, which produces on the bundled sample:

```text
Pipeline summary
  Ingestion status          : run run-3124830627d8af4f: 8 stages co...
  Weather steps             : 72
  Detected events           : 8
  Event tracks              : 1
  GNN availability          : yes
  Downscaling availability  : yes
  Alerts generated          : 8
  Pipeline run ID           : run-3124830627d8af4f
  Evaluation status         : UNAVAILABLE
  Validation status         : READY
```

**Point at `Evaluation status: UNAVAILABLE` and explain it.** That line is the
most important thing in the output. No ground-truth reference exists, so the
system reports nothing rather than reporting a zero. A system that showed `0%
accuracy` here would be making a claim it cannot support.

## With the dashboard

```bash
cd frontend && npm install && npm run dev     # http://localhost:5173
```

With the backend running, the dashboard opens on **System status**: readiness
badge, data source and timestamp, per-component availability with reasons, and
the stored counts. The prototype banner is at the top of the page.

## With Docker

```bash
docker compose up --build     # backend :8000, frontend :5173
```

The frontend waits for the backend's liveness probe before starting, so the
dashboard never loads against a backend that cannot serve.

## Proving it is reproducible

```bash
python scripts/verify_reproducibility.py
```

It runs the pipeline twice into two scratch directories and reports
`PASS`/`FAIL`/`NOT_APPLICABLE` per check. A correct run reports **11 passed, 0
failed, 0 not applicable** — event ids, track ids, alert ids, the graph
structure, the provenance metadata and both run ids are identical, while
timestamps and durations are excluded because they are meant to differ.

## If something goes wrong

| Symptom | Fix |
| --- | --- |
| `ECONNRESET` on first request | The app was still importing (15–36 s, PyTorch). Poll `/health` first. |
| Dashboard shows "Backend unreachable" | Start uvicorn, or set `VITE_API_BASE_URL`. |
| Every panel errors but the page loads | Add the browser origin to `CORS_ORIGINS`. |
| Models report UNAVAILABLE | Expected with no checkpoint. Run the training scripts, or explain why it is correct. |
| `NOT READY` | `GET /ready` → `components` → `directories` → `notes`. |

Full troubleshooting: `docs/SIH_DEMO_GUIDE.md` §11.
