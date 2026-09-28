# Live demo script (SIH26078) — 7 minutes

> **Say this in the first 20 seconds, and say it again at the end:** this is a
> research prototype. It shows no forecast skill, and no accuracy number is
> reported because none can honestly be computed.

Every claim below is supported by the repository. Where a number is not
available, the script says so — that is the demonstration, not a gap in it.

---

## 0:00–0:30 — Problem

> "Extreme weather losses are limited less by computation than by **timeliness
> and trust**. Public warnings are isolated text products with no
> machine-readable history, so they cannot be analysed or integrated. And
> dashboards routinely show metrics without provenance — where an unavailable
> number is quietly rendered as zero.
>
> Our goal was the pipeline *and* the honesty: a system that reports what it
> measured, and says `unavailable` where it did not."

**Do not** claim lives saved, economic benefit, or disaster prevention. We
measure neither.

---

## 0:30–1:15 — Architecture

Show `docs/sih/03_system_architecture.md`. Walk the chain left to right.

> "Ingestion and quality gate, preprocessing, anomaly detection against a
> climatology, event tracking, graph construction, a GNN prototype, AI
> downscaling, a rule-based alert engine, orchestration, evaluation, and the
> operations surface.
>
> Three design commitments. First: **unavailable is `null`, never `0`**.
> Second: **reuse over reimplementation** — the demo, the API and the CLIs all
> call the same orchestrator, so the demonstration cannot drift from the
> product. Third: every stored record carries a seed, a dataset checksum and a
> code fingerprint."

Point at the `[PROTOTYPE]` markers. Say the word "prototype" out loud.

---

## 1:15–2:00 — Data ingestion

```bash
python scripts/run_sih_demo.py --dry-run
```

> "First the dry run, because it is the honest opening: every stage executes,
> **nothing is written**, and the summary has the same shape as the real run."

Then the real run:

```bash
python scripts/run_sih_demo.py
```

> "72 time steps ingested. The data source is labelled at the top —
> `LOCAL FILE`. The `fixture` provider would say `DETERMINISTIC SYNTHETIC`;
> the Open-Meteo provider would say `ERA5 REANALYSIS … HISTORICAL`. The system
> never blurs those."

**If asked about the injected event:** it is deliberately injected into the
sample dataset so the detection stages have something real to find. It is
synthetic, and it is labelled synthetic.

---

## 2:00–2:45 — Event detection and tracking

Point at the summary block:

> "8 events detected, 1 track. Detection is a climatology-relative score with a
> per-variable threshold ladder, then connected-component grouping with a
> minimum-cell guard so a single noisy pixel is not an event.
>
> Tracking matches events across time steps by cost — a weighted blend of
> distance and intensity — assigned greedily with deterministic tie-breaking.
## 2:45–3:30 — GNN and downscaling

> "The graph: nodes are track positions, temporal edges connect consecutive
> positions, spatial edges connect nearby nodes across tracks. The split is
> **chronological** — train, then validation, then test — and it is asserted in
> code, not assumed.
>
> Now the part I want to be direct about. The GNN is a **prototype**. Trained
> on synthetic data, never validated against observed weather. On the available
> data it overfits visibly — training loss 0.0014 against test 64.56. We report
> that rather than hide it.
>
> The AI downscaler is also a prototype, and it **loses to cubic interpolation
> on MAE**. We compute the baseline on every run specifically so that
> comparison cannot be avoided."

Expect a question here. Answer it directly; it is the strongest thing in the
demo.

---

## 3:30–4:15 — Alerts

```bash
curl localhost:8000/api/v1/alerts
```

> "Alerts come from a rule-based risk score over intensity, proximity, speed and
> persistence. Each has a lifecycle: new, active, acknowledged, resolved, with
> a terminal expired state and time-bucket deduplication.
>
> One thing to be very clear about: **these severities are this project's own
> categories. They are not official IMD or WMO warnings** and must not be used
> for emergency decisions. The API says so in its own response, so nobody can
> lift a severity out of this system without the caveat."

---

## 4:15–5:00 — Evaluation

Point at the demo output:

```text
Evaluation status : UNAVAILABLE
```

> "This is the most important line in the output. No verified ground-truth
> warning catalogue exists for this deployment, so **no accuracy, precision,
> recall or F1 is reported**. The system prints `unavailable` and the reason.
>
> A system that showed `0% accuracy` here would be making a claim it cannot
> support — and a reader would have no way to tell the difference.
>
> What *is* measured: event, track and alert counts with their ids; graph
> structure; stage durations; and baseline comparisons on a chronological test
> split where a reference exists."

---

## 5:00–6:00 — Dashboard and API

```bash
cd frontend && npm run dev
```

Open the dashboard. Then:

```bash
curl localhost:8000/api/v1/system/status
```

> "The dashboard opens on system status. Readiness badge, the data source and
> its timestamp, each component with the reason it is unavailable, and the
> stored counts — where anything unknown reads `unavailable` rather than `0`.
>
> The prototype banner is on every page.
>
> `/health` is liveness and never touches a model. `/ready` is readiness and
> reports `READY`, `DEGRADED` or `NOT_READY`. They are deliberately different:
> a liveness probe that failed on a missing model would restart-loop a
> perfectly healthy process."

Then show reproducibility:

```bash
python scripts/verify_reproducibility.py
```

> "Two independent runs, identical seeded inputs. Event ids, track ids, alert
> ids, graph structure, provenance and both run ids are identical. Timestamps
> and durations are excluded — they are *meant* to differ. Eleven checks, all
> passing."

---

## 6:00–7:00 — Limitations and future scope

> "What this system does **not** do, plainly:
>
> - **No forecast skill is demonstrated.** No ground-truth catalogue existed for
>   any dataset here. That is the central limitation.
> - **The data is not live.** Synthetic fields and ERA5 reanalysis. Reanalysis
>   is a reconstruction of the past carrying model error.
> - **There is no authentication.** It needs an authenticating reverse proxy on
>   any untrusted network.
> - **The models are prototypes**, and the downscaler does not beat cubic
>   interpolation. Both are reported, not concealed.
> - **The real-data validation is tiny**: three days, a 3×3 grid, one region.
>
> What would change the picture, in order: acquiring a verified warning
> catalogue so skill can be measured at all; multi-region, multi-season data;
> and only then, model work. The engineering foundation is done. The
> measurement is what is missing, and we would rather say so than fill the gap
> with a number we cannot defend."

**Close on that.** The gap is named, not hidden — which is the whole argument
of the project.

---

## If something breaks

| Symptom | Say this |
| --- | --- |
| `ECONNRESET` on first request | "The app was still importing — 15 to 36 seconds, PyTorch. Our validator polls `/health` before requesting, which is the fix we shipped." |
| Dashboard unreachable | "Backend isn't running, or `VITE_API_BASE_URL` is wrong." |
| Models UNAVAILABLE | "Expected with no checkpoint. It reports SKIPPED with a reason rather than substituting a default." |
| `NOT READY` | "`GET /ready` → `components` → `directories` → `notes`. Usually an unwritable data directory." |

Every one of these is a real limitation with a documented cause. Say the cause.

> That is why the ids are reproducible."

**If asked how good the detection is:** we cannot say, and that is the point.
There is no ground truth.
