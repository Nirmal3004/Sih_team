# 07 - Limitations (SIH26078)

Every item here is a real constraint on what this system can be used for. They
are listed so a reader can decide what the system is **not** suitable for,
which is the only honest basis on which to evaluate it.

## The three that matter most

**1. No forecast skill is demonstrated.**
No verified ground-truth warning catalogue was available for any dataset this
project touches. Without one, accuracy, precision, recall and F1 cannot be
computed, and this project reports them as `unavailable` rather than as zero.
The system demonstrates an engineering pipeline, not predictive skill.

**2. The data is not live.**
Sources are locally generated synthetic fields and ERA5 *reanalysis*.
Reanalysis is a reconstruction of the past that carries model error. Nothing
here ingests a live feed, and "near-real-time" describes the orchestration
loop, not the data.

**3. There is no authentication.**
The API is unauthenticated, as it has been since Phase 0. It requires an
authenticating reverse proxy on any untrusted network. See `docs/SECURITY.md`.

## Model limitations

**GNN prototype.** Trained on synthetic data, never validated against observed
weather. On the available data it overfits visibly — train loss 0.0014 against
validation 5.88 and test 64.56. The sample is small (one track, four training
samples), so this is a demonstration of the mechanism, not of capability.

**AI downscaling prototype.** Produces a finer grid, which is not the same as
producing a more accurate field. On the available data it **loses to cubic
interpolation on MAE**. The baseline is computed alongside every run precisely
so this comparison cannot be avoided.

**Both are optional.** With no trained checkpoint, both stages report `SKIPPED`
with the reason. No default, fallback or placeholder prediction is ever
substituted.

## Data limitations

| Limitation | Detail |
| --- | --- |
| Small real-data sample | Three days, a 3×3 grid (~53 km), one region |
| Single region | No multi-region or multi-basin evaluation |
| GRIB untested here | The optional backend is not installed; only its missing-dependency error path is exercised |
| Resolution is small | 0.125° (~12 km) input, 0.05° (~5 km) target — far coarser than an operational NWP grid |
| Synthetic data is smooth by construction | The `fixture` provider contains no extremes, so it detects zero events. That is correct behaviour, not a bug. |

## Alerting limitations

The severities are **this project's own categories**, not official IMD or WMO
warnings. The risk score is a rule-based triage aid, not a validated
disaster-risk index. Notifications are appended to a local log; nothing is
delivered to any external service. There is no paging, no escalation and no
on-call workflow.

## Operational limitations

| # | Limitation |
| --- | --- |
| 1 | No authentication or authorisation |
| 2 | Single process. Run history and incremental state are files; two workers would race. There is no distributed lock. |
| 3 | No HA, failover or backup scheduling |
| 4 | No SLA. Nothing here is monitored or paged. |
| 5 | Polling, not push. The Phase 8 scheduler polls; there is no streaming ingest. |
| 6 | Metrics are process-local and lost on restart |
| 7 | Import cost is high on this platform (~15–36 s, PyTorch), so cold start is slow |
| 8 | CORS is a browser control; it does not authenticate a non-browser client |
| 9 | No dependency scanning is configured; that belongs in CI |

## What is explicitly not claimed

- No claim of lives saved, injuries avoided or economic benefit.
- No guarantee of evacuation, preparedness or disaster prevention.
- No claim that the alert severities correspond to any official warning level.
- No claim that the AI downscaler improves accuracy over interpolation.
- No claim that the GNN predicts real storm evolution.

The strongest true statement about this system is: **it is a reproducible,
honest engineering foundation for extreme-weather intelligence, with its
prototype components clearly marked as prototypes.**
