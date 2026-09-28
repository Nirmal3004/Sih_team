# Explainability (SIH26078, Phase 9 - IMPLEMENTED)

> **Status: IMPLEMENTED.** `src/extreme_weather/evaluation/explainability.py`
> and `GET /api/v1/evaluation/explain/{event_id}`.
>
> **Research prototype.** An explanation here describes what the pipeline
> recorded and what a model consumed. It is not an account of why anything
> happened in the atmosphere.

## The one rule

**Report what was used, never what caused anything.**

A correlation between a feature and a prediction is not a cause. Neither is a
perturbation experiment: changing an input and watching the output move tells
you about the fitted function, not about the world.

This module therefore uses the words *supporting evidence* and *input
sensitivity*, and never *cause*, *driver* or *reason*. The distinction is
enforced three ways:

1. a fixed, non-optional `CAUSALITY_DISCLAIMER` is attached to every explanation;
2. evidence is typed `supporting` / `context` / `unavailable`;
3. the test suite searches **every generated string** for causal phrasing
   (`"caused by"`, `"root cause"`, `"responsible for"`, ...) and fails if any
   appears.

## What counts as evidence

| Type | Meaning | Example |
| --- | --- | --- |
| `supporting` | A model consumed it as input | anomaly score, GNN prediction |
| `context` | A recorded fact, consumed by no model | track direction, data freshness |
| `unavailable` | Sought, and does not exist | GNN prediction with no checkpoint |

The distinction between `supporting` and `context` matters: a compass direction
is genuinely useful to a human reading an alert, but no model in this project
consumed it, so presenting it as a model input would overstate what the
pipeline used.

## Never fabricated

A field the event does not carry produces an explicit `unavailable` entry naming
what was missing. It is never filled with a plausible value, a zero, or an
omission that leaves the reader thinking nothing was sought.

Concretely:

| Situation | Result |
| --- | --- |
| No track matched | `track_speed_kmh`, `track_direction`, `persistence_hours` all `unavailable` - **not** 0 km/h |
| No GNN checkpoint | `gnn_prediction` `unavailable` - **not** a predicted coordinate |
| No downscaling checkpoint | `downscaled_impact_field` `unavailable` - nothing simulated in its place |
| No climatological reference | Percentile evidence is **not emitted**, and the absence is stated as a limitation |

## Event evidence

```text
GET /api/v1/evaluation/explain/{event_id}
```

```json
{
  "event_id": "EXTREME_RAINFALL_20240630T1200Z_0",
  "evidence": [
    {
      "feature": "intensity",
      "value": 55.0,
      "unit": "mm h-1",
      "context": "Peak physical value inside the event, in the source variable's unit.",
      "contribution_type": "supporting",
      "available": true,
      "source": "phase1.anomaly"
    }
  ],
  "available_evidence": ["..."],
  "unavailable_evidence": ["..."],
  "evidence_counts": { "available": 9, "unavailable": 4 },
  "limitations": ["..."],
  "disclaimer": "These items are supporting evidence: ... not causal evidence ..."
}
```

Evidence assembled, when it exists:

| Feature | Source | Type |
| --- | --- | --- |
| `anomaly_score` | Phase 1 | supporting |
| `intensity` | Phase 1 | supporting |
| `affected_cell_count` | Phase 1 | supporting |
| `affected_radius_km` | Phase 1 | supporting |
| `severity` | Phase 5 | supporting |
| `track_speed_kmh`, `track_max_speed_kmh` | Phase 2 | supporting |
| `persistence_hours` | Phase 2 | supporting |
| `track_direction` | Phase 2 | **context** |
| `intensity_trend` | Phase 2 | **context** |
| `predicted_*` | Phase 3 | supporting, when a model ran |
| `downscaled_impact_field` | Phase 4 | supporting, when available |
| `data_freshness` | Phase 8 | **context** |

**Stale data is a caveat on everything else.** When the freshness verdict is
`STALE` or `EXPIRED`, that is shown as context, because it weakens every
quantitative item alongside it.

## Alert evidence

`build_alert_explanation` reuses the Phase 5 `ContributingFactor` terms where
present. They already carry the raw value, the weight and the points
contributed, so a reader can check the arithmetic rather than take the score on
trust.

Always reported, either way:

* each risk factor's points, weight, normalised value and detail;
* `source_components` - exactly which pipeline stages contributed. A degraded
  run says so rather than looking complete;
* the predicted values, or an explicit absence when no model ran;
* an explanation that the score is a **rule-based triage aid**, not a validated
  disaster-risk index;
* an explanation that **no skill score exists**, because no verified
  ground-truth warning catalogue is available - and that *absence of evidence of
  error is not evidence of accuracy*.

An alert with no risk assessment reports `risk_score: unavailable` rather than an
invented score.

## Feature sensitivity

`sensitivity_report` perturbs one node feature at a time and measures how far the
prediction moves.

```json
{
  "method": "one-at-a-time input perturbation",
  "reported_as": "input_sensitivity",
  "experiments": [
    {
      "feature": "intensity_scaled",
      "baseline_value": 0.55,
      "perturbed_value": 0.62,
      "prediction_delta": [0.011, 0.004],
      "delta_magnitude": 0.0117,
      "direction": "increases_output",
      "perturbed": true
    }
  ],
  "ranked_by_response": ["intensity_scaled"],
  "interpretation": "These deltas are the model's response to a changed input. They are a property of the fitted function, not evidence that any feature causes the physical event..."
}
```

`delta` is **perturbed minus baseline**: how far the prediction *moved* in
response to the input change.

If a named feature is not a model input, the entry says so and is marked
`perturbed: false`. Reporting `delta: 0` there would read as "this feature does
not matter", which is a different and unsupported claim.

## Dashboard

`ExplainabilityPanel` renders the same three groups the API returns:

* **supporting evidence**, with its type and the stage that recorded it;
* **evidence not available**, listed as prominently as what exists;
* **limitations**, never collapsed.

The disclaimer is rendered at the top, not only in the footer, so the caveat
cannot be scrolled past.

## Responsible interpretation

A good reading:

> "The event was detected with a peak intensity of 55 mm h-1, spanned 18 cells
> over a 42 km radius, and was tracked for 3 hours moving north-east at about
> 12 km/h. No GNN prediction is attached because no checkpoint was available."

A reading this project explicitly rejects:

> "The high intensity caused the alert."

Nothing in this output supports a causal statement, and the API is designed so
that no consumer can assemble one from it without going beyond what was
measured.

## Limitations

* Evidence is **descriptive**, not diagnostic. A high anomaly score is recorded,
  not explained.
* Evidence from a later stage cannot be read as having influenced an earlier
  one: the pipeline runs in one direction.
* No percentile or standardized-anomaly evidence exists, because no
  climatological reference period long enough to compute one is available.
* Feature sensitivity is a property of one fitted model on one dataset. It does
  not transfer to another checkpoint, and it says nothing about the atmosphere.
* Severity and risk scores are project-internal categories.
