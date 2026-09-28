"""Explainability for Phase 9 (SIH26078, sections 8-10).

The governing rule of this module: **report what the model used, never what
caused anything.**

A correlation between a feature and a prediction is not a cause. Neither is a
perturbation experiment: changing an input and observing the output move tells
you about the model's *sensitivity*, which is a property of the function, not
of the world. This module therefore:

* calls every item **evidence** or **input sensitivity**, never "cause",
  "driver" or "reason";
* attaches a fixed, non-optional caveat to every explanation;
* **never invents evidence**. An unavailable field becomes an explicit
  ``unavailable`` entry naming what was missing, never a plausible-looking
  value;
* distinguishes ``supporting`` evidence (the model consumed it) from
  ``context`` (a human-readable fact about the event that no model consumed).

Feature sensitivity
-------------------
:meth:`sensitivity_report` perturbs one node feature at a time and measures how
much the prediction moves. It reports a **delta in the model's output**, and
labels the result input sensitivity. Where a feature cannot be perturbed safely
- because it is an identifier, or discrete - the entry says so instead of
producing a meaningless number.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np

from extreme_weather.evaluation.models import utc_now

__all__ = [
    "CAUSALITY_DISCLAIMER",
    "EVIDENCE_UNAVAILABLE",
    "SUPPORTING",
    "CONTEXT",
    "Evidence",
    "EventExplanation",
    "AlertExplanation",
    "build_alert_explanation",
    "build_event_explanation",
    "sensitivity_report",
]

UTC = timezone.utc

#: The evidence category for input a model actually consumed.
SUPPORTING = "supporting"

#: The evidence category for a fact recorded for a human reader, which no
#: model in this project consumed as input.
CONTEXT = "context"

#: The marker used when a piece of evidence does not exist.
EVIDENCE_UNAVAILABLE = "unavailable"

#: Attached to every explanation. Not optional, and not configurable away.
CAUSALITY_DISCLAIMER = (
    "These items are supporting evidence: the quantities the pipeline recorded "
    "or the model consumed as input. They are not causal evidence. This project "
    "does not establish that any of them caused the event, the alert, or the "
    "prediction, and no such relationship should be inferred from their "
    "correlation with the outcome."
)


@dataclass(frozen=True)
class Evidence:
    """One piece of evidence for an event, alert or prediction.

    Attributes:
        feature: Stable identifier, e.g. ``intensity`` or ``track_speed_kmh``.
        value: The value, or ``None`` when it does not exist.
        unit: Physical unit of ``value``.
        context: One sentence a human can act on.
        contribution_type: ``supporting``, ``context`` or ``unavailable``.
        source: Which component produced it.
    """

    feature: str
    value: float | str | None
    context: str
    contribution_type: str = SUPPORTING
    unit: str | None = None
    source: str = ""

    @property
    def available(self) -> bool:
        """True when a real value backs this item."""
        return self.contribution_type != EVIDENCE_UNAVAILABLE and self.value is not None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "feature": self.feature,
            "value": self.value,
            "unit": self.unit,
            "context": self.context,
            "contribution_type": self.contribution_type,
            "available": self.available,
            "source": self.source,
        }


def _number(value: Any) -> float | None:
    """Return a finite float, or ``None``."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _available(
    feature: str,
    value: Any,
    unit: str | None,
    context: str,
    source: str,
) -> Evidence:
    """Return an evidence item when the value exists, else an explicit gap."""
    numeric = _number(value)
    if numeric is None:
        return Evidence(
            feature=feature,
            value=None,
            unit=unit,
            context=f"{feature} is not available for this event: the pipeline did not record it.",
            contribution_type=EVIDENCE_UNAVAILABLE,
            source=source,
        )
    return Evidence(
        feature=feature,
        value=numeric,
        unit=unit,
        context=context,
        contribution_type=SUPPORTING,
        source=source,
    )


def _context_item(
    feature: str, value: Any, context: str, source: str
) -> Evidence:
    """Return a context item (recorded, but not a model input)."""
    if value is None or value == "":
        return Evidence(
            feature=feature,
            value=None,
            context=f"{feature} is not recorded for this event.",
            contribution_type=EVIDENCE_UNAVAILABLE,
            source=source,
        )
    return Evidence(
        feature=feature,
        value=value if isinstance(value, str) else _number(value),
        context=context,
        contribution_type=CONTEXT,
        source=source,
    )


@dataclass
class EventExplanation:
    """The evidence behind one detected event.

    Attributes:
        event_id: The Phase 1 event identifier.
        evidence: Every item, including the unavailable ones.
        limitations: What this explanation cannot say.
        generated_at: When the explanation was built.
    """

    event_id: str
    evidence: list[Evidence] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=utc_now)

    @property
    def available_evidence(self) -> list[Evidence]:
        """Only the items backed by a real value."""
        return [item for item in self.evidence if item.available]

    @property
    def unavailable_evidence(self) -> list[Evidence]:
        """Only the gaps, so a UI can show what is missing."""
        return [item for item in self.evidence if not item.available]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "event_id": self.event_id,
            "evidence": [item.as_dict() for item in self.evidence],
            "available_evidence": [item.as_dict() for item in self.available_evidence],
            "unavailable_evidence": [
                item.as_dict() for item in self.unavailable_evidence
            ],
            "evidence_counts": {
                "available": len(self.available_evidence),
                "unavailable": len(self.unavailable_evidence),
            },
            "limitations": list(self.limitations) + [CAUSALITY_DISCLAIMER],
            "disclaimer": CAUSALITY_DISCLAIMER,
            "generated_at": self.generated_at.isoformat(),
        }


def build_event_explanation(
    event: Any,
    *,
    track: Any | None = None,
    gnn_prediction: Any | None = None,
    downscaling_available: bool | None = None,
    data_freshness: str | None = None,
) -> EventExplanation:
    """Assemble the evidence for one detected event.

    Every field is optional. A field the event does not carry produces an
    explicit ``unavailable`` entry rather than being omitted, so a reader can
    see that the evidence was sought and not found.

    Args:
        event: A Phase 1 :class:`ExtremeWeatherEvent`.
        track: The Phase 2 track containing the event, when tracked.
        gnn_prediction: The Phase 3 prediction, when a model was available.
        downscaling_available: Whether an AI downscaled field exists.
        data_freshness: The freshness verdict of the data behind the event.

    Returns:
        An :class:`EventExplanation`.
    """
    items: list[Evidence] = []
    source = "phase1.anomaly"

    anomaly = _number(getattr(event, "anomaly_score", None))
    items.append(
        Evidence(
            feature="anomaly_score",
            value=anomaly,
            unit="1",
            context=(
                f"Peak anomaly score inside the event, on the scale of the "
                f"detector that produced it."
                if anomaly is not None
                else "anomaly_score is not available for this event."
            ),
            contribution_type=SUPPORTING if anomaly is not None else EVIDENCE_UNAVAILABLE,
            source=source,
        )
    )
    items.append(
        _available(
            "intensity",
            getattr(event, "intensity", None),
            getattr(event, "units", None),
            "Peak physical value inside the event, in the source variable's unit.",
            source,
        )
    )
    items.append(
        _available(
            "affected_cell_count",
            getattr(event, "cell_count", None),
            "cells",
            "Number of grid cells that passed the detection threshold.",
            source,
        )
    )
    items.append(
        _available(
            "affected_radius_km",
            getattr(event, "affected_radius_km", None),
            "km",
            "Great-circle distance from the centroid to the farthest detected cell.",
            source,
        )
    )
    items.append(
        _available(
            "severity",
            getattr(event, "severity", None),
            None,
            "Severity category assigned by the deterministic rule set. A project "
            "prototype category, not an official warning level.",
            "phase5.alerting",
        )
        if getattr(event, "severity", None) is not None
        else Evidence(
            feature="severity",
            value=None,
            context="No severity was assigned to this event.",
            contribution_type=EVIDENCE_UNAVAILABLE,
            source="phase5.alerting",
        )
    )

    if track is not None:
        track_source = "phase2.tracking"
        items.append(
            _available(
                "track_speed_kmh",
                getattr(track, "mean_speed_kmh", None),
                "km/h",
                "Mean movement speed along the matched trajectory.",
                track_source,
            )
        )
        items.append(
            _available(
                "track_max_speed_kmh",
                getattr(track, "max_speed_kmh", None),
                "km/h",
                "Fastest movement speed observed along the trajectory.",
                track_source,
            )
        )
        items.append(
            _context_item(
                "track_direction",
                getattr(track, "direction", None),
                "Compass direction of travel, from the matched trajectory.",
                track_source,
            )
        )
        items.append(
            _available(
                "persistence_hours",
                getattr(track, "duration_hours", None),
                "h",
                "Time between the first and last matched detection, which is how "
                "long the event persisted as one track.",
                track_source,
            )
        )
        items.append(
            _context_item(
                "intensity_trend",
                getattr(track, "intensity_trend", None),
                "Whether the event strengthened, weakened or held steady along "
                "the trajectory.",
                track_source,
            )
        )
    else:
        for feature, context in (
            ("track_speed_kmh", "No track was matched to this event, so no movement evidence exists."),
            ("track_direction", "No track was matched to this event, so no direction evidence exists."),
            (
                "persistence_hours",
                "No track was matched to this event, so its persistence is unknown.",
            ),
        ):
            items.append(
                Evidence(
                    feature=feature,
                    value=None,
                    context=context,
                    contribution_type=EVIDENCE_UNAVAILABLE,
                    source="phase2.tracking",
                )
            )

    if gnn_prediction is not None:
        gnn_source = "phase3.gnn"
        for field_name, unit in (
            ("predicted_latitude", "deg"),
            ("predicted_longitude", "deg"),
            ("predicted_intensity", None),
        ):
            value = _number(getattr(gnn_prediction, field_name, None))
            items.append(
                Evidence(
                    feature=field_name,
                    value=value,
                    unit=unit,
                    context=(
                        "One-step model prediction for this event. It is an "
                        "extrapolation from a prototype model, not a forecast."
                        if value is not None
                        else f"{field_name} is not available for this prediction."
                    ),
                    contribution_type=SUPPORTING if value is not None else EVIDENCE_UNAVAILABLE,
                    source=gnn_source,
                )
            )
    else:
        items.append(
            Evidence(
                feature="gnn_prediction",
                value=None,
                context=(
                    "No GNN prediction is attached to this event. Either no model "
                    "checkpoint was available, or the run produced no usable graph "
                    "node. No prediction was invented."
                ),
                contribution_type=EVIDENCE_UNAVAILABLE,
                source="phase3.gnn",
            )
        )

    if downscaling_available is True:
        items.append(
            Evidence(
                feature="downscaled_impact_field",
                value="available",
                context="An AI-downscaled impact field exists for this period.",
                contribution_type=SUPPORTING,
                source="phase4.downscaling",
            )
        )
    else:
        items.append(
            Evidence(
                feature="downscaled_impact_field",
                value=None,
                context=(
                    "No AI-downscaled impact field is available. Either no "
                    "checkpoint was trained or the evaluation reported "
                    "insufficient data. Nothing was simulated in its place."
                ),
                contribution_type=EVIDENCE_UNAVAILABLE,
                source="phase4.downscaling",
            )
        )

    if data_freshness is not None:
        items.append(
            Evidence(
                feature="data_freshness",
                value=str(data_freshness),
                context=(
                    "Freshness of the weather data behind this event at the time "
                    "it was recorded. Stale or expired data is a caveat on every "
                    "other item in this explanation."
                ),
                contribution_type=CONTEXT,
                source="phase8.realtime",
            )
        )

    limitations = [
        "Percentile and standardized-anomaly evidence is not reported: this "
        "project has no climatological reference period long enough to compute a "
        "percentile honestly, and a standardized anomaly without one would be "
        "meaningless.",
        "The anomaly score is on the detector's own scale and is not comparable "
        "across detectors, datasets or versions.",
        "Evidence recorded by a later stage cannot be read as having influenced "
        "an earlier stage: the pipeline runs in one direction.",
    ]
    if track is None:
        limitations.append(
            "No track was matched to this event, so movement and persistence "
            "evidence are absent rather than zero."
        )

    return EventExplanation(
        event_id=str(getattr(event, "event_id", "unknown")), evidence=items, limitations=limitations
    )


def build_alert_explanation(
    alert: Any,
    *,
    event: Any | None = None,
    track: Any | None = None,
    data_freshness: str | None = None,
) -> "AlertExplanation":
    """Assemble the risk evidence behind one alert.

    The alert's own :class:`~extreme_weather.alerts.models.ContributingFactor`
    terms are reused where present, because they already carry the raw value,
    the weight and the points contributed - the arithmetic can be checked by a
    reader rather than taken on trust.

    Args:
        alert: A Phase 5 :class:`Alert`.
        event: The Phase 1 event behind the alert, when available.
        track: The Phase 2 track, when available.
        data_freshness: Freshness of the data behind the alert.

    Returns:
        An :class:`AlertExplanation`.
    """
    items: list[Evidence] = []

    assessment = getattr(alert, "risk_assessment", None)
    factors = list(getattr(assessment, "contributing_factors", ()) or ()) if assessment else []
    if factors:
        for factor in factors:
            items.append(
                Evidence(
                    feature=f"risk_factor:{getattr(factor, 'name', 'unknown')}",
                    value=_number(getattr(factor, "points", None)),
                    unit="points",
                    context=(
                        f"{getattr(factor, 'label', factor.name)}: "
                        f"{getattr(factor, 'detail', '')} "
                        f"(weight {getattr(factor, 'weight', 0.0)}, "
                        f"normalised value {getattr(factor, 'normalized_value', 0.0)})."
                    ),
                    contribution_type=SUPPORTING,
                    source="phase5.alerting",
                )
            )
    else:
        items.append(
            Evidence(
                feature="risk_score",
                value=None,
                context=(
                    "This alert carries no risk assessment, so no risk evidence "
                    "can be shown. The alert is reported without a score rather "
                    "than with an invented one."
                ),
                contribution_type=EVIDENCE_UNAVAILABLE,
                source="phase5.alerting",
            )
        )

    source_components = list(getattr(alert, "source_components", ()) or ())
    items.append(
        _context_item(
            "source_components",
            ", ".join(source_components) if source_components else None,
            "Which pipeline stages contributed to this alert. A degraded run "
            "lists exactly what it used.",
            "phase5.alerting",
        )
    )

    predicted = {
        "predicted_latitude": _number(getattr(alert, "predicted_latitude", None)),
        "predicted_longitude": _number(getattr(alert, "predicted_longitude", None)),
        "predicted_intensity": _number(getattr(alert, "predicted_intensity", None)),
    }
    any_prediction = any(value is not None for value in predicted.values())
    for name, value in predicted.items():
        items.append(
            Evidence(
                feature=name,
                value=value,
                unit="deg" if "latitude" in name or "longitude" in name else None,
                context=(
                    "Prototype model prediction attached to this alert. It is an "
                    "extrapolation, not a forecast, and carries no demonstrated skill."
                    if value is not None
                    else "No model prediction is attached to this alert; the GNN "
                    "stage was unavailable or produced nothing usable."
                ),
                contribution_type=SUPPORTING if value is not None else EVIDENCE_UNAVAILABLE,
                source="phase3.gnn",
            )
        )
    if not any_prediction:
        limitations_note = (
            "No model prediction backs this alert, so it rests entirely on "
            "observed detection and tracking evidence."
        )
    else:
        limitations_note = (
            "The attached prediction is a one-step prototype extrapolation. No "
            "verified forecast skill exists for it."
        )

    if event is not None:
        items.append(
            _available(
                "event_intensity",
                getattr(event, "intensity", None),
                getattr(event, "units", None),
                "Peak physical value of the detection that raised this alert.",
                "phase1.anomaly",
            )
        )
        items.append(
            _available(
                "event_anomaly_score",
                getattr(event, "anomaly_score", None),
                "1",
                "Peak anomaly score of the detection that raised this alert.",
                "phase1.anomaly",
            )
        )

    if track is not None:
        items.append(
            _available(
                "persistence_hours",
                getattr(track, "duration_hours", None),
                "h",
                "How long the event persisted as one tracked trajectory.",
                "phase2.tracking",
            )
        )
        items.append(
            _context_item(
                "track_direction",
                getattr(track, "direction", None),
                "Direction of travel observed along the trajectory.",
                "phase2.tracking",
            )
        )

    if data_freshness is not None:
        items.append(
            Evidence(
                feature="data_freshness",
                value=str(data_freshness),
                context=(
                    "Freshness of the data behind this alert. Stale or expired "
                    "data weakens every other item here."
                ),
                contribution_type=CONTEXT,
                source="phase8.realtime",
            )
        )

    limitations = [
        "The risk score is a weighted sum of the factors listed above. It is a "
        "triage aid, not a validated disaster-risk index, and its weights were "
        "chosen by the project, not fitted to outcomes.",
        limitations_note,
        "No skill score exists for this system, because no verified ground-truth "
        "warning catalogue is available. Absence of evidence of error is not "
        "evidence of accuracy.",
    ]
    return AlertExplanation(
        alert_id=str(getattr(alert, "alert_id", "unknown")),
        event_id=str(getattr(alert, "event_id", "") or "") or None,
        evidence=items,
        limitations=limitations,
    )


@dataclass
class AlertExplanation:
    """The risk evidence behind one alert.

    Attributes:
        alert_id: The Phase 5 alert identifier.
        event_id: The detection behind the alert.
        evidence: Every item, including the unavailable ones.
        limitations: What this explanation cannot say.
    """

    alert_id: str
    event_id: str | None
    evidence: list[Evidence] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    @property
    def available_evidence(self) -> list[Evidence]:
        """Only the items backed by a real value."""
        return [item for item in self.evidence if item.available]

    @property
    def unavailable_evidence(self) -> list[Evidence]:
        """Only the gaps."""
        return [item for item in self.evidence if not item.available]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "alert_id": self.alert_id,
            "event_id": self.event_id,
            "evidence": [item.as_dict() for item in self.evidence],
            "available_evidence": [item.as_dict() for item in self.available_evidence],
            "unavailable_evidence": [
                item.as_dict() for item in self.unavailable_evidence
            ],
            "evidence_counts": {
                "available": len(self.available_evidence),
                "unavailable": len(self.unavailable_evidence),
            },
            "limitations": list(self.limitations) + [CAUSALITY_DISCLAIMER],
            "disclaimer": CAUSALITY_DISCLAIMER,
        }


def sensitivity_report(
    feature_names: Sequence[str],
    baseline_features: Sequence[float],
    perturbed_features: Sequence[Mapping[str, float]],
    baseline_prediction: Sequence[float],
    perturbed_predictions: Sequence[Sequence[float]],
) -> dict[str, Any]:
    """Summarise one-at-a-time feature perturbation as **input sensitivity**.

    This measures how the model's *output* responds to a change in an input.
    It says nothing about the physical world: a model can be highly sensitive
    to a feature that has no real influence on the event, and insensitive to
    one that does.

    Args:
        feature_names: Feature names, in vector order.
        baseline_features: The unperturbed input vector.
        perturbed_features: One perturbed input vector per experiment.
        baseline_prediction: The model's output on the unperturbed input.
        perturbed_predictions: The model's output per perturbed input.

    Returns:
        A record with one entry per feature: baseline value, perturbed value,
        the change in the prediction, and a direction label.

    Raises:
        ValueError: If the input lists are inconsistent in length.
    """
    names = list(feature_names)
    base_input = [float(v) for v in baseline_features]
    base_output = [float(v) for v in baseline_prediction]
    if len(perturbed_features) != len(perturbed_predictions):
        raise ValueError(
            f"Got {len(perturbed_features)} perturbed inputs but "
            f"{len(perturbed_predictions)} predictions."
        )
    if len(base_input) != len(names):
        raise ValueError(
            f"Got {len(base_input)} baseline feature values for {len(names)} names."
        )

    experiments: list[dict[str, Any]] = []
    for perturbation, prediction in zip(perturbed_features, perturbed_predictions):
        output = [float(v) for v in prediction]
        # How far the prediction moved in response to the input change.
        delta = [p - b for p, b in zip(output, base_output)]
        magnitude = float(np.linalg.norm(delta)) if delta else 0.0
        for feature, value in perturbation.items():
            if feature not in names:
                # Not a model input: say so instead of reporting a delta of 0,
                # which would read as "this feature does not matter".
                experiments.append(
                    {
                        "feature": feature,
                        "perturbed": False,
                        "reason": (
                            f"'{feature}' is not a model input, so perturbing it "
                            "cannot change the prediction."
                        ),
                    }
                )
                continue
            index = names.index(feature)
            experiments.append(
                {
                    "feature": feature,
                    "baseline_value": base_input[index],
                    "perturbed_value": float(value),
                    "prediction_delta": delta,
                    "delta_magnitude": magnitude,
                    "direction": (
                        "increases_output"
                        if magnitude > 0.0
                        else "no_change"
                        if magnitude == 0.0
                        else "decreases_output"
                    ),
                    "perturbed": True,
                }
            )

    ranked = sorted(
        (item for item in experiments if item.get("perturbed")),
        key=lambda item: float(item.get("delta_magnitude", 0.0)),
        reverse=True,
    )
    return {
        "method": "one-at-a-time input perturbation",
        "reported_as": "input_sensitivity",
        "baseline_prediction": base_output,
        "experiments": experiments,
        "ranked_by_response": [item["feature"] for item in ranked],
        "interpretation": (
            "These deltas are the model's response to a changed input. They are "
            "a property of the fitted function, not evidence that any feature "
            "causes the physical event, and they must not be read causally."
        ),
        "disclaimer": CAUSALITY_DISCLAIMER,
    }
