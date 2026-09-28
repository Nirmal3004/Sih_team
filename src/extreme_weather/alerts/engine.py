"""Deterministic risk / priority engine (SIH26078, Phase 5).

The engine turns the facts of a tracked event into a **transparent** 0-100 risk
score. "Transparent" is a hard requirement, so the design rules are:

* **No black box.** The score is a plain weighted sum of named factors. Every
  point is attributable to a :class:`~extreme_weather.alerts.models.ContributingFactor`
  with its weight, its normalised value and its raw input, so a reader can redo
  the arithmetic by hand.
* **No randomness.** No sampling, no clock reads, no model inference. The same
  inputs always produce the same score, which is what makes the tests and the
  ``--dry-run`` mode meaningful.
* **No probability.** The score is a triage aid, not a forecast probability and
  not a validated disaster-risk index. Every payload repeats that statement.
* **Graceful degradation.** A missing GNN prediction or a missing downscaled
  field contributes a *zero* factor with an explicit ``detail`` string, instead
  of raising. The alert then honestly reports which components it used.

Available inputs
----------------
The engine consumes a :class:`RiskContext`, which the caller assembles from the
Phase 1-4 stores. Every field of the context is optional except the observed
event state, so the same function serves an events-only run and a full
events + tracks + GNN + downscaling run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

from extreme_weather.alerts.models import ContributingFactor, RiskAssessment
from extreme_weather.alerts.severity import (
    AlertSeverity,
    SeverityThresholds,
    severity_for_score,
)
from extreme_weather.schemas.event import EventType, Severity as EventSeverity
from extreme_weather.tracking.metrics import (
    TREND_STABLE,
    TREND_STRENGTHENING,
    TREND_WEAKENING,
)

__all__ = [
    "FACTOR_WEIGHTS",
    "RISK_DISCLAIMER",
    "RiskConfig",
    "RiskContext",
    "assess_risk",
    "clamp01",
]


RISK_DISCLAIMER = (
    "Rule-based prototype triage score. It is NOT a validated disaster-risk "
    "index, NOT calibrated against impact, and NOT a probability of occurrence."
)

#: Maximum points each factor may contribute. They sum to exactly 100 so the
#: weighted sum lands directly on the 0-100 scale the severity ladder expects.
FACTOR_WEIGHTS: dict[str, float] = {
    "anomaly_severity": 22.0,
    "intensity": 14.0,
    "intensity_trend": 12.0,
    "movement_speed": 8.0,
    "predicted_intensity_change": 12.0,
    "predicted_movement": 6.0,
    "affected_radius": 10.0,
    "persistence": 10.0,
    "downscaled_threat": 6.0,
}

#: Default reference values used to normalise physical magnitudes onto [0, 1].
#: They are *display/reference* constants of the prototype ladder - they are
#: configurable, and they are not claimed to be physical thresholds.
DEFAULT_INTENSITY_REFERENCE: dict[EventType, float] = {
    EventType.EXTREME_RAINFALL: 50.0,
    EventType.EXTREME_WIND: 15.0,
    EventType.EXTREME_TEMPERATURE: 8.0,
}
DEFAULT_INTENSITY_REFERENCE_FALLBACK: float = 50.0
DEFAULT_RADIUS_REFERENCE_KM: float = 150.0
DEFAULT_SPEED_REFERENCE_KMH: float = 60.0
DEFAULT_PERSISTENCE_REFERENCE_HOURS: float = 12.0


def clamp01(value: float) -> float:
    """Clamp a value onto ``[0, 1]``, mapping non-finite input to ``0.0``."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number != number or number in (float("inf"), float("-inf")):
        return 0.0
    if number < 0.0:
        return 0.0
    if number > 1.0:
        return 1.0
    return number


@dataclass(frozen=True)
class RiskConfig:
    """Configurable reference values of the scoring ladder.

    Changing any of these changes the score; none of them is a claim about
    physical danger thresholds.
    """

    intensity_reference: Mapping[EventType, float] = field(
        default_factory=lambda: dict(DEFAULT_INTENSITY_REFERENCE)
    )
    intensity_reference_fallback: float = DEFAULT_INTENSITY_REFERENCE_FALLBACK
    radius_reference_km: float = DEFAULT_RADIUS_REFERENCE_KM
    speed_reference_kmh: float = DEFAULT_SPEED_REFERENCE_KMH
    persistence_reference_hours: float = DEFAULT_PERSISTENCE_REFERENCE_HOURS
    severity_thresholds: SeverityThresholds = SeverityThresholds()

    def __post_init__(self) -> None:
        for name in (
            "radius_reference_km",
            "speed_reference_kmh",
            "persistence_reference_hours",
            "intensity_reference_fallback",
        ):
            if float(getattr(self, name)) <= 0.0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}.")

    def intensity_reference_for(self, event_type: EventType | str | None) -> float:
        """Return the reference intensity for an event type."""
        value = None
        if event_type is not None:
            resolved = (
                event_type
                if isinstance(event_type, EventType)
                else EventType.from_value(str(event_type))
            )
            value = self.intensity_reference.get(resolved)
        return float(value) if value else self.intensity_reference_fallback

    def as_dict(self) -> dict[str, Any]:
        """Return the configuration as a JSON-serialisable dictionary."""
        return {
            "intensity_reference": {
                key.value: value for key, value in self.intensity_reference.items()
            },
            "intensity_reference_fallback": self.intensity_reference_fallback,
            "radius_reference_km": self.radius_reference_km,
            "speed_reference_kmh": self.speed_reference_kmh,
            "persistence_reference_hours": self.persistence_reference_hours,
            "severity_thresholds": self.severity_thresholds.as_dict(),
            "factor_weights": dict(FACTOR_WEIGHTS),
        }


@dataclass(frozen=True)
class RiskContext:
    """Everything the scorer may use about one tracked event.

    Only ``intensity``, ``event_severity`` and ``event_id`` are always required
    (they come from the Phase 1 detection). Everything else is optional and
    corresponds to a pipeline stage that may be unavailable.
    """

    event_id: str
    event_type: EventType
    intensity: float
    event_severity: EventSeverity
    affected_radius_km: float = 0.0
    # --- Phase 2 (tracking) ---
    track_id: str | None = None
    duration_hours: float = 0.0
    movement_speed_kmh: float = 0.0
    intensity_trend: str | None = None
    peak_intensity: float | None = None
    # --- Phase 3 (GNN prediction) ---
    predicted_intensity: float | None = None
    predicted_speed_kmh: float | None = None
    # --- Phase 4 (AI downscaling) ---
    downscaled_threat: float | None = None
    downscaled_resolution_km: float | None = None
    # --- bookkeeping ---
    source_components: Sequence[str] = ()


def _factor(
    name: str,
    label: str,
    normalized: float,
    detail: str,
    raw_value: float | None = None,
    weights: Mapping[str, float] = FACTOR_WEIGHTS,
) -> ContributingFactor:
    """Build one explainable scoring term."""
    weight = float(weights[name])
    bounded = clamp01(normalized)
    return ContributingFactor(
        name=name,
        label=label,
        weight=weight,
        normalized_value=round(bounded, 4),
        raw_value=None if raw_value is None else round(float(raw_value), 4),
        points=round(weight * bounded, 4),
        detail=detail,
    )


def _trend_factor(
    trend: str | None,
    first: float | None,
    last: float,
) -> tuple[float, str]:
    """Normalise an intensity trend.

    A known Phase 2 trend label is used directly. When the track did not report
    one, the trend is inferred from the first and last observed intensity - but
    only when a first intensity actually exists. An untracked event has no
    history, so inferring a trend from a placeholder zero would be inventing
    evidence; it is reported as "no trend information" instead.
    """
    if trend == TREND_STRENGTHENING:
        return 1.0, "intensity is strengthening over the track"
    if trend == TREND_WEAKENING:
        return 0.0, "intensity is weakening over the track"
    if trend == TREND_STABLE:
        return 0.4, "intensity is roughly stable over the track"
    if trend is not None and trend.strip():
        return 0.4, f"unrecognised trend '{trend}' treated as stable"
    if first is None:
        return 0.4, "no track history available; trend treated as neutral"

    scale = max(abs(first), abs(last))
    if scale == 0.0:
        return 0.4, "no trend information available; treated as stable"
    change = (last - first) / scale
    if change > 0.05:
        return 1.0, f"inferred strengthening (intensity {first:.2f} -> {last:.2f})"
    if change < -0.05:
        return 0.0, f"inferred weakening (intensity {first:.2f} -> {last:.2f})"
    return 0.4, f"inferred stable (intensity {first:.2f} -> {last:.2f})"


def _explanation(
    level: AlertSeverity,
    factors: Sequence[ContributingFactor],
    config: RiskConfig,
    context: RiskContext,
) -> str:
    """Compose the human-readable explanation of a score.

    The lines are ordered by the points each factor contributed, so the most
    influential reasons appear first.
    """
    ranked = sorted(factors, key=lambda f: (-f.points, f.name))
    lines = [
        f"{level.value} priority (score {round(sum(f.points for f in factors), 2)}"
        f"/100) because:"
    ]
    for factor in ranked:
        if factor.points <= 0.0:
            continue
        lines.append(f"- {factor.label}: {factor.detail} (+{factor.points:.2f})")

    if not context.predicted_intensity:
        lines.append(
            "- no GNN prediction available, so no forward-looking intensity was used"
        )
    if context.downscaled_threat is None:
        lines.append(
            "- no AI-downscaled threat field available, so no ~5 km threat "
            "information was used"
        )
    lines.append(RISK_DISCLAIMER)
    return "\n".join(lines)


def assess_risk(
    context: RiskContext,
    config: RiskConfig | None = None,
    generated_at: datetime | None = None,
) -> RiskAssessment:
    """Score one event deterministically and explain the result.

    Args:
        context: The observed facts about the event.
        config: Reference values of the ladder. Defaults to :class:`RiskConfig`.
        generated_at: Timestamp to stamp on the assessment. Defaults to *now*;
            pass an explicit value to keep a whole run reproducible.

    Returns:
        A :class:`~extreme_weather.alerts.models.RiskAssessment` whose
        ``contributing_factors`` sum exactly to ``score``.

    Raises:
        ValueError: If ``context`` is missing a required observed value.
    """
    active = config or RiskConfig()
    if context.intensity is None or context.event_severity is None:
        raise ValueError("assess_risk requires the observed intensity and event severity.")

    stamp = generated_at or datetime.now(timezone.utc).replace(microsecond=0)
    factors: list[ContributingFactor] = []

    # 1. Anomaly severity: how strong the Phase 1 detection was.
    severity_ratio = context.event_severity.rank / max(1, len(tuple(EventSeverity)) - 1)
    factors.append(
        _factor(
            "anomaly_severity",
            "Anomaly severity",
            severity_ratio,
            f"Phase 1 severity {context.event_severity.value}",
            raw_value=float(context.event_severity.rank),
        )
    )

    # 2. Absolute intensity relative to the reference for this event type.
    reference = active.intensity_reference_for(context.event_type)
    factors.append(
        _factor(
            "intensity",
            "Observed intensity",
            clamp01(context.intensity / reference),
            f"{context.intensity:.2f} against a reference of {reference:.2f}",
            raw_value=context.intensity,
        )
    )

    # 3. Trend: strengthening raises priority, weakening lowers it.
    trend_value, trend_detail = _trend_factor(
        context.intensity_trend,
        float(context.peak_intensity) if context.peak_intensity is not None else None,
        float(context.intensity),
    )
    factors.append(
        _factor("intensity_trend", "Intensity trend", trend_value, trend_detail)
    )

    # 4. Movement speed: a fast-moving cell reaches new areas sooner.
    factors.append(
        _factor(
            "movement_speed",
            "Movement speed",
            clamp01(context.movement_speed_kmh / active.speed_reference_kmh),
            f"{context.movement_speed_kmh:.2f} km/h against a reference of "
            f"{active.speed_reference_kmh:.2f} km/h",
            raw_value=context.movement_speed_kmh,
        )
    )


    # 5. Predicted intensity change (Phase 3), zero when unavailable.
    if context.predicted_intensity is None:
        factors.append(
            _factor(
                "predicted_intensity_change",
                "Predicted intensity change",
                0.0,
                "GNN prediction unavailable - factor contributes nothing",
            )
        )
    else:
        delta = context.predicted_intensity - context.intensity
        factors.append(
            _factor(
                "predicted_intensity_change",
                "Predicted intensity change",
                clamp01(delta / reference),
                f"predicted intensity {context.predicted_intensity:.2f} "
                f"({delta:+.2f} against the current {context.intensity:.2f})",
                raw_value=delta,
            )
        )

    # 6. Predicted movement speed (Phase 3), zero when unavailable.
    if context.predicted_speed_kmh is None:
        factors.append(
            _factor(
                "predicted_movement",
                "Predicted movement",
                0.0,
                "GNN prediction unavailable - factor contributes nothing",
            )
        )
    else:
        factors.append(
            _factor(
                "predicted_movement",
                "Predicted movement",
                clamp01(context.predicted_speed_kmh / active.speed_reference_kmh),
                f"predicted speed {context.predicted_speed_kmh:.2f} km/h",
                raw_value=context.predicted_speed_kmh,
            )
        )

    # 7. Affected radius: the spatial footprint of the event.
    factors.append(
        _factor(
            "affected_radius",
            "Affected radius",
            clamp01(context.affected_radius_km / active.radius_reference_km),
            f"{context.affected_radius_km:.2f} km against a reference of "
            f"{active.radius_reference_km:.2f} km",
            raw_value=context.affected_radius_km,
        )
    )

    # 8. Persistence: a long-lived track is a more durable threat.
    if context.track_id is None:
        factors.append(
            _factor(
                "persistence",
                "Event persistence",
                0.0,
                "event is not tracked - persistence is unknown",
            )
        )
    else:
        factors.append(
            _factor(
                "persistence",
                "Event persistence",
                clamp01(context.duration_hours / active.persistence_reference_hours),
                f"tracked for {context.duration_hours:.2f} h against a reference of "
                f"{active.persistence_reference_hours:.2f} h",
                raw_value=context.duration_hours,
            )
        )

    # 9. Downscaled threat field (Phase 4), zero when unavailable.
    if context.downscaled_threat is None:
        factors.append(
            _factor(
                "downscaled_threat",
                "AI-downscaled threat field",
                0.0,
                "no downscaled threat field available - factor contributes nothing",
            )
        )
    else:
        factors.append(
            _factor(
                "downscaled_threat",
                "AI-downscaled threat field",
                clamp01(context.downscaled_threat / reference),
                f"downscaled threat value {context.downscaled_threat:.2f} at "
                f"~{context.downscaled_resolution_km or 5.0:.1f} km resolution",
                raw_value=context.downscaled_threat,
            )
        )

    score = round(sum(factor.points for factor in factors), 4)
    return RiskAssessment.from_score(
        score=score,
        factors=factors,
        explanation=_explanation(severity_for_score(score, active.severity_thresholds),
                                 factors, active, context),
        generated_at=stamp,
        thresholds=active.severity_thresholds,
    )

