"""Alert domain model (SIH26078, Phase 5).

These are the *external* contract of the alerting layer. The engine, the store,
the CLI and the FastAPI router all use the same objects, so the JSON
representation of an alert is defined exactly once - mirroring how Phase 1
(:mod:`extreme_weather.schemas.event`), Phase 2
(:mod:`extreme_weather.schemas.track`), Phase 3
(:mod:`extreme_weather.gnn.models`) and Phase 4
(:mod:`extreme_weather.downscaling.api_models`) each define theirs.

Everything here is a validated Pydantic model: an alert is never a loose
dictionary, and an invalid status, coordinate or timestamp is rejected at the
boundary rather than propagated into the store.

Honesty rules encoded in the model
----------------------------------
* Every alert carries a :class:`RiskAssessment` with a human-readable
  ``explanation``: an alert must be able to say *why* it exists.
* Every alert carries ``source_components``: if the GNN model was unavailable,
  the alert says so instead of silently pretending a prediction was used.
* No field is a probability of correctness. ``confidence_label`` is a coarse,
  documented band describing *input completeness*, never "95% accurate".
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from extreme_weather.alerts.severity import (
    AlertSeverity,
    SeverityThresholds,
    severity_for_score,
)
from extreme_weather.config import PROJECT_ID, PROJECT_NAME
from extreme_weather.schemas.event import EventType

__all__ = [
    "ALERT_NOTICE",
    "CONFIDENCE_LABELS",
    "TIME_BUCKET_MINUTES",
    "Alert",
    "AlertListResponse",
    "AlertStatus",
    "AlertStoreResult",
    "AlertSummary",
    "ConfidenceLabel",
    "ContributingFactor",
    "RiskAssessment",
    "SeverityCount",
    "StatusCount",
    "build_alert_id",
    "deduplication_key",
    "time_bucket",
]


#: Shown wherever an alert is displayed. The wording is deliberately explicit
#: that this is a prototype and not a warning product.
ALERT_NOTICE = (
    "Prototype alerting layer built on synthetic data. Alerts are produced by "
    "deterministic rules, not by a validated disaster-prediction model, and are "
    "NOT an official warning. Do not use them for emergency decisions."
)

#: Granularity, in minutes, of the deduplication time bucket.
TIME_BUCKET_MINUTES: int = 60


def _utc(value: datetime) -> datetime:
    """Interpret a naive datetime as UTC and convert an aware one to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class AlertStatus(StrEnum):
    """Lifecycle state of an alert.

    The legal transitions are enforced by :mod:`extreme_weather.alerts.service`:
    ``NEW -> ACTIVE -> ACKNOWLEDGED -> RESOLVED``, plus ``ACTIVE -> EXPIRED``.
    ``RESOLVED`` and ``EXPIRED`` are terminal.
    """

    NEW = "NEW"
    ACTIVE = "ACTIVE"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    EXPIRED = "EXPIRED"

    @property
    def is_terminal(self) -> bool:
        """True for the two states an alert can never leave."""
        return self in (AlertStatus.RESOLVED, AlertStatus.EXPIRED)

    @property
    def is_open(self) -> bool:
        """True while the alert still needs operator attention."""
        return not self.is_terminal

    @classmethod
    def from_value(cls, value: str) -> "AlertStatus":
        """Parse a status from a case-insensitive string.

        Raises:
            ValueError: If the status is unknown.
        """
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            supported = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown alert status '{value}'. Supported statuses: {supported}."
            ) from exc


class ConfidenceLabel(StrEnum):
    """How complete the *inputs* behind an alert were.

    This is a data-availability band, **not** a probability that the alert is
    correct. ``HIGH`` means every pipeline component contributed; ``LOW`` means
    the alert was raised from detections alone.
    """

    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

    @classmethod
    def for_components(cls, components: Sequence[str]) -> "ConfidenceLabel":
        """Derive the band from the set of contributing components.

        The rule is explicit and documented rather than tuned: an alert with a
        prediction *and* a downscaled field is ``HIGH``, one with only one of the
        two optional components is ``MEDIUM``, and a detection-only alert is
        ``LOW``.
        """
        present = set(components)
        has_gnn = "gnn" in present
        has_down = "downscaling" in present
        if has_gnn and has_down:
            return cls.HIGH
        if has_gnn or has_down:
            return cls.MEDIUM
        return cls.LOW


#: Documented meaning of each band, exposed through the API.
CONFIDENCE_LABELS: dict[str, str] = {
    ConfidenceLabel.HIGH.value: (
        "Detection, tracking, a GNN prediction and a downscaled field all contributed."
    ),
    ConfidenceLabel.MEDIUM.value: (
        "Detection and tracking plus one optional component (either a GNN "
        "prediction or a downscaled field)."
    ),
    ConfidenceLabel.LOW.value: (
        "Detection and tracking only - the GNN and downscaling models were "
        "unavailable, so no forward-looking information was used."
    ),
}


class ContributingFactor(BaseModel):
    """One explainable term of the risk score.

    An alert score is never an unexplained number: every point added to it is
    attributable to one of these, and the raw inputs are kept so a reader can
    check the arithmetic.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(min_length=1, description="Stable factor identifier.")
    label: str = Field(min_length=1, description="Human-readable factor name.")
    weight: float = Field(ge=0.0, description="Maximum points this factor can add.")
    normalized_value: float = Field(
        ge=0.0, le=1.0, description="Factor value mapped onto [0, 1]."
    )
    raw_value: float | None = Field(
        default=None, description="The unmodified input value, when numeric."
    )
    points: float = Field(ge=0.0, description="Points actually contributed.")
    detail: str = Field(default="", description="Why this factor scored as it did.")


class RiskAssessment(BaseModel):
    """A deterministic, explainable 0-100 risk score.

    The score is a plain weighted sum of :class:`ContributingFactor` terms whose
    weights sum to 100, computed by
    :func:`extreme_weather.alerts.engine.assess_risk`. It is reproducible: the
    same inputs always give the same score, with no randomness, no learned
    weights and no hidden state.

    .. warning::

       This score is a **rule-based triage aid** for a research prototype. It is
       not a scientifically validated disaster-risk index, it is not calibrated
       against impact or casualties, and it carries no probability meaning.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    score: float = Field(ge=0.0, le=100.0, description="Weighted risk score in [0, 100].")
    level: AlertSeverity = Field(description="Severity the score maps onto.")
    contributing_factors: list[ContributingFactor] = Field(
        default_factory=list, description="Every term that built the score."
    )
    explanation: str = Field(
        min_length=1, description="Plain-language statement of why this level was chosen."
    )
    generated_at: datetime = Field(description="UTC time the assessment was produced.")
    severity_thresholds: dict[str, float] = Field(
        default_factory=dict, description="Cut points applied to reach the level."
    )
    notice: str = Field(
        default=(
            "Rule-based prototype score, not a validated disaster-risk index and "
            "not a probability of occurrence."
        ),
        description="Prototype disclaimer.",
    )

    @field_validator("generated_at")
    @classmethod
    def _generated_to_utc(cls, value: datetime) -> datetime:
        return _utc(value)

    @property
    def total_points(self) -> float:
        """Sum of the contributed points (equals :attr:`score` by construction)."""
        return round(sum(factor.points for factor in self.contributing_factors), 4)

    def reason_lines(self) -> list[str]:
        """Return the explanation split into one reason per line."""
        return [line.strip() for line in self.explanation.splitlines() if line.strip()]

    @classmethod
    def from_score(
        cls,
        score: float,
        factors: Sequence[ContributingFactor],
        explanation: str,
        generated_at: datetime,
        thresholds: SeverityThresholds | None = None,
    ) -> "RiskAssessment":
        """Build an assessment from a score, deriving the level from the thresholds."""
        applied = thresholds or SeverityThresholds()
        return cls(
            score=round(float(score), 2),
            level=severity_for_score(score, applied),
            contributing_factors=list(factors),
            explanation=explanation,
            generated_at=generated_at,
            severity_thresholds=applied.as_dict(),
        )


def time_bucket(moment: datetime, minutes: int = TIME_BUCKET_MINUTES) -> str:
    """Return the deterministic time-bucket label of ``moment``.

    Buckets are aligned to the epoch so the same instant always falls into the
    same bucket regardless of when the pipeline happens to run - that is what
    makes deduplication stable.
    """
    if minutes <= 0:
        raise ValueError("Bucket size must be a positive number of minutes.")
    stamp = _utc(moment)
    epoch_minutes = int(stamp.timestamp()) // 60
    bucket_index = epoch_minutes // minutes
    return f"{bucket_index * minutes:012d}"


def deduplication_key(
    event_id: str,
    level: AlertSeverity,
    moment: datetime,
    minutes: int = TIME_BUCKET_MINUTES,
) -> str:
    """Return the deduplication key ``event_id + alert_level + time_bucket``.

    Two generation runs inside the same bucket that resolve to the same severity
    produce the same key, so the second run updates the existing alert instead
    of creating a duplicate. Raising the severity moves the key, which is
    exactly when a *new* alert is the right behaviour.
    """
    return f"{event_id}|{level.value}|{time_bucket(moment, minutes)}"


def build_alert_id(
    event_id: str,
    level: AlertSeverity,
    moment: datetime,
    minutes: int = TIME_BUCKET_MINUTES,
) -> str:
    """Build the deterministic alert identifier.

    The id is derived from the deduplication key, so it is stable across runs
    and contains no random component: re-running the generator over the same
    data reproduces the same ids.
    """
    key = deduplication_key(event_id, level, moment, minutes)
    return f"ALT-{key.replace('|', '-')}"



class Alert(BaseModel):
    """One structured alert derived from a tracked extreme weather event.

    Field semantics
    ---------------
    * ``alert_id`` - deterministic, built from the deduplication key (never
      random, never a UUID), so re-running the pipeline is idempotent.
    * ``event_id`` - the Phase 1 detection this alert is about. Always present:
      an alert without a detection behind it would be fabricated.
    * ``track_id`` - the Phase 2 trajectory, when the event was tracked.
    * ``current_*`` - the last **observed** state (copied from the track, never
      predicted).
    * ``predicted_*`` - the Phase 3 GNN next-state prediction, ``None`` when the
      model was unavailable. Absence is explicit, not silently zero.
    * ``intensity`` / ``predicted_intensity`` - physical values in the source
      variable's unit (mm h-1, m s-1, K), not anomaly scores.
    * ``source_components`` - which pipeline stages contributed. A degraded run
      reports exactly what it used, e.g. ``["events", "tracking"]``.
    * ``valid_from`` / ``valid_until`` - the window the alert is considered
      current for; ``valid_until`` comes from the configured validity minutes.
    * ``acknowledged_at`` / ``resolved_at`` - real timestamps recorded when the
      operator acted, or ``None``. They are never back-dated or invented.
    """

    model_config = ConfigDict(extra="forbid")

    alert_id: str = Field(min_length=1, description="Deterministic alert identifier.")
    event_id: str = Field(min_length=1, description="Phase 1 event identifier.")
    track_id: str | None = Field(default=None, description="Phase 2 track identifier.")
    event_type: EventType
    severity: AlertSeverity
    priority: AlertSeverity = Field(
        description="Operator priority; equals severity unless a rule elevates it."
    )
    title: str = Field(min_length=1, max_length=200)
    message: str = Field(min_length=1, max_length=2000)

    created_at: datetime
    updated_at: datetime
    valid_from: datetime
    valid_until: datetime
    status: AlertStatus = AlertStatus.NEW

    confidence_label: ConfidenceLabel = ConfidenceLabel.LOW
    risk: RiskAssessment

    current_lat: float = Field(ge=-90.0, le=90.0)
    current_lon: float = Field(ge=-180.0, le=180.0)
    predicted_lat: float | None = Field(default=None, ge=-90.0, le=90.0)
    predicted_lon: float | None = Field(default=None, ge=-180.0, le=180.0)
    intensity: float
    predicted_intensity: float | None = None
    movement_speed_kmh: float = Field(default=0.0, ge=0.0)
    movement_direction: str = Field(default="CALM", min_length=1)
    affected_radius_km: float = Field(default=0.0, ge=0.0)

    source_components: list[str] = Field(
        default_factory=list,
        description="Pipeline stages that contributed (events, tracking, gnn, downscaling).",
    )
    deduplication_key: str = Field(min_length=1)

    acknowledged_at: datetime | None = None
    resolved_at: datetime | None = None
    resolved_reason: str | None = None

    notice: str = Field(default=ALERT_NOTICE, description="Prototype disclaimer.")

    @field_validator(
        "created_at",
        "updated_at",
        "valid_from",
        "valid_until",
        "acknowledged_at",
        "resolved_at",
    )
    @classmethod
    def _timestamps_to_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @field_validator("source_components")
    @classmethod
    def _unique_components(cls, value: list[str]) -> list[str]:
        """Reject duplicated component names rather than storing them twice."""
        if len(set(value)) != len(value):
            raise ValueError("source_components must not contain duplicates.")
        return value

    @model_validator(mode="after")
    def _validate_alert(self) -> "Alert":
        """Check the internal consistency of the alert."""
        if self.valid_until < self.valid_from:
            raise ValueError("valid_until must not be earlier than valid_from.")
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not be earlier than created_at.")
        if self.status is AlertStatus.ACKNOWLEDGED and self.acknowledged_at is None:
            raise ValueError("An ACKNOWLEDGED alert must carry acknowledged_at.")
        if self.status is AlertStatus.RESOLVED and self.resolved_at is None:
            raise ValueError("A RESOLVED alert must carry resolved_at.")
        if (self.predicted_lat is None) != (self.predicted_lon is None):
            raise ValueError(
                "predicted_lat and predicted_lon must be provided together or both be null."
            )
        return self

    @property
    def is_open(self) -> bool:
        """True while the alert still needs operator attention."""
        return self.status.is_open

    @property
    def has_prediction(self) -> bool:
        """True when a GNN prediction contributed to this alert."""
        return self.predicted_lat is not None and self.predicted_lon is not None



class StatusCount(BaseModel):
    """Number of alerts in one lifecycle state."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: AlertStatus
    count: int = Field(ge=0)


class SeverityCount(BaseModel):
    """Number of alerts at one severity level."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    severity: AlertSeverity
    count: int = Field(ge=0)


class AlertSummary(BaseModel):
    """Dashboard headline counters (``GET /api/v1/alerts/summary``).

    Every counter is a count of *stored* alerts - nothing here is predicted and
    nothing is estimated.
    """

    model_config = ConfigDict(extra="forbid")

    total_alerts: int = Field(ge=0)
    active_alerts: int = Field(ge=0, description="Alerts in NEW, ACTIVE or ACKNOWLEDGED.")
    severe_alerts: int = Field(ge=0, description="Open alerts at SEVERE_WARNING or EXTREME.")
    acknowledged_alerts: int = Field(ge=0)
    resolved_alerts: int = Field(ge=0)
    expired_alerts: int = Field(ge=0)
    tracked_events: int = Field(
        ge=0, description="Distinct Phase 2 tracks referenced by stored alerts."
    )
    predicted_threats: int = Field(
        ge=0, description="Open alerts that carry a GNN predicted position."
    )
    highest_severity: AlertSeverity | None = Field(
        default=None, description="Highest severity among open alerts."
    )
    by_status: list[StatusCount] = Field(default_factory=list)
    by_severity: list[SeverityCount] = Field(default_factory=list)
    source: str | None = Field(default=None, description="Alert file the counts came from.")
    generated_at: datetime | None = Field(default=None)
    message: str | None = Field(default=None, description="Explains an empty result.")
    notice: str = Field(default=ALERT_NOTICE)

    @field_validator("generated_at")
    @classmethod
    def _generated_to_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)


class AlertListResponse(BaseModel):
    """Result of ``GET /api/v1/alerts``."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    alerts: list[Alert] = Field(default_factory=list)
    source: str | None = Field(default=None, description="Alert file the alerts came from.")
    generated_at: datetime | None = Field(default=None)
    message: str | None = Field(default=None, description="Explains an empty result.")
    notice: str = Field(default=ALERT_NOTICE)

    @field_validator("generated_at")
    @classmethod
    def _generated_to_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)



class AlertStoreResult(BaseModel):
    """Everything one alert-generation run produced (the stored container).

    This mirrors :class:`~extreme_weather.tracking.models.TrackResult` and
    :class:`~extreme_weather.anomaly.models.AnomalyAnalysisResult`: a versioned,
    JSON-serialisable envelope written to ``data/processed/alerts.json``.
    """

    model_config = ConfigDict(extra="forbid")

    alerts: list[Alert] = Field(default_factory=list)
    generated_at: datetime | None = None
    created_count: int = Field(default=0, ge=0)
    updated_count: int = Field(default=0, ge=0)
    suppressed_count: int = Field(
        default=0, ge=0, description="Candidates dropped by deduplication or the trigger rule."
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("generated_at")
    @classmethod
    def _generated_to_utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @property
    def alert_count(self) -> int:
        """Number of alerts currently held."""
        return len(self.alerts)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary with a stable key order."""
        return {
            "schema_version": "1.0",
            "project": PROJECT_ID,
            "service": PROJECT_NAME,
            "phase": "Phase 5 - alerting and emergency response (prototype)",
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "alert_count": self.alert_count,
            "created_count": self.created_count,
            "updated_count": self.updated_count,
            "suppressed_count": self.suppressed_count,
            "status_counts": {
                status.value: sum(1 for a in self.alerts if a.status is status)
                for status in AlertStatus
            },
            "severity_counts": {
                severity.value: sum(1 for a in self.alerts if a.severity is severity)
                for severity in AlertSeverity
            },
            "metadata": dict(self.metadata),
            "alerts": [alert.model_dump(mode="json") for alert in self.alerts],
            "notice": ALERT_NOTICE,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AlertStoreResult":
        """Rebuild a result from :meth:`as_dict` output.

        Malformed individual records are skipped rather than poisoning the whole
        file: one bad alert must not make the dashboard unreachable. The number
        skipped is recorded in ``metadata["skipped_records"]``.
        """
        raw_alerts = payload.get("alerts", [])
        alerts: list[Alert] = []
        seen: set[str] = set()
        skipped = 0
        for item in raw_alerts if isinstance(raw_alerts, list) else []:
            try:
                alert = Alert.model_validate(item)
            except Exception:  # a malformed record must not break the store
                skipped += 1
                continue
            if alert.alert_id in seen:  # duplicate ids: keep the first occurrence
                skipped += 1
                continue
            seen.add(alert.alert_id)
            alerts.append(alert)

        generated = payload.get("generated_at")
        metadata = dict(payload.get("metadata", {}))
        if skipped:
            metadata["skipped_records"] = skipped
        return cls(
            alerts=alerts,
            generated_at=datetime.fromisoformat(generated) if generated else None,
            created_count=int(payload.get("created_count", 0)),
            updated_count=int(payload.get("updated_count", 0)),
            suppressed_count=int(payload.get("suppressed_count", 0)),
            metadata=metadata,
        )

