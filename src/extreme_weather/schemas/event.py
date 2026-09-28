"""Pydantic schemas for extreme weather events (SIH26078, Phase 1).

These models are the *external* contract of the anomaly-detection service. The
analysis modules in :mod:`extreme_weather.anomaly` produce them directly, and
the FastAPI router reuses them for responses, so the JSON representation of an
event is defined exactly once.

Note on semantics: a field named ``timestamp`` is the **valid time** of the grid
slice the event was detected in (not a model run time). Naive datetimes are
interpreted as UTC.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EventType(StrEnum):
    """Supported extreme weather event types (Phase 1 scope)."""

    EXTREME_RAINFALL = "EXTREME_RAINFALL"
    EXTREME_WIND = "EXTREME_WIND"
    EXTREME_TEMPERATURE = "EXTREME_TEMPERATURE"

    @classmethod
    def from_value(cls, value: str) -> "EventType":
        """Parse an event type from a case-insensitive string.

        Raises:
            ValueError: If the value does not name a supported event type.
        """
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:  # pragma: no cover - message is the contract
            supported = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown event type '{value}'. Supported types: {supported}."
            ) from exc


class Severity(StrEnum):
    """Ordered severity classes used by the prototype threshold ladders."""

    NORMAL = "NORMAL"
    WATCH = "WATCH"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"
    EXTREME = "EXTREME"

    @property
    def rank(self) -> int:
        """Numeric rank, 0 (NORMAL) to 4 (EXTREME), for comparisons and sorting."""
        return _SEVERITY_ORDER.index(self)

    @property
    def is_extreme(self) -> bool:
        """True for the highest severity class."""
        return self is Severity.EXTREME

    @classmethod
    def from_value(cls, value: str) -> "Severity":
        """Parse a severity from a case-insensitive string.

        Raises:
            ValueError: If the value does not name a supported severity.
        """
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            supported = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown severity '{value}'. Supported severities: {supported}."
            ) from exc


#: Declaration order defines the severity ranking (NORMAL < ... < EXTREME).
_SEVERITY_ORDER: tuple[Severity, ...] = (
    Severity.NORMAL,
    Severity.WATCH,
    Severity.MODERATE,
    Severity.SEVERE,
    Severity.EXTREME,
)


class ExtremeWeatherEvent(BaseModel):
    """A single extreme weather event detected in one time slice.

    Field semantics:

    * ``event_id`` - deterministic identifier built from the event type, the
      valid time and the cluster index (never random, never global state).
    * ``latitude``/``longitude`` - intensity-weighted centroid of the detected
      cells, in degrees.
    * ``intensity`` - the peak physical value inside the event (for example the
      maximum precipitation rate in mm/h), i.e. the physical magnitude of the
      event, not the anomaly.
    * ``anomaly_score`` - the peak anomaly score inside the event, on the scale
      of the score function that produced it (see ``extreme_weather.anomaly.scores``).
    * ``affected_radius_km`` - maximum geodesic distance from the centroid to any
      detected cell (haversine approximation).
    * ``cell_count`` - number of grid cells in the event (always >= 1).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1, description="Deterministic event identifier.")
    event_type: EventType
    timestamp: datetime = Field(description="Valid time of the detected grid slice (UTC).")
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    intensity: float = Field(description="Peak physical value inside the event.")
    anomaly_score: float = Field(description="Peak anomaly score inside the event.")
    severity: Severity
    affected_radius_km: float = Field(ge=0.0)
    cell_count: int = Field(ge=1)
    # Optional fields - only those the pipeline actually produces.
    variable: str | None = Field(default=None, description="Source variable name.")
    units: str | None = Field(default=None, description="Units of the source variable.")
    max_value: float | None = Field(default=None, description="Maximum physical value in the event.")
    mean_value: float | None = Field(default=None, description="Mean physical value in the event.")

    @field_validator("timestamp")
    @classmethod
    def _normalise_timestamp(cls, value: datetime) -> datetime:
        """Interpret naive timestamps as UTC and convert aware ones to UTC."""
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class EventListResponse(BaseModel):
    """Result of ``GET /api/v1/events``."""

    model_config = ConfigDict(extra="forbid")

    count: int
    events: list[ExtremeWeatherEvent]
    source: str | None = Field(
        default=None, description="Analysis file the events were read from."
    )
    generated_at: datetime | None = Field(default=None, description="Analysis generation time.")
    message: str | None = Field(
        default=None,
        description="Explains an empty result (for example 'no analysis available yet').",
    )
