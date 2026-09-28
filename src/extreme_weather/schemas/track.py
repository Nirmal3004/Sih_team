"""Pydantic schemas for extreme weather event tracks (SIH26078, Phase 2).

These models are the *external* contract of the tracking service. The tracking
modules in :mod:`extreme_weather.tracking` produce them directly and the FastAPI
router reuses them for responses, so the JSON representation of a track is
defined exactly once (mirroring :mod:`extreme_weather.schemas.event` for Phase 1).

Semantics
---------
* ``start_time``/``end_time``/``points[*].timestamp`` are **valid times** of the
  detected events (UTC; naive values are interpreted as UTC).
* ``speed_kmh``/``bearing_degrees``/``direction`` of the **first** point of a
  track are ``0.0``/``None``/``"CALM"`` because there is no previous position.
* ``duration_hours`` is the elapsed time between the first and last point; a
  single-point track has duration ``0.0``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from extreme_weather.schemas.event import EventType, Severity

__all__ = [
    "EventTrack",
    "TrackListResponse",
    "TrackPoint",
    "TrackStatus",
    "TrajectoryResponse",
    "TrajectoryWaypoint",
]


class TrackStatus(StrEnum):
    """Lifecycle state of a track after the tracking run."""

    NEW = "NEW"
    ACTIVE = "ACTIVE"
    ENDED = "ENDED"

    @classmethod
    def from_value(cls, value: str) -> "TrackStatus":
        """Parse a status from a case-insensitive string.

        Raises:
            ValueError: If the status is unknown.
        """
        try:
            return cls(str(value).strip().upper())
        except ValueError as exc:
            supported = ", ".join(member.value for member in cls)
            raise ValueError(
                f"Unknown track status '{value}'. Supported statuses: {supported}."
            ) from exc


def _normalise_timestamp(value: datetime) -> datetime:
    """Interpret naive timestamps as UTC and convert aware ones to UTC."""
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


class TrackPoint(BaseModel):
    """One detected event inside a track, enriched with movement metrics.

    * ``distance_from_previous_km``, ``speed_kmh`` and ``bearing_degrees`` are
      measured from the previous point; the first point of a track has
      ``0.0``/``0.0``/``None``.
    * ``direction`` is a 16-point compass label, or ``"CALM"`` when the event did
      not move (or when it is the first point of the track).
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    event_id: str = Field(min_length=1)
    timestamp: datetime
    latitude: float = Field(ge=-90.0, le=90.0)
    longitude: float = Field(ge=-180.0, le=180.0)
    intensity: float
    anomaly_score: float
    severity: Severity
    speed_kmh: float = Field(default=0.0, ge=0.0)
    bearing_degrees: float | None = Field(default=None, ge=0.0, lt=360.0)
    direction: str = Field(default="CALM", min_length=1)
    distance_from_previous_km: float = Field(default=0.0, ge=0.0)
    cell_count: int | None = Field(default=None, ge=1)

    @field_validator("timestamp")
    @classmethod
    def _timestamp_to_utc(cls, value: datetime) -> datetime:
        return _normalise_timestamp(value)


class EventTrack(BaseModel):
    """A continuous track of one extreme weather event over time.

    * ``status`` - ``NEW`` (single point at the end of the run), ``ACTIVE``
      (still moving in the last processed step) or ``ENDED`` (no longer matched
      before the end of the run).
    * ``points`` - chronologically ordered, at least one.
    * ``intensity_trend`` - ``STRENGTHENING``, ``WEAKENING`` or ``STABLE``.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    track_id: str = Field(min_length=1)
    event_type: EventType
    status: TrackStatus
    start_time: datetime
    end_time: datetime
    duration_hours: float = Field(ge=0.0)
    points: tuple[TrackPoint, ...] = Field(min_length=1)
    peak_intensity: float
    mean_intensity: float
    max_speed_kmh: float = Field(ge=0.0)
    mean_speed_kmh: float = Field(ge=0.0)
    distance_travelled_km: float = Field(ge=0.0)
    intensity_trend: str = Field(min_length=1)
    direction: str = Field(default="CALM", min_length=1)
    bearing_degrees: float | None = Field(default=None, ge=0.0, lt=360.0)
    variable: str | None = None
    units: str | None = None
    peak_severity: Severity | None = None

    @field_validator("start_time", "end_time")
    @classmethod
    def _timestamps_to_utc(cls, value: datetime) -> datetime:
        return _normalise_timestamp(value)

    @model_validator(mode="after")
    def _validate_track(self) -> "EventTrack":
        """Check the internal consistency of the trajectory."""
        if self.end_time < self.start_time:
            raise ValueError("end_time must not be earlier than start_time.")
        stamps = [point.timestamp for point in self.points]
        if any(later < earlier for earlier, later in zip(stamps, stamps[1:])):
            raise ValueError("Track points must be ordered chronologically.")
        if stamps[0] != self.start_time or stamps[-1] != self.end_time:
            raise ValueError("start_time/end_time must match the first and last track point.")
        if self.peak_intensity < max(point.intensity for point in self.points):
            raise ValueError("peak_intensity must be the maximum point intensity.")
        if self.max_speed_kmh > max(point.speed_kmh for point in self.points) + 1e-9:
            raise ValueError("max_speed_kmh must be the maximum point speed.")
        if self.status is TrackStatus.NEW and len(self.points) > 1:
            raise ValueError("A NEW track must contain exactly one point.")
        if self.status is not TrackStatus.NEW and len(self.points) < 2:
            raise ValueError("An ACTIVE or ENDED track must contain at least two points.")
        return self

    @property
    def point_count(self) -> int:
        """Number of events in the track.

        Deliberately *not* a serialised field: the JSON payload already carries
        ``points`` (the consumer can use its length) and keeping computed keys out
        of the model makes ``model_dump()`` round-trip through
        ``model_validate()`` even with ``extra="forbid"``.
        """
        return len(self.points)


class TrackListResponse(BaseModel):
    """Result of ``GET /api/v1/tracks``."""

    model_config = ConfigDict(extra="forbid")

    count: int
    tracks: list[EventTrack]
    source: str | None = Field(default=None, description="Tracking file the tracks were read from.")
    generated_at: datetime | None = Field(default=None, description="Tracking generation time.")
    message: str | None = Field(
        default=None,
        description="Explains an empty result (for example 'no tracking available yet').",
    )


class TrajectoryWaypoint(BaseModel):
    """One lightweight position of a trajectory (used by the map layer)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    timestamp: datetime
    longitude: float = Field(ge=-180.0, le=180.0)
    latitude: float = Field(ge=-90.0, le=90.0)
    intensity: float
    speed_kmh: float = Field(ge=0.0)
    direction: str
    event_id: str


class TrajectoryResponse(BaseModel):
    """Result of ``GET /api/v1/tracks/{track_id}/trajectory``."""

    model_config = ConfigDict(extra="forbid")

    track_id: str
    event_type: EventType
    status: TrackStatus
    point_count: int = Field(ge=1)
    distance_travelled_km: float = Field(ge=0.0)
    trajectory: list[TrajectoryWaypoint]
