"""Tracking result models (SIH26078, Phase 2).

The strongly typed track objects live in :mod:`extreme_weather.schemas.track`
because they are part of the public API contract; they are re-exported here so
callers need a single import. This module adds the container type that describes
a whole tracking run.

The stored schema is intentionally versioned (``schema_version``) and stable:
the Phase 2 CLI writes it, the tracking API reads it, and nothing else in the
project depends on its internal layout.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Mapping

from extreme_weather.schemas.track import (
    EventTrack,
    TrackListResponse,
    TrackPoint,
    TrackStatus,
    TrajectoryResponse,
    TrajectoryWaypoint,
)

__all__ = [
    "EventTrack",
    "TrackListResponse",
    "TrackPoint",
    "TrackResult",
    "TrackStatus",
    "TrajectoryResponse",
    "TrajectoryWaypoint",
]


@dataclass(frozen=True)
class TrackResult:
    """Everything one tracking run produced.

    Attributes:
        tracks: The tracks, ordered by start time and track id.
        event_count: Number of input events that were considered.
        matched_event_count: Number of events that belong to a multi-point track.
        generated_at: UTC timestamp of the run.
        metadata: Provenance (matching configuration, event source, disclaimers).
    """

    tracks: tuple[EventTrack, ...] = ()
    event_count: int = 0
    matched_event_count: int = 0
    generated_at: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def track_count(self) -> int:
        """Number of tracks."""
        return len(self.tracks)

    @property
    def duration_hours(self) -> float:
        """Total tracked time, i.e. the sum of all track durations."""
        return sum(track.duration_hours for track in self.tracks)

    def by_status(self, status: TrackStatus | str) -> list[EventTrack]:
        """Return the tracks in one lifecycle state."""
        wanted = status if isinstance(status, TrackStatus) else TrackStatus.from_value(str(status))
        return [track for track in self.tracks if track.status is wanted]

    def longest(self) -> EventTrack | None:
        """Return the track with the largest duration (``None`` when empty)."""
        if not self.tracks:
            return None
        return max(self.tracks, key=lambda track: (track.duration_hours, track.track_id))

    def fastest(self) -> EventTrack | None:
        """Return the track with the highest maximum speed (``None`` when empty)."""
        if not self.tracks:
            return None
        return max(self.tracks, key=lambda track: (track.max_speed_kmh, track.track_id))

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary with a stable key order."""
        return {
            "schema_version": "1.0",
            "project": "SIH26078",
            "service": "DeepWeather AI",
            "phase": "Phase 2 - event tracking (prototype)",
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
            "event_count": self.event_count,
            "track_count": self.track_count,
            "matched_event_count": self.matched_event_count,
            "total_duration_hours": self.duration_hours,
            "status_counts": {
                status.value: len(self.by_status(status))
                for status in TrackStatus
            },
            "metadata": dict(self.metadata),
            "tracks": [track.model_dump(mode="json") for track in self.tracks],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "TrackResult":
        """Rebuild a result from :meth:`as_dict` output."""
        tracks = tuple(EventTrack.model_validate(item) for item in payload.get("tracks", []))
        return cls(
            tracks=tracks,
            event_count=int(payload.get("event_count", len(tracks))),
            matched_event_count=int(payload.get("matched_event_count", 0)),
            generated_at=(
                datetime.fromisoformat(payload["generated_at"])
                if payload.get("generated_at")
                else None
            ),
            metadata=dict(payload.get("metadata", {})),
        )

    def to_asdict(self) -> dict[str, Any]:
        """Alias of :meth:`as_dict` kept for symmetry with the Phase 1 reports."""
        return asdict(self)
