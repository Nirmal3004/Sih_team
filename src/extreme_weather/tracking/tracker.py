"""Track construction and lifecycle (SIH26078, Phase 2).

The tracker walks the detected events **chronologically** (the input order is
irrelevant), groups them by valid time and connects consecutive steps with
:func:`~extreme_weather.tracking.matching.match_events`.

Lifecycle
---------
```text
step t1  event A                 -> new track, status NEW
step t2  event A' matched to A   -> track grows, status ACTIVE
step t3  event A'' matched to A'  -> track grows, status ACTIVE
step t4  no event nearby         -> track is no longer extended
```

Status is decided when the run finishes, which makes it deterministic:

* **NEW** - the track holds exactly one point and that point is in the last
  processed step (the event appeared at the end of the window);
* **ACTIVE** - the track has at least two points *and* its last point is in the
  last processed step (it was still moving);
* **ENDED** - the track stopped being matched before the last processed step.

Metrics for every point are computed with
:mod:`extreme_weather.tracking.metrics`: the first point has no predecessor, so
its speed is ``0.0`` and its direction is ``"CALM"``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterable, Mapping, Sequence

from extreme_weather.schemas.event import ExtremeWeatherEvent, EventType, Severity
from extreme_weather.schemas.track import EventTrack, TrackPoint, TrackStatus
from extreme_weather.tracking.matching import MatchingConfig, match_events
from extreme_weather.tracking.metrics import (
    bearing_degrees,
    build_track_id,
    compass_direction,
    elapsed_hours,
    haversine_distance_km,
    intensity_trend,
    movement_speed_kmh,
    total_distance_km,
)

__all__ = ["EventTracker", "TrackingError", "group_events_by_time", "sort_events"]


class TrackingError(ValueError):
    """Raised when the tracker receives inconsistent input."""


def sort_events(events: Iterable[ExtremeWeatherEvent]) -> list[ExtremeWeatherEvent]:
    """Return the events sorted by ``(timestamp, event_id)``.

    Sorting makes the result independent of the order in which events were
    produced, loaded or shuffled.
    """
    return sorted(events, key=lambda event: (event.timestamp, event.event_id))


def group_events_by_time(
    events: Iterable[ExtremeWeatherEvent],
) -> dict[datetime, list[ExtremeWeatherEvent]]:
    """Group chronologically sorted events by their valid time.

    Returns:
        ``{timestamp: [events at that timestamp]}`` with both the keys and the
        values in deterministic order.
    """
    grouped: dict[datetime, list[ExtremeWeatherEvent]] = {}
    for event in sort_events(events):
        grouped.setdefault(event.timestamp, []).append(event)
    return dict(sorted(grouped.items()))


@dataclass
class _OpenTrack:
    """Mutable bookkeeping for a track while the run is in progress."""

    event_type: EventType
    events: list[ExtremeWeatherEvent] = field(default_factory=list)
    track_id: str | None = None

    @property
    def last_event(self) -> ExtremeWeatherEvent:
        return self.events[-1]

    def append(self, event: ExtremeWeatherEvent) -> None:
        self.events.append(event)


def build_point(
    event: ExtremeWeatherEvent, previous: ExtremeWeatherEvent | None
) -> TrackPoint:
    """Build a :class:`TrackPoint` with movement metrics relative to ``previous``.

    The first point of a track (no ``previous``) reports ``speed_kmh = 0.0``,
    ``distance_from_previous_km = 0.0``, ``bearing_degrees = None`` and
    ``direction = "CALM"`` - there is no movement to measure yet.
    """
    if previous is None:
        return TrackPoint(
            event_id=event.event_id,
            timestamp=event.timestamp,
            latitude=event.latitude,
            longitude=event.longitude,
            intensity=event.intensity,
            anomaly_score=event.anomaly_score,
            severity=event.severity,
            speed_kmh=0.0,
            bearing_degrees=None,
            direction=compass_direction(None),
            distance_from_previous_km=0.0,
            cell_count=event.cell_count,
        )

    distance = haversine_distance_km(
        previous.latitude, previous.longitude, event.latitude, event.longitude
    )
    hours = elapsed_hours(previous.timestamp, event.timestamp)
    speed = movement_speed_kmh(distance, hours)
    bearing = bearing_degrees(
        previous.latitude, previous.longitude, event.latitude, event.longitude
    )
    return TrackPoint(
        event_id=event.event_id,
        timestamp=event.timestamp,
        latitude=event.latitude,
        longitude=event.longitude,
        intensity=event.intensity,
        anomaly_score=event.anomaly_score,
        severity=event.severity,
        speed_kmh=speed,
        bearing_degrees=bearing,
        direction=compass_direction(bearing),
        distance_from_previous_km=distance,
        cell_count=event.cell_count,
    )


def finalise_track(
    open_track: _OpenTrack,
    track_id: str,
    last_processed: datetime,
) -> EventTrack:
    """Turn an open track into a validated :class:`EventTrack`.

    Args:
        open_track: The accumulated events of the track.
        track_id: Deterministic identifier assigned by the tracker.
        last_processed: Timestamp of the last detection step of the run.
    """
    events = open_track.events
    points = tuple(
        build_point(event, events[index - 1] if index else None)
        for index, event in enumerate(events)
    )

    if len(points) == 1:
        status = TrackStatus.NEW
    elif points[-1].timestamp >= last_processed:
        status = TrackStatus.ACTIVE
    else:
        status = TrackStatus.ENDED

    intensities = [point.intensity for point in points]
    speeds = [point.speed_kmh for point in points]
    coordinates = [(point.latitude, point.longitude) for point in points]
    last_point = points[-1]

    return EventTrack(
        track_id=track_id,
        event_type=open_track.event_type,
        status=status,
        start_time=points[0].timestamp,
        end_time=last_point.timestamp,
        duration_hours=elapsed_hours(points[0].timestamp, last_point.timestamp),
        points=points,
        peak_intensity=max(intensities),
        mean_intensity=sum(intensities) / len(intensities),
        max_speed_kmh=max(speeds),
        mean_speed_kmh=sum(speeds) / len(speeds),
        distance_travelled_km=total_distance_km(coordinates),
        intensity_trend=intensity_trend(intensities[0], intensities[-1]),
        direction=last_point.direction,
        bearing_degrees=last_point.bearing_degrees,
        variable=events[-1].variable,
        units=events[-1].units,
        peak_severity=max((event.severity for event in events), key=lambda item: item.rank),
    )


class EventTracker:
    """Connect consecutive detections into :class:`EventTrack` objects.

    The tracker holds no global state: two trackers with different
    :class:`MatchingConfig` objects can run side by side (this is what the tests
    rely on).

    Example::

        tracker = EventTracker(MatchingConfig(max_distance_km=150.0))
        tracks = tracker.track(events)
    """

    def __init__(
        self,
        config: MatchingConfig | None = None,
        *,
        match: bool = True,
    ) -> None:
        """
        Args:
            config: Matching configuration; the default is used when omitted.
            match: Set to ``False`` to build one single-point track per event
                (useful as a diagnostic baseline).
        """
        self.config = config or MatchingConfig()
        self.match = bool(match)

    def track(self, events: Iterable[ExtremeWeatherEvent]) -> list[EventTrack]:
        """Track ``events`` and return the tracks in deterministic order.

        The result does not depend on the order of ``events``.
        """
        grouped = group_events_by_time(events)
        if not grouped:
            return []

        timestamps = list(grouped)
        last_processed = timestamps[-1]
        open_tracks: list[_OpenTrack] = []
        id_counters: dict[tuple[str, datetime], int] = {}

        for step_index, timestamp in enumerate(timestamps):
            current = grouped[timestamp]
            by_id: Mapping[str, ExtremeWeatherEvent] = {event.event_id: event for event in current}

            continuations: dict[int, ExtremeWeatherEvent] = {}
            if self.match and open_tracks and step_index > 0:
                previous_events = [track.last_event for track in open_tracks]
                result = match_events(previous_events, current, self.config)
                previous_index = {
                    track.last_event.event_id: position for position, track in enumerate(open_tracks)
                }
                for previous_id, next_id in result.pairs.items():
                    continuations[previous_index[previous_id]] = by_id[next_id]

            still_open: list[_OpenTrack] = []
            for position, track in enumerate(open_tracks):
                continuation = continuations.get(position)
                if continuation is not None:
                    track.append(continuation)
                # A track stays open whether or not it was continued; an
                # unmatched track is simply waiting for the next step.
                still_open.append(track)
            open_tracks = still_open

            matched_next = {
                event.event_id for event in continuations.values()
            }
            for event in current:
                if event.event_id in matched_next:
                    continue
                new_track = _OpenTrack(event_type=event.event_type, events=[event])
                key = (event.event_type.value, timestamp)
                id_counters[key] = id_counters.get(key, 0) + 1
                new_track.track_id = build_track_id(event.event_type, timestamp, id_counters[key])
                open_tracks.append(new_track)

        tracks = [
            finalise_track(track, track.track_id or build_track_id(track.event_type, track.events[0].timestamp, 1), last_processed)
            for track in open_tracks
        ]
        tracks.sort(key=lambda item: (item.start_time, item.event_type.value, item.track_id))
        return tracks
