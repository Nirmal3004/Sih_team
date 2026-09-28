"""Extreme weather event tracking (SIH26078, Phase 2).

```text
Phase 1 ExtremeWeatherEvent objects
        ↓  service.TrackingService (load + sort + group by time)
        ↓  matching.match_events  (deterministic one-to-one, gated by MatchingConfig)
        ↓  metrics                 (distance, speed, bearing, compass, intensity trend)
        ↓  tracker.EventTracker    (lifecycle NEW / ACTIVE / ENDED)
        ↓
EventTrack objects -> data/processed/tracks.json -> GET /api/v1/tracks
```

Every track is a *deterministic* function of its input events: the tracker sorts
them chronologically, matching breaks ties on stable identifiers, and track ids
encode the event type, the start time and a counter.

GNN-based tracking (Phase 3) is **FUTURE** and will refine the linking decision
that :mod:`extreme_weather.tracking.matching` makes here with a simple,
explainable cost function.

Quick start::

    from extreme_weather.tracking import TrackingService

    result = TrackingService().run()
    print(result.track_count, "tracks")
"""

from __future__ import annotations

from extreme_weather.schemas.track import (
    EventTrack,
    TrackListResponse,
    TrackPoint,
    TrackStatus,
    TrajectoryResponse,
    TrajectoryWaypoint,
)
from extreme_weather.tracking.matching import (
    MatchResult,
    MatchingConfig,
    TrackingError,
    intensity_mismatch,
    match_cost,
    match_events,
)
from extreme_weather.tracking.metrics import (
    CALM,
    CARDINAL_DIRECTIONS,
    EARTH_RADIUS_KM,
    TREND_STABLE,
    TREND_STRENGTHENING,
    TREND_WEAKENING,
    bearing_degrees,
    build_track_id,
    compass_direction,
    elapsed_hours,
    haversine_distance_km,
    intensity_trend,
    movement_speed_kmh,
    total_distance_km,
)
from extreme_weather.tracking.models import TrackResult
from extreme_weather.tracking.service import DATA_DISCLAIMER, TrackingService, track_events
from extreme_weather.tracking.store import TrackStore, TrackingStoreError
from extreme_weather.tracking.tracker import (
    EventTracker,
    group_events_by_time,
    sort_events,
)

__all__ = [
    "CALM",
    "CARDINAL_DIRECTIONS",
    "DATA_DISCLAIMER",
    "EARTH_RADIUS_KM",
    "TREND_STABLE",
    "TREND_STRENGTHENING",
    "TREND_WEAKENING",
    "EventTrack",
    "EventTracker",
    "MatchResult",
    "MatchingConfig",
    "TrackListResponse",
    "TrackPoint",
    "TrackResult",
    "TrackStatus",
    "TrackStore",
    "TrackingError",
    "TrackingService",
    "TrackingStoreError",
    "TrajectoryResponse",
    "TrajectoryWaypoint",
    "bearing_degrees",
    "build_track_id",
    "compass_direction",
    "elapsed_hours",
    "group_events_by_time",
    "haversine_distance_km",
    "intensity_mismatch",
    "intensity_trend",
    "match_cost",
    "match_events",
    "movement_speed_kmh",
    "sort_events",
    "total_distance_km",
    "track_events",
]
