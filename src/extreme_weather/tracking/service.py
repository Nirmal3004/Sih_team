"""Tracking service (SIH26078, Phase 2).

This module is the business layer between the Phase 1 events and the tracking
API/CLI:

```text
events.json (EventStore)
        ↓  load
sort chronologically  (input order is irrelevant)
        ↓
group by valid time
        ↓
match consecutive steps (matching.MatchingConfig)
        ↓
build tracks + movement metrics (metrics.py)
        ↓
TrackResult -> tracks.json (store.TrackStore)
```

No database, no global state: every service instance carries its own matching
configuration and store.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from extreme_weather.anomaly.service import EventStore
from extreme_weather.config import PACKAGE_VERSION, PROJECT_ID
from extreme_weather.schemas.event import ExtremeWeatherEvent
from extreme_weather.schemas.track import EventTrack
from extreme_weather.tracking.matching import MatchingConfig
from extreme_weather.tracking.models import TrackResult
from extreme_weather.tracking.store import TrackStore, TrackingStoreError
from extreme_weather.tracking.tracker import EventTracker

__all__ = ["TrackingError", "TrackingService", "track_events"]

#: Attached to every result so consumers know what they are looking at.
DATA_DISCLAIMER = (
    "Prototype tracking of synthetic events. Not an operational forecast, not a real-time "
    "warning product, and not verified against observed weather."
)


class TrackingError(RuntimeError):
    """Raised when a tracking run cannot be completed."""


def _utc_now() -> datetime:
    """Return the current UTC time (second resolution)."""
    return datetime.now(timezone.utc).replace(microsecond=0)


class TrackingService:
    """Load events, build tracks and persist the result.

    Example::

        service = TrackingService()
        result = service.run()                      # reads data/processed/events.json
        print(result.track_count, "tracks")
    """

    def __init__(
        self,
        *,
        event_store: EventStore | None = None,
        track_store: TrackStore | None = None,
        config: MatchingConfig | None = None,
    ) -> None:
        self.event_store = event_store or EventStore()
        self.track_store = track_store or TrackStore()
        self.config = config or MatchingConfig()

    def load_events(self, events_path: Path | str | None = None) -> list[ExtremeWeatherEvent]:
        """Load the Phase 1 events.

        Args:
            events_path: Explicit path to an events JSON file; defaults to the
                configured :class:`~extreme_weather.anomaly.service.EventStore`.

        Returns:
            The stored events (in file order - the tracker sorts them).

        Raises:
            TrackingError: If no analysis file exists or it cannot be parsed.
        """
        store = EventStore(Path(events_path)) if events_path is not None else self.event_store
        try:
            analysis = store.load()
        except Exception as exc:  # noqa: BLE001 - normalised into a domain error
            raise TrackingError(
                f"Could not read the Phase 1 analysis: {type(exc).__name__}: {exc}"
            ) from exc
        if analysis is None:
            raise TrackingError(
                f"No event analysis found at '{store.path}'. Run "
                "'python scripts/detect_anomalies.py' first."
            )
        return list(analysis.events)

    def track(self, events: Iterable[ExtremeWeatherEvent]) -> list[EventTrack]:
        """Track the given events (the input order is irrelevant)."""
        return EventTracker(self.config).track(events)

    def run(
        self,
        events: Iterable[ExtremeWeatherEvent] | None = None,
        *,
        events_path: Path | str | None = None,
        write_output: bool = False,
    ) -> TrackResult:
        """Track the Phase 1 events and optionally persist the result.

        Args:
            events: Events to track; loaded from the event store when omitted.
            events_path: Explicit path of the events JSON file.
            write_output: Write ``tracks.json`` into the processed directory.

        Returns:
            A :class:`~extreme_weather.tracking.models.TrackResult`.

        Raises:
            TrackingError: If the events cannot be loaded or the result cannot be
                written.
        """
        loaded = list(events) if events is not None else self.load_events(events_path)
        tracks = self.track(loaded)
        matched = sum(track.point_count for track in tracks if track.point_count > 1)

        result = TrackResult(
            tracks=tuple(tracks),
            event_count=len(loaded),
            matched_event_count=matched,
            generated_at=_utc_now(),
            metadata={
                "tracker_version": PACKAGE_VERSION,
                "project": PROJECT_ID,
                "phase": "Phase 2 - event tracking (prototype)",
                "event_source": str(events_path or self.event_store.path),
                "matching": self.config.model_dump(),
                "matching_note": (
                    "cost = (distance_weight * d/max_distance + intensity_weight * "
                    "intensity_mismatch) / (distance_weight + intensity_weight); pairs are "
                    "assigned greedily by increasing cost with deterministic tie-breaking."
                ),
                "lifecycle_rule": (
                    "NEW = single point at the last processed step, ACTIVE = last point at the "
                    "last processed step, ENDED = stopped being matched earlier."
                ),
                "data_disclaimer": DATA_DISCLAIMER,
            },
        )

        if write_output:
            try:
                self.track_store.save(result)
            except TrackingStoreError as exc:
                raise TrackingError(str(exc)) from exc
        return result

    def load_tracks(self) -> TrackResult | None:
        """Return the stored tracking result, or ``None`` when absent."""
        return self.track_store.load()


def track_events(
    events_path: Path | str | None = None,
    *,
    tracks_path: Path | str | None = None,
    config: MatchingConfig | None = None,
    write_output: bool = True,
) -> TrackResult:
    """Convenience wrapper used by the CLI and the tests.

    Args:
        events_path: Phase 1 events file; defaults to the configured store.
        tracks_path: Where to write ``tracks.json``; defaults to the configured
            track store.
        config: Matching configuration.
        write_output: Persist the result.

    Returns:
        The :class:`~extreme_weather.tracking.models.TrackResult`.
    """
    service = TrackingService(
        event_store=EventStore(Path(events_path)) if events_path is not None else EventStore(),
        track_store=TrackStore(Path(tracks_path)) if tracks_path is not None else TrackStore(),
        config=config,
    )
    return service.run(events_path=events_path, write_output=write_output)
