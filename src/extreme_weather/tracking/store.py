"""File-backed store for tracking results (SIH26078, Phase 2).

``tracks.json`` is written by ``scripts/track_events.py`` (or by
:meth:`TrackingService.run`) and read by the tracking API. There is no database
in Phase 2: a versioned JSON file keeps the demo reproducible and the
infrastructure minimal.

The store only ever writes into the **processed** directory - ``data/raw`` is
never touched.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from extreme_weather.config import Settings, get_settings
from extreme_weather.schemas.event import EventType
from extreme_weather.schemas.track import EventTrack, TrackStatus
from extreme_weather.tracking.models import TrackResult

__all__ = ["TrackStore", "TrackingStoreError"]


class TrackingStoreError(RuntimeError):
    """Raised when a tracking file cannot be read or written."""


class TrackStore:
    """Read/write access to the latest tracking result.

    Args:
        path: JSON file holding the tracks. Defaults to
            ``<DATA_DIR>/processed/<settings.tracks_filename>``.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        settings = get_settings()
        self._path = (
            Path(path) if path is not None else settings.processed_dir / settings.tracks_filename
        )

    @property
    def path(self) -> Path:
        """Location of the tracking file."""
        return self._path

    def exists(self) -> bool:
        """True when a tracking file is present."""
        return self._path.is_file()

    def save(self, result: TrackResult) -> Path:
        """Write ``result`` as JSON, creating parent directories as needed.

        Raises:
            TrackingStoreError: If the file cannot be written.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as handle:
                json.dump(result.as_dict(), handle, indent=2, ensure_ascii=False, default=str)
                handle.write("\n")
        except OSError as exc:
            raise TrackingStoreError(f"Could not write tracks to '{self._path}': {exc}") from exc
        return self._path

    def load(self) -> TrackResult | None:
        """Read the stored tracking result, or ``None`` when nothing is stored.

        Raises:
            TrackingStoreError: If the file exists but cannot be parsed.
        """
        if not self._path.is_file():
            return None
        try:
            with self._path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise TrackingStoreError(
                f"Could not read tracking file '{self._path}': {exc}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise TrackingStoreError(
                f"Tracking file '{self._path}' must contain a JSON object, got "
                f"{type(payload).__name__}."
            )
        return TrackResult.from_dict(payload)

    def get(self, track_id: str) -> EventTrack | None:
        """Return one track by id, or ``None`` when it is not present."""
        result = self.load()
        if result is None:
            return None
        for track in result.tracks:
            if track.track_id == track_id:
                return track
        return None

    def list(
        self,
        *,
        event_type: EventType | str | None = None,
        status: TrackStatus | str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[EventTrack]:
        """Return the stored tracks matching the given filters.

        * ``event_type``/``status`` accept strings or enum members;
        * ``start_time`` keeps tracks that **end** at or after the given time;
        * ``end_time`` keeps tracks that **start** at or before the given time.
        """
        result = self.load()
        if result is None:
            return []

        tracks = list(result.tracks)
        if event_type is not None:
            wanted = (
                event_type
                if isinstance(event_type, EventType)
                else EventType.from_value(str(event_type))
            )
            tracks = [track for track in tracks if track.event_type is wanted]
        if status is not None:
            wanted_status = (
                status if isinstance(status, TrackStatus) else TrackStatus.from_value(str(status))
            )
            tracks = [track for track in tracks if track.status is wanted_status]
        if start_time is not None:
            tracks = [track for track in tracks if track.end_time >= start_time]
        if end_time is not None:
            tracks = [track for track in tracks if track.start_time <= end_time]
        return tracks

    def clear(self) -> None:
        """Delete the stored tracking file if it exists."""
        self._path.unlink(missing_ok=True)

    def result(self) -> TrackResult | None:
        """Alias of :meth:`load` used by the service and the CLI."""
        return self.load()
