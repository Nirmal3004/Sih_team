"""Optional ground-truth / event-reference datasets (SIH26078, Phase 7).

This repository ships **no** ground truth. That is a fact about the project, not
a gap in the code: the modules here exist so that if a user supplies a trusted
event catalogue the evaluation can use it - and so that when they do not, the
absence is reported precisely rather than papered over.

What counts as usable ground truth
----------------------------------
An event catalogue is only accepted if its provenance is stated. A file of
timestamps with no source, no issuing body and no matching criteria cannot be
treated as truth, because its own errors would then be scored as the model's.
So :class:`GroundTruth` requires:

* a ``provenance`` string naming who produced it and how;
* documented matching criteria - time window, spatial radius, event type;
* explicit records with time, latitude, longitude and event type.

When the status is not ``AVAILABLE`` every report carries the exact sentence
``"Ground truth unavailable - metric not computed."`` rather than a zero, which
would read as "zero skill" - a different and false claim.

Matching uses the haversine formula, because on a sphere a degree of longitude
is not a constant number of kilometres and a planar approximation would
systematically favour low latitudes.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from extreme_weather.config import KM_PER_DEGREE_LATITUDE
from extreme_weather.ingestion.base import utc_now

__all__ = [
    "GROUND_TRUTH_UNAVAILABLE",
    "NOT_COMPUTED",
    "GroundTruth",
    "GroundTruthError",
    "GroundTruthRecord",
    "GroundTruthStatus",
    "haversine_km",
    "load_ground_truth",
    "match_events",
]


#: The exact sentence every report carries when no reference data exists.
GROUND_TRUTH_UNAVAILABLE = "Ground truth unavailable - metric not computed."

#: Marker used where a value could not be computed.
NOT_COMPUTED = GROUND_TRUTH_UNAVAILABLE


class GroundTruthError(ValueError):
    """Raised when a ground-truth file is present but unusable."""


class GroundTruthStatus:
    """Why a ground-truth dataset is or is not usable.

    Strings rather than an enum, so they serialise straight into a report
    without a custom encoder.
    """

    AVAILABLE = "AVAILABLE"
    UNKNOWN_PROVENANCE = "UNKNOWN_PROVENANCE"
    NO_RECORDS = "NO_RECORDS"
    UNAVAILABLE = "UNAVAILABLE"

    @classmethod
    def all(cls) -> tuple[str, ...]:
        """Return every status value."""
        return (
            cls.AVAILABLE,
            cls.UNKNOWN_PROVENANCE,
            cls.NO_RECORDS,
            cls.UNAVAILABLE,
        )


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres between two lat/lon points.

    The haversine formula is used rather than a planar approximation because a
    degree of longitude shrinks towards the poles; an approximate Euclidean
    distance would overstate east-west separation in the tropics and understate
    it near the poles, which would bias every spatial match.

    Args:
        lat1: Latitude of the first point, degrees.
        lon1: Longitude of the first point, degrees.
        lat2: Latitude of the second point, degrees.
        lon2: Longitude of the second point, degrees.

    Returns:
        The distance in kilometres.
    """
    radius = KM_PER_DEGREE_LATITUDE * (180.0 / math.pi)
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = phi2 - phi1
    delta_lambda = math.radians(lon2 - lon1)
    inner = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * radius * math.asin(min(1.0, math.sqrt(inner)))


@dataclass(frozen=True)
class GroundTruthRecord:
    """One reference event.

    Attributes:
        event_id: Identifier within the reference catalogue.
        time: When the event occurred.
        latitude: Latitude of the event, degrees.
        longitude: Longitude of the event, degrees.
        event_type: Category, matched exactly against the detected type.
        intensity: Optional reference intensity, in the project's units.
    """

    event_id: str
    time: np.datetime64
    latitude: float
    longitude: float
    event_type: str
    intensity: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "event_id": self.event_id,
            "time": str(np.datetime_as_string(self.time, unit="s")),
            "latitude": self.latitude,
            "longitude": self.longitude,
            "event_type": self.event_type,
            "intensity": self.intensity,
        }


@dataclass(frozen=True)
class GroundTruth:
    """A reference catalogue, with the provenance that makes it usable.

    Attributes:
        records: The reference events.
        provenance: Who produced the catalogue and how. Required for use.
        source_file: Where it came from, when file-based.
        time_tolerance_hours: How far apart two events may be and still match.
        spatial_tolerance_km: How far apart two events may be and still match.
        event_types: The event categories the catalogue covers.
        loaded_at: When it was loaded.
    """

    records: tuple[GroundTruthRecord, ...] = ()
    provenance: str = ""
    source_file: str | None = None
    time_tolerance_hours: float = 6.0
    spatial_tolerance_km: float = 50.0
    event_types: tuple[str, ...] = ()
    loaded_at: datetime | None = None

    @property
    def status(self) -> str:
        """One of the :class:`GroundTruthStatus` values."""
        if not self.records:
            return (
                GroundTruthStatus.UNAVAILABLE
                if not self.provenance
                else GroundTruthStatus.NO_RECORDS
            )
        if not self.provenance.strip():
            return GroundTruthStatus.UNKNOWN_PROVENANCE
        return GroundTruthStatus.AVAILABLE

    @property
    def available(self) -> bool:
        """True only when the catalogue has records *and* stated provenance."""
        return self.status == GroundTruthStatus.AVAILABLE

    @property
    def matching_criteria(self) -> dict[str, Any]:
        """The documented criteria used when matching detected to reference."""
        return {
            "time_tolerance_hours": self.time_tolerance_hours,
            "spatial_tolerance_km": self.spatial_tolerance_km,
            "event_types": list(self.event_types),
            "distance_formula": "haversine",
        }

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable description of the ground truth."""
        return {
            "status": self.status,
            "available": self.available,
            "record_count": len(self.records),
            "provenance": self.provenance,
            "source_file": self.source_file,
            "event_types": list(self.event_types),
            "matching_criteria": self.matching_criteria,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
            "note": None if self.available else NOT_COMPUTED,
        }


def empty_ground_truth() -> GroundTruth:
    """Return the explicit "no ground truth" marker object."""
    return GroundTruth(records=(), provenance="", loaded_at=utc_now())


def _parse_time(value: Any) -> np.datetime64:
    """Parse a timestamp into ``datetime64``, accepting ISO strings.

    Raises:
        GroundTruthError: If the value cannot be interpreted as a time.
    """
    if isinstance(value, np.datetime64):
        return value
    try:
        return np.datetime64(str(value))
    except Exception as exc:  # noqa: BLE001 - normalised into a domain error
        raise GroundTruthError(f"'{value}' is not a valid timestamp.") from exc



def load_ground_truth(
    path: Path | str | None,
    *,
    time_tolerance_hours: float = 6.0,
    spatial_tolerance_km: float = 50.0,
) -> GroundTruth:
    """Load an optional reference catalogue from a JSON file.

    A missing file is **not** an error: this project ships no ground truth, so
    "no file" is the expected state and is reported as such. A file that exists
    but is malformed **is** an error, because silently ignoring a reference the
    user meant to use would produce a report that quietly omits skill metrics.

    Args:
        path: Path to the catalogue, or ``None`` when there is none.
        time_tolerance_hours: Match window in time.
        spatial_tolerance_km: Match window in space.

    Returns:
        A :class:`GroundTruth` whose ``available`` flag is False when the
        catalogue is absent, empty, or has no stated provenance.

    Raises:
        GroundTruthError: If the file exists but cannot be parsed, or a record
            is missing a required field.
    """
    if path is None:
        return empty_ground_truth()

    target = Path(path).expanduser()
    if not target.is_file():
        return empty_ground_truth()

    try:
        payload = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GroundTruthError(
            f"Ground-truth file '{target}' exists but could not be read: {exc}"
        ) from exc

    if not isinstance(payload, Mapping):
        raise GroundTruthError(
            f"Ground-truth file '{target}' must contain a JSON object, got "
            f"{type(payload).__name__}."
        )

    raw_records = payload.get("events", payload.get("records", []))
    if isinstance(raw_records, (str, bytes)) or not isinstance(raw_records, Sequence):
        raise GroundTruthError(
            f"Ground-truth file '{target}' must contain a list under 'events'."
        )

    records: list[GroundTruthRecord] = []
    for index, item in enumerate(raw_records):
        if not isinstance(item, Mapping):
            raise GroundTruthError(
                f"Ground-truth record {index} in '{target}' is not an object."
            )
        missing = [
            field
            for field in ("time", "latitude", "longitude", "event_type")
            if field not in item
        ]
        if missing:
            raise GroundTruthError(
                f"Ground-truth record {index} in '{target}' is missing "
                f"{', '.join(missing)}."
            )
        intensity = item.get("intensity")
        records.append(
            GroundTruthRecord(
                event_id=str(item.get("event_id", f"GT-{index:04d}")),
                time=_parse_time(item["time"]),
                latitude=float(item["latitude"]),
                longitude=float(item["longitude"]),
                event_type=str(item["event_type"]),
                intensity=None if intensity is None else float(intensity),
            )
        )

    return GroundTruth(
        records=tuple(records),
        provenance=str(payload.get("provenance", "")),
        source_file=str(target),
        time_tolerance_hours=float(time_tolerance_hours),
        spatial_tolerance_km=float(spatial_tolerance_km),
        event_types=tuple(sorted({record.event_type for record in records})),
        loaded_at=utc_now(),
    )



def match_events(
    detected: Sequence[Mapping[str, Any]],
    truth: GroundTruth,
) -> dict[str, Any]:
    """Match detected events against a reference catalogue.

    Each detected event matches **at most one** reference event (a greedy
    nearest match, taken in increasing time order). Matching requires all three
    of: the same event type, within ``time_tolerance_hours``, and within
    ``spatial_tolerance_km``.

    Args:
        detected: Detected events, each with ``time``, ``latitude``,
            ``longitude`` and ``event_type``.
        truth: The reference catalogue.

    Returns:
        A dict with integer counts, the matched pairs, and a ``note`` that is
        the explicit "not computed" sentence when no ground truth exists.

    Raises:
        GroundTruthError: If a detected event is missing a required field.
    """
    parsed: list[dict[str, Any]] = []
    for index, item in enumerate(detected):
        missing = [
            field
            for field in ("time", "latitude", "longitude", "event_type")
            if field not in item
        ]
        if missing:
            raise GroundTruthError(
                f"Detected event {index} is missing {', '.join(missing)}."
            )
        parsed.append(
            {
                "id": str(item.get("event_id", f"DET-{index:04d}")),
                "time": _parse_time(item["time"]),
                "latitude": float(item["latitude"]),
                "longitude": float(item["longitude"]),
                "event_type": str(item["event_type"]),
            }
        )

    if not truth.available:
        return {
            "true_positives": None,
            "false_positives": None,
            "false_negatives": None,
            "matches": [],
            "note": NOT_COMPUTED,
            "ground_truth_status": truth.status,
        }

    unmatched = list(truth.records)
    matches: list[dict[str, Any]] = []
    true_positives = 0
    tolerance = np.timedelta64(int(truth.time_tolerance_hours * 3600), "s")
    order = sorted(range(len(parsed)), key=lambda i: parsed[i]["time"])

    for index in order:
        candidate = parsed[index]
        best: GroundTruthRecord | None = None
        best_distance = float("inf")
        for record in unmatched:
            if record.event_type != candidate["event_type"]:
                continue
            if abs(record.time - candidate["time"]) > tolerance:
                continue
            distance = haversine_km(
                candidate["latitude"],
                candidate["longitude"],
                record.latitude,
                record.longitude,
            )
            if distance <= truth.spatial_tolerance_km and distance < best_distance:
                best = record
                best_distance = distance

        if best is not None:
            unmatched.remove(best)
            true_positives += 1
            matches.append(
                {
                    "detected_id": candidate["id"],
                    "reference_id": best.event_id,
                    "distance_km": round(best_distance, 3),
                }
            )

    return {
        "true_positives": true_positives,
        "false_positives": len(parsed) - true_positives,
        "false_negatives": len(unmatched),
        "matches": matches,
        "note": None,
        "ground_truth_status": truth.status,
    }

