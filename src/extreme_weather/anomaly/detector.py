"""Extreme grid-cell detection (SIH26078, Phase 1).

```text
weather field -> anomaly score -> threshold ladder -> boolean mask
```

The detector is deliberately thin: it turns a score field into a mask (and,
optionally, a severity field) using a
:class:`~extreme_weather.anomaly.thresholds.ThresholdLadder`. It never
hard-codes numbers and never mutates its inputs.

Guarantees
----------
* the returned mask keeps ``time``, ``latitude``/``longitude``, all coordinates
  and the useful attributes of the input;
* the mask dtype is ``bool`` (the severity field is object-dtype);
* NaN scores are **never** extreme - they become ``False``/``NORMAL``, so a
  missing cell can not create a false event;
* the detection level is the first (lowest) ladder threshold, so
  ``mask == (severity != NORMAL)`` always holds.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from extreme_weather.anomaly.thresholds import classify_scores
from extreme_weather.data.validator import TIME_DIM
from extreme_weather.schemas.event import EventType, Severity

__all__ = [
    "DetectionError",
    "detect_extreme_cells",
    "detection_summary",
    "mask_from_ladder",
]


class DetectionError(ValueError):
    """Raised when detection is requested with inconsistent inputs."""


def _event_type_of(value: EventType | str) -> EventType:
    """Normalise an event type argument."""
    if isinstance(value, EventType):
        return value
    return EventType.from_value(str(value))


def mask_from_ladder(
    scores: xr.DataArray,
    ladder: object,
    *,
    name: str = "extreme_mask",
) -> xr.DataArray:
    """Return the boolean extreme-cell mask of ``scores`` for ``ladder``.

    Args:
        scores: Score field with the dimensions that must be preserved.
        ladder: Threshold ladder (anything exposing ``thresholds``), typically a
            :class:`~extreme_weather.anomaly.thresholds.ThresholdLadder`.
        name: Name of the returned DataArray.

    Returns:
        Boolean DataArray; ``True`` where the score reaches the first threshold.

    Raises:
        DetectionError: If ``scores`` is not a DataArray or ``ladder`` is unusable.
    """
    if not isinstance(scores, xr.DataArray):
        raise DetectionError(
            f"detect_extreme_cells expects an xarray.DataArray, got {type(scores).__name__}."
        )
    thresholds = getattr(ladder, "thresholds", None)
    if thresholds is None or len(thresholds) == 0:
        raise DetectionError(f"A threshold ladder with at least one step is required, got {ladder!r}.")

    level = float(np.min(np.asarray(thresholds, dtype="float64")))
    values = np.asarray(scores.values, dtype="float64")
    mask_values = np.isfinite(values) & (values >= level)

    mask = xr.DataArray(mask_values, dims=scores.dims, coords=scores.coords, name=name)
    mask.attrs = {
        "long_name": "extreme weather cell mask",
        "threshold": level,
        "ladder": getattr(ladder, "name", "custom"),
        "score_name": getattr(ladder, "score_name", ""),
        "flag_values": np.asarray([0, 1], dtype="int8"),
        "flag_meanings": "normal extreme",
        "comment": "Prototype detection mask (SIH26078 Phase 1).",
    }
    return mask


def detect_extreme_cells(
    scores: xr.DataArray,
    ladder: object,
    *,
    event_type: EventType | str | None = None,
    severity_field: bool = True,
    name: str = "extreme_mask",
) -> xr.DataArray:
    """Detect extreme grid cells and record their severity classes.

    Args:
        scores: Anomaly score field, typically from
            :mod:`extreme_weather.anomaly.scores`.
        ladder: Threshold ladder to apply.
        event_type: Optional event type recorded in the output attributes. When
            given it must match the ladder's event type, which catches
            mismatched configurations early.
        severity_field: Also classify the field into severities (used to annotate
            the mask attributes with per-class counts).
        name: Name of the returned mask.

    Returns:
        The boolean mask. Use
        :func:`~extreme_weather.anomaly.thresholds.classify_scores` for the
        severity array itself.

    Raises:
        DetectionError: If inputs are inconsistent or the event type contradicts
            the ladder.
    """
    mask = mask_from_ladder(scores, ladder, name=name)

    if event_type is not None:
        wanted = _event_type_of(event_type)
        ladder_type = getattr(ladder, "event_type", None)
        if ladder_type is not None and wanted is not _event_type_of(ladder_type):
            raise DetectionError(
                f"Event type {wanted.value} does not match the ladder's event type "
                f"{_event_type_of(ladder_type).value}."
            )
        mask.attrs["event_type"] = wanted.value

    if severity_field:
        severity = classify_scores(scores, ladder)
        mask.attrs["severity_classes"] = ", ".join(
            severity_class.value for severity_class in Severity
        )
        mask.attrs["detected_cells"] = int(np.count_nonzero(mask.values))
    return mask


def detection_summary(
    mask: xr.DataArray, severity: xr.DataArray | None = None
) -> dict[str, int]:
    """Return per-time-step and total counts of extreme cells.

    Args:
        mask: Boolean extreme-cell mask.
        severity: Optional severity field; when given, counts per class are
            added under ``severity_<CLASS>``.

    Returns:
        A dictionary of counts (keys ``total``, one per time coordinate and
        optionally ``severity_<CLASS>``).
    """
    if not isinstance(mask, xr.DataArray):
        raise DetectionError(
            f"detection_summary expects an xarray.DataArray, got {type(mask).__name__}."
        )
    values = np.asarray(mask.values)
    summary: dict[str, int] = {"total": int(np.count_nonzero(values))}

    if TIME_DIM in mask.dims:
        for index, stamp in enumerate(np.asarray(mask[TIME_DIM].values)):
            # datetime_as_string keeps the key stable across datetime units.
            key = str(np.datetime_as_string(stamp, unit="s"))
            summary[key] = int(np.count_nonzero(values[index]))

    if severity is not None:
        severity_values = np.asarray(severity.values)
        for severity_class in Severity:
            summary[f"severity_{severity_class.value}"] = int(
                np.count_nonzero(severity_values == severity_class)
            )
    return summary
