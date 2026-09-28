"""Data freshness (SIH26078, Phase 8, section 4).

The single rule this module exists to enforce: **stale data is never silently
treated as current data.** A dataset from three days ago will happily flow
through anomaly detection, tracking and alerting without complaint, and the
resulting dashboard will look authoritative while describing the past. So
freshness is computed, stored on every run, and surfaced in the API.

States
------
``FRESH``     within the configured window; safe to treat as current
``STALE``     past the fresh window but inside the expiry window; usable, flagged
``EXPIRED``   past the expiry window; must not be presented as current
``UNKNOWN``   the source timestamp could not be determined

The distinction between ``STALE`` and ``EXPIRED`` matters: stale data is a
caveat, expired data is a refusal. Note that ``UNKNOWN`` is *not* treated as
fresh - a missing timestamp is a reason for caution, not for a green badge.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any

from extreme_weather.realtime.models import EXPIRED, FRESH, STALE, UNKNOWN

__all__ = [
    "DEFAULT_FRESHNESS_THRESHOLDS",
    "EXPIRED",
    "FRESH",
    "FreshnessAssessment",
    "FreshnessThresholds",
    "FreshnessTracker",
    "STALE",
    "UNKNOWN",
    "assess_freshness",
]

UTC = timezone.utc


@dataclass(frozen=True)
class FreshnessThresholds:
    """How old data may be before its status changes.

    Attributes:
        fresh_within: Data younger than this is ``FRESH``.
        stale_within: Data younger than this is ``STALE``; beyond it is
            ``EXPIRED``.
    """

    fresh_within: timedelta = timedelta(hours=3)
    stale_within: timedelta = timedelta(hours=12)

    def __post_init__(self) -> None:
        if self.fresh_within < timedelta(0):
            raise ValueError("fresh_within cannot be negative.")
        if self.stale_within < self.fresh_within:
            raise ValueError(
                "stale_within must be >= fresh_within, otherwise the STALE "
                "window would be empty."
            )

    def as_dict(self) -> dict[str, float]:
        """Return the thresholds in hours."""
        return {
            "fresh_within_hours": self.fresh_within.total_seconds() / 3600.0,
            "stale_within_hours": self.stale_within.total_seconds() / 3600.0,
        }


#: The default windows: fresh for 3 hours, stale up to 12, expired beyond.
DEFAULT_FRESHNESS_THRESHOLDS = FreshnessThresholds()


@dataclass(frozen=True)
class FreshnessAssessment:
    """The verdict for one timestamp.

    Attributes:
        status: FRESH, STALE, EXPIRED or UNKNOWN.
        source_timestamp: The data's own timestamp, if known.
        evaluated_at: When the assessment was made.
        age: Elapsed time between the two, or ``None`` when unknown.
        message: Why this status, in words fit for a dashboard tooltip.
    """

    status: str
    source_timestamp: datetime | None = None
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    age: timedelta | None = None
    message: str = ""

    @property
    def age_minutes(self) -> float | None:
        """Age in minutes, or ``None`` when the timestamp is unknown."""
        if self.age is None:
            return None
        return self.age.total_seconds() / 60.0

    @property
    def age_hours(self) -> float | None:
        """Age in hours, or ``None`` when the timestamp is unknown."""
        if self.age is None:
            return None
        return self.age.total_seconds() / 3600.0

    @property
    def is_usable_as_current(self) -> bool:
        """True only for ``FRESH``.

        Deliberately narrow: ``STALE`` data is not current data. A caller that
        wants to process stale data anyway must say so explicitly rather than
        inheriting a permissive default.
        """
        return self.status == FRESH

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable assessment."""
        return {
            "status": self.status,
            "source_timestamp": (
                self.source_timestamp.isoformat() if self.source_timestamp else None
            ),
            "evaluated_at": self.evaluated_at.isoformat(),
            "age_minutes": self.age_minutes,
            "age_hours": self.age_hours,
            "message": self.message,
        }


def _coerce(moment: datetime | str | None) -> datetime | None:
    """Return an aware UTC datetime, or ``None`` if the input is unusable."""
    if moment is None:
        return None
    if isinstance(moment, str):
        text = moment.strip()
        if not text:
            return None
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            moment = datetime.fromisoformat(text)
        except ValueError:
            return None
    if not isinstance(moment, datetime):
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    return moment.astimezone(UTC)


def assess_freshness(
    source_timestamp: datetime | str | None,
    *,
    now: datetime | None = None,
    thresholds: FreshnessThresholds = DEFAULT_FRESHNESS_THRESHOLDS,
) -> FreshnessAssessment:
    """Classify how old a dataset is.

    Args:
        source_timestamp: The newest timestamp the data carries. A naive
            datetime is interpreted as UTC; an unparseable value yields
            ``UNKNOWN`` rather than an optimistic ``FRESH``.
        now: Reference time; defaults to the current UTC time.
        thresholds: The configured freshness windows.

    Returns:
        A :class:`FreshnessAssessment`. Never raises: an unknown timestamp is a
        reportable state, not an error.
    """
    reference = _coerce(now) or datetime.now(UTC)
    parsed = _coerce(source_timestamp)

    if parsed is None:
        return FreshnessAssessment(
            status=UNKNOWN,
            source_timestamp=None,
            evaluated_at=reference,
            age=None,
            message=(
                "The source did not report a usable timestamp, so freshness "
                "cannot be established. Treated as unknown, not as fresh."
            ),
        )

    age = reference - parsed
    if age < timedelta(0):
        return FreshnessAssessment(
            status=UNKNOWN,
            source_timestamp=parsed,
            evaluated_at=reference,
            age=age,
            message=(
                f"The data is timestamped {abs(age).total_seconds() / 3600.0:.1f} h "
                "in the future. This usually means a clock or timezone problem, "
                "so freshness cannot be established."
            ),
        )

    if age <= thresholds.fresh_within:
        status, message = FRESH, f"Data is {_humanise(age)} old."
    elif age <= thresholds.stale_within:
        status, message = (
            STALE,
            f"Data is {_humanise(age)} old, past the "
            f"{thresholds.fresh_within.total_seconds() / 3600.0:g} h freshness "
            "window. Usable, but not current.",
        )
    else:
        status, message = (
            EXPIRED,
            f"Data is {_humanise(age)} old, past the "
            f"{thresholds.stale_within.total_seconds() / 3600.0:g} h expiry "
            "window. It must not be presented as current.",
        )

    return FreshnessAssessment(
        status=status,
        source_timestamp=parsed,
        evaluated_at=reference,
        age=age,
        message=message,
    )


def _humanise(age: timedelta) -> str:
    """Render a duration compactly, e.g. ``12 min`` or ``3.5 h``."""
    seconds = age.total_seconds()
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} min"
    if seconds < 172800:
        return f"{seconds / 3600:.1f} h"
    return f"{seconds / 86400:.1f} d"

class FreshnessTracker:
    """Remembers the last assessment so a status endpoint can answer cheaply.

    Recomputing freshness is trivial; *remembering* it is what lets the API
    report the state of the last ingestion without re-reading data, and what
    lets a run be classified consistently between its stages.
    """

    def __init__(
        self,
        thresholds: FreshnessThresholds = DEFAULT_FRESHNESS_THRESHOLDS,
    ) -> None:
        self._thresholds = thresholds
        self._last: FreshnessAssessment | None = None
        self._last_timestamp: datetime | None = None

    @property
    def thresholds(self) -> FreshnessThresholds:
        """The configured freshness windows."""
        return self._thresholds

    def assess(
        self,
        source_timestamp: datetime | str | None,
        *,
        now: datetime | None = None,
    ) -> FreshnessAssessment:
        """Classify a timestamp and remember the result."""
        self._last = assess_freshness(
            source_timestamp, now=now, thresholds=self._thresholds
        )
        self._last_timestamp = _coerce(source_timestamp)
        return self._last

    @property
    def last(self) -> FreshnessAssessment | None:
        """The most recent assessment, or ``None`` if none has been made."""
        return self._last

    @property
    def last_status(self) -> str:
        """The most recent status, or ``UNKNOWN`` when nothing is known."""
        return self._last.status if self._last else UNKNOWN

    @property
    def last_timestamp(self) -> datetime | None:
        """The most recently assessed source timestamp."""
        return self._last_timestamp

    def reset(self) -> None:
        """Forget the last assessment."""
        self._last = None
        self._last_timestamp = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable view of the tracker."""
        return {
            "thresholds": self._thresholds.as_dict(),
            "last": self._last.as_dict() if self._last else None,
            "status": self.last_status,
        }

