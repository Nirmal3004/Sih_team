"""Freshness tests (SIH26078, Phase 8, section 4).

The required matrix is fresh / stale / expired / unknown. Two properties are
asserted alongside it because they are the point of the feature:

* an unknown timestamp is **not** treated as fresh, and
* ``STALE`` data is not reported as usable-as-current.

Without those, the module would satisfy the letter of the spec while still
letting stale data pass as current.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from extreme_weather.realtime.freshness import (
    DEFAULT_FRESHNESS_THRESHOLDS,
    EXPIRED,
    FRESH,
    STALE,
    UNKNOWN,
    FreshnessThresholds,
    FreshnessTracker,
    assess_freshness,
)

from tests.phase8_support import REFERENCE_NOW

UTC = timezone.utc


@pytest.fixture()
def thresholds() -> FreshnessThresholds:
    """Return the default 3 h / 12 h freshness windows."""
    return DEFAULT_FRESHNESS_THRESHOLDS


# --------------------------------------------------------------------------- #
# The four states
# --------------------------------------------------------------------------- #
def test_recent_data_is_fresh(thresholds: FreshnessThresholds) -> None:
    """Data inside the fresh window is FRESH."""
    result = assess_freshness(REFERENCE_NOW - timedelta(minutes=30), now=REFERENCE_NOW, thresholds=thresholds)

    assert result.status == FRESH
    assert result.is_usable_as_current is True


def test_data_past_the_fresh_window_is_stale(thresholds: FreshnessThresholds) -> None:
    """Data past 3 h but inside 12 h is STALE, and not usable as current."""
    result = assess_freshness(REFERENCE_NOW - timedelta(hours=5), now=REFERENCE_NOW, thresholds=thresholds)

    assert result.status == STALE
    assert result.is_usable_as_current is False
    assert "not current" in result.message


def test_data_past_the_stale_window_is_expired(thresholds: FreshnessThresholds) -> None:
    """Data past 12 h is EXPIRED and must not be presented as current."""
    result = assess_freshness(REFERENCE_NOW - timedelta(hours=20), now=REFERENCE_NOW, thresholds=thresholds)

    assert result.status == EXPIRED
    assert result.is_usable_as_current is False
    assert "must not be presented as current" in result.message


@pytest.mark.parametrize("value", [None, "", "   ", "not-a-timestamp", 12345])
def test_unusable_timestamp_is_unknown(value: object, thresholds: FreshnessThresholds) -> None:
    """Anything unparseable yields UNKNOWN, never an optimistic FRESH."""
    result = assess_freshness(value, now=REFERENCE_NOW, thresholds=thresholds)  # type: ignore[arg-type]

    assert result.status == UNKNOWN
    assert result.is_usable_as_current is False
    assert result.age_minutes is None


def test_future_timestamp_is_unknown_not_fresh(thresholds: FreshnessThresholds) -> None:
    """A future timestamp means a clock or timezone problem, so freshness
    cannot be established. Reporting FRESH would be the worst outcome."""
    result = assess_freshness(REFERENCE_NOW + timedelta(hours=2), now=REFERENCE_NOW, thresholds=thresholds)

    assert result.status == UNKNOWN
    assert "future" in result.message


# --------------------------------------------------------------------------- #
# Boundaries
# --------------------------------------------------------------------------- #
def test_exactly_at_the_fresh_boundary_is_fresh(thresholds: FreshnessThresholds) -> None:
    """The fresh window is inclusive at its edge."""
    result = assess_freshness(
        REFERENCE_NOW - thresholds.fresh_within, now=REFERENCE_NOW, thresholds=thresholds
    )

    assert result.status == FRESH


def test_just_past_the_fresh_boundary_is_stale(thresholds: FreshnessThresholds) -> None:
    """One second past the boundary flips the state."""
    result = assess_freshness(
        REFERENCE_NOW - thresholds.fresh_within - timedelta(seconds=1),
        now=REFERENCE_NOW,
        thresholds=thresholds,
    )

    assert result.status == STALE


def test_exactly_at_the_stale_boundary_is_stale(thresholds: FreshnessThresholds) -> None:
    """The stale window is inclusive at its edge too."""
    result = assess_freshness(
        REFERENCE_NOW - thresholds.stale_within, now=REFERENCE_NOW, thresholds=thresholds
    )

    assert result.status == STALE


# --------------------------------------------------------------------------- #
# Configurability
# --------------------------------------------------------------------------- #
def test_thresholds_are_configurable() -> None:
    """Widening the windows moves a fixed age from STALE to FRESH."""
    strict = FreshnessThresholds(fresh_within=timedelta(hours=1), stale_within=timedelta(hours=3))
    loose = FreshnessThresholds(fresh_within=timedelta(hours=6), stale_within=timedelta(hours=24))
    moment = REFERENCE_NOW - timedelta(hours=5)

    # Same data, opposite verdicts: this is the point of making the windows
    # configurable rather than hard-coding them.
    assert assess_freshness(moment, now=REFERENCE_NOW, thresholds=strict).status == EXPIRED
    assert assess_freshness(moment, now=REFERENCE_NOW, thresholds=loose).status == FRESH


def test_stale_window_must_not_precede_the_fresh_window() -> None:
    """An inverted window would make STALE unreachable, so it is refused."""
    with pytest.raises(ValueError, match="stale_within must be >= fresh_within"):
        FreshnessThresholds(fresh_within=timedelta(hours=6), stale_within=timedelta(hours=1))


def test_negative_fresh_window_is_refused() -> None:
    """A negative window is meaningless."""
    with pytest.raises(ValueError, match="cannot be negative"):
        FreshnessThresholds(fresh_within=timedelta(hours=-1), stale_within=timedelta(hours=1))


def test_thresholds_serialise_in_hours(thresholds: FreshnessThresholds) -> None:
    """The API reports hours, not microseconds."""
    payload = thresholds.as_dict()

    assert payload["fresh_within_hours"] == 3.0
    assert payload["stale_within_hours"] == 12.0


# --------------------------------------------------------------------------- #
# Input handling
# --------------------------------------------------------------------------- #
def test_iso_string_is_accepted(thresholds: FreshnessThresholds) -> None:
    """A provider that reports an ISO string is handled like a datetime."""
    result = assess_freshness(
        (REFERENCE_NOW - timedelta(hours=1)).isoformat(), now=REFERENCE_NOW, thresholds=thresholds
    )

    assert result.status == FRESH


def test_zulu_suffix_is_accepted(thresholds: FreshnessThresholds) -> None:
    """A trailing Z is a common ISO form and must parse."""
    result = assess_freshness(
        (REFERENCE_NOW - timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
        now=REFERENCE_NOW,
        thresholds=thresholds,
    )

    assert result.status == FRESH


def test_naive_datetime_is_treated_as_utc(thresholds: FreshnessThresholds) -> None:
    """A naive timestamp is UTC by this project's convention."""
    naive = (REFERENCE_NOW - timedelta(hours=1)).replace(tzinfo=None)
    result = assess_freshness(naive, now=REFERENCE_NOW, thresholds=thresholds)

    assert result.status == FRESH


def test_assessment_reports_age_in_minutes_and_hours(thresholds: FreshnessThresholds) -> None:
    """Age is reported in both units, because the dashboard needs minutes and a
    human reading needs hours."""
    result = assess_freshness(REFERENCE_NOW - timedelta(hours=2), now=REFERENCE_NOW, thresholds=thresholds)

    assert result.age_minutes == pytest.approx(120.0)
    assert result.age_hours == pytest.approx(2.0)


def test_assessment_serialises(thresholds: FreshnessThresholds) -> None:
    """An assessment is returned over the API, so it must be JSON-ready."""
    payload = assess_freshness(
        REFERENCE_NOW - timedelta(hours=2), now=REFERENCE_NOW, thresholds=thresholds
    ).as_dict()

    assert payload["status"] == FRESH
    assert payload["source_timestamp"] is not None
    assert payload["age_minutes"] == pytest.approx(120.0)


# --------------------------------------------------------------------------- #
# Tracker
# --------------------------------------------------------------------------- #
def test_tracker_remembers_the_last_assessment() -> None:
    """A status endpoint must not have to re-read the data to answer."""
    tracker = FreshnessTracker()

    assert tracker.last is None
    assert tracker.last_status == UNKNOWN

    tracker.assess(REFERENCE_NOW - timedelta(hours=5), now=REFERENCE_NOW)

    assert tracker.last_status == STALE
    assert tracker.last_timestamp is not None


def test_tracker_reset_forgets_everything() -> None:
    """Reset returns the tracker to its pre-ingestion state."""
    tracker = FreshnessTracker()
    tracker.assess(REFERENCE_NOW, now=REFERENCE_NOW)
    tracker.reset()

    assert tracker.last is None
    assert tracker.last_status == UNKNOWN


def test_tracker_serialises(thresholds: FreshnessThresholds) -> None:
    """The tracker reports its thresholds alongside its last verdict."""
    tracker = FreshnessTracker(thresholds=thresholds)
    tracker.assess(REFERENCE_NOW - timedelta(hours=1), now=REFERENCE_NOW)

    payload = tracker.as_dict()

    assert payload["status"] == FRESH
    assert payload["thresholds"]["fresh_within_hours"] == 3.0
