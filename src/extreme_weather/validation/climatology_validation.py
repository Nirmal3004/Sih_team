"""Climatology coverage diagnostics (SIH26078, Phase 7).

The Phase 1 climatology is a *reference distribution* computed from whatever
dataset it is handed. On three days of ERA5 that is a legitimate reference for
those three days, and nothing more. This module measures whether a dataset is
long enough for the grouping it is being asked for, and says so when it is not.

The rule this module enforces
-----------------------------
**A climatology bucket with too few samples is reported as insufficient, not
quietly filled from elsewhere.** A day-of-year climatology built from a week of
data and silently filled from a pooled monthly mean would produce smooth,
confident, wrong numbers - and the confidence is the danger. So:

* ``CoverageReport.usable`` is False when any requested bucket is under
  ``min_samples_per_bucket``;
* ``CoverageReport.message`` names the shortfall explicitly, for example
  ``"Insufficient historical coverage for day-of-year climatology."``;
* the month and day-of-year variants are assessed separately, because a dataset
  can be perfectly adequate monthly and hopeless day-of-year.

A real climatology needs **decades**. This project is honest about the gap
between what it has and what that would require.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Sequence

import numpy as np
import xarray as xr

from extreme_weather.anomaly.climatology import ClimatologyGrouping
from extreme_weather.ingestion.base import utc_now

__all__ = [
    "INSUFFICIENT_DAY_OF_YEAR",
    "INSUFFICIENT_MONTH",
    "MIN_DAYS_FOR_DAY_OF_YEAR",
    "MIN_MONTHS_FOR_MONTHLY",
    "MIN_SAMPLES_PER_BUCKET",
    "ClimatologyCoverage",
    "Diagnostics",
    "assess_climatology_coverage",
    "build_climatology_diagnostics",
]


#: The exact sentence required when day-of-year coverage is insufficient.
INSUFFICIENT_DAY_OF_YEAR = (
    "Insufficient historical coverage for day-of-year climatology."
)

#: The equivalent message for the monthly grouping.
INSUFFICIENT_MONTH = "Insufficient historical coverage for monthly climatology."

#: Samples a bucket needs before its statistics are trusted.
MIN_SAMPLES_PER_BUCKET: int = 30

#: Distinct days a dataset must cover for day-of-year statistics to mean
#: anything. 365 distinct days is the natural minimum; below it, most of the
#: calendar is unobserved.
MIN_DAYS_FOR_DAY_OF_YEAR: int = 365

#: Distinct months a dataset must cover for monthly statistics to be complete.
MIN_MONTHS_FOR_MONTHLY: int = 12


@dataclass(frozen=True)
class ClimatologyCoverage:
    """How much history a dataset actually has, per grouping.

    Attributes:
        grouping: The grouping this assessment is for.
        distinct_days: Number of distinct calendar days covered.
        distinct_months: Number of distinct calendar months covered.
        first: Earliest timestamp, as ISO 8601.
        last: Latest timestamp, as ISO 8601.
        buckets_expected: How many buckets the grouping defines (12 or 366).
        buckets_covered: How many of them have at least one sample.
        buckets_sufficient: How many have at least ``min_samples_per_bucket``.
        min_samples_per_bucket: The threshold applied.
        span_days: Number of days between ``first`` and ``last``.
        missing_fraction: Fraction of the span with no observation at all.
    """

    grouping: str
    distinct_days: int
    distinct_months: int
    first: str | None
    last: str | None
    buckets_expected: int
    buckets_covered: int
    buckets_sufficient: int
    min_samples_per_bucket: int = MIN_SAMPLES_PER_BUCKET
    span_days: int = 0
    missing_fraction: float = 0.0

    @property
    def usable(self) -> bool:
        """True when every expected bucket has enough samples.

        Deliberately strict. A partially-covered climatology is not a
        climatology, and a report that said otherwise would be the single most
        misleading thing this project could do.
        """
        return (
            self.buckets_expected > 0
            and self.buckets_sufficient == self.buckets_expected
        )

    @property
    def message(self) -> str:
        """A human-readable verdict, ``""`` when the grouping is usable."""
        if self.usable:
            return ""
        if self.grouping == ClimatologyGrouping.DAY_OF_YEAR.value:
            return INSUFFICIENT_DAY_OF_YEAR
        if self.grouping == ClimatologyGrouping.MONTH.value:
            return INSUFFICIENT_MONTH
        return (
            f"Insufficient historical coverage for {self.grouping} climatology: "
            f"{self.buckets_sufficient}/{self.buckets_expected} buckets have at "
            f"least {self.min_samples_per_bucket} samples."
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "grouping": self.grouping,
            "usable": self.usable,
            "message": self.message,
            "distinct_days": self.distinct_days,
            "distinct_months": self.distinct_months,
            "first": self.first,
            "last": self.last,
            "span_days": self.span_days,
            "buckets_expected": self.buckets_expected,
            "buckets_covered": self.buckets_covered,
            "buckets_sufficient": self.buckets_sufficient,
            "min_samples_per_bucket": self.min_samples_per_bucket,
            "missing_fraction": round(self.missing_fraction, 6),
        }


@dataclass(frozen=True)
class Diagnostics:
    """The full climatology picture for one dataset and split.

    Attributes:
        split: Which split was measured (``"train"``, ``"validation"`` ...).
        dataset_id: Identifier of the dataset the statistics describe.
        period_start: Earliest timestamp, ISO 8601.
        period_end: Latest timestamp, ISO 8601.
        time_steps: Number of time steps.
        spatial_points: Latitude points x longitude points.
        variables: Variables inspected.
        missing_fraction: Fraction of all data cells that are NaN.
        per_variable_missing: NaN fraction per variable.
        coverages: Coverage assessment per grouping.
        hour_of_day_supported: Whether an hour-of-day climatology is defensible.
        hour_of_day_message: The reason when it is not.
        generated_at: When the diagnostics were computed.
    """

    split: str
    dataset_id: str
    period_start: str | None
    period_end: str | None
    time_steps: int
    spatial_points: int
    variables: tuple[str, ...]
    missing_fraction: float
    per_variable_missing: dict[str, float]
    coverages: dict[str, ClimatologyCoverage] = field(default_factory=dict)
    hour_of_day_supported: bool = False
    hour_of_day_message: str = ""
    generated_at: datetime | None = None

    def coverage(self, grouping: str) -> ClimatologyCoverage | None:
        """Return the coverage for one grouping, or ``None`` when not assessed."""
        return self.coverages.get(grouping)

    @property
    def is_sufficient(self) -> bool:
        """True when every assessed grouping is usable.

        Almost always False for a research dataset - which is the point. The
        report carries the reason rather than silently degrading.
        """
        return all(coverage.usable for coverage in self.coverages.values())

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "split": self.split,
            "dataset_id": self.dataset_id,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "time_steps": self.time_steps,
            "spatial_points": self.spatial_points,
            "variables": list(self.variables),
            "missing_fraction": round(self.missing_fraction, 6),
            "per_variable_missing": {
                key: round(value, 6) for key, value in self.per_variable_missing.items()
            },
            "is_sufficient": self.is_sufficient,
            "coverages": {
                name: coverage.as_dict() for name, coverage in self.coverages.items()
            },
            "hour_of_day_supported": self.hour_of_day_supported,
            "hour_of_day_message": self.hour_of_day_message,
            "generated_at": self.generated_at.isoformat() if self.generated_at else None,
        }


def _iso(value: np.datetime64) -> str:
    """Format a ``datetime64`` scalar as a compact ISO 8601 string."""
    return str(np.datetime_as_string(np.asarray(value), unit="s"))


def _bucket_of(times: np.ndarray, grouping: str) -> np.ndarray:
    """Return the climatology bucket key of every timestamp."""
    months = times.astype("datetime64[M]").astype(int) % 12 + 1
    if grouping == ClimatologyGrouping.MONTH.value:
        return months
    if grouping == ClimatologyGrouping.DAY_OF_YEAR.value:
        return (
            (times.astype("datetime64[D]") - times.astype("datetime64[Y]")).astype(int)
            + 1
        )
    if grouping == "hour_of_day":
        return times.astype("datetime64[h]").astype(int) % 24
    raise ValueError(f"Unknown grouping '{grouping}'.")


def assess_climatology_coverage(
    times: Sequence[np.datetime64] | np.ndarray,
    grouping: str,
    *,
    min_samples_per_bucket: int = MIN_SAMPLES_PER_BUCKET,
) -> ClimatologyCoverage:
    """Measure how much history a time axis provides for one grouping.

    Args:
        times: The timestamps to assess.
        grouping: ``"month"``, ``"day_of_year"`` or ``"hour_of_day"``.
        min_samples_per_bucket: Samples a bucket needs to be trusted.

    Returns:
        A :class:`ClimatologyCoverage`. ``usable`` is False whenever any bucket
        is short, and ``message`` then explains the shortfall in words.

    Raises:
        ValueError: If the grouping is unknown or the axis is empty.
    """
    values = np.asarray(times)
    if values.size == 0:
        raise ValueError("Cannot assess coverage of an empty time axis.")
    if not np.issubdtype(values.dtype, np.datetime64):
        raise ValueError(f"time must be datetime64; got {values.dtype}.")

    ordered = np.sort(values)
    days = np.unique(ordered.astype("datetime64[D]"))
    months = np.unique(ordered.astype("datetime64[M]"))

    if grouping == ClimatologyGrouping.MONTH.value:
        expected = 12
    elif grouping == ClimatologyGrouping.DAY_OF_YEAR.value:
        expected = 366
    elif grouping == "hour_of_day":
        expected = 24
    else:
        raise ValueError(f"Unknown grouping '{grouping}'.")

    keys = _bucket_of(ordered, grouping)
    counts = np.bincount(keys, minlength=expected + 1)[1:]
    covered = int(np.count_nonzero(counts))
    sufficient = int(np.count_nonzero(counts >= int(min_samples_per_bucket)))

    span_days = int((ordered[-1].astype("datetime64[D]") - ordered[0].astype("datetime64[D]")).astype(int)) + 1
    return ClimatologyCoverage(
        grouping=grouping,
        distinct_days=int(days.size),
        distinct_months=int(months.size),
        first=_iso(ordered[0]),
        last=_iso(ordered[-1]),
        buckets_expected=expected,
        buckets_covered=covered,
        buckets_sufficient=sufficient,
        min_samples_per_bucket=int(min_samples_per_bucket),
        span_days=span_days,
        missing_fraction=round(1.0 - (covered / expected), 6),
    )



def build_climatology_diagnostics(
    dataset: xr.Dataset,
    *,
    split: str = "train",
    dataset_id: str = "unknown",
    min_samples_per_bucket: int = MIN_SAMPLES_PER_BUCKET,
    hour_of_day_min_days: int = 30,
    generated_at: datetime | None = None,
) -> Diagnostics:
    """Describe a dataset's climatology coverage and data completeness.

    The climatology is **not** built here - Phase 1 owns that. This measures
    whether the dataset is long enough for the groupings it will be asked for,
    and records the missing-data statistics a reader needs to interpret any
    later number.

    Args:
        dataset: The dataset to describe.
        split: Which split this is, recorded in the result.
        dataset_id: Identifier recorded in the result.
        min_samples_per_bucket: Samples a bucket needs to be trusted.
        hour_of_day_min_days: Distinct days needed before an hour-of-day
            climatology is defensible. A diurnal cycle needs many days; two
            hours from one afternoon is not a climatology.
        generated_at: Timestamp to stamp on the result.

    Returns:
        A :class:`Diagnostics` with one coverage assessment per grouping.

    Raises:
        ValueError: If the dataset has no 1-D ``time`` axis.
    """
    if "time" not in dataset.coords or dataset.coords["time"].ndim != 1:
        raise ValueError("Dataset must carry a 1-D 'time' coordinate.")

    times = np.asarray(dataset.coords["time"].values)
    variables = tuple(sorted(str(name) for name in dataset.data_vars))

    per_variable: dict[str, float] = {}
    total_cells = 0
    nan_cells = 0
    for name in variables:
        variable = dataset[name]
        if variable.ndim == 0 or variable.dtype.kind not in "fc":
            continue
        values = np.asarray(variable.values, dtype="float64")
        total_cells += values.size
        nan_cells += int(np.isnan(values).sum())
        per_variable[name] = float(np.isnan(values).sum() / values.size) if values.size else 0.0
    missing_fraction = (nan_cells / total_cells) if total_cells else 0.0

    latitude_points = int(dataset.sizes.get("latitude", 0))
    longitude_points = int(dataset.sizes.get("longitude", 0))

    coverages = {
        grouping: assess_climatology_coverage(
            times, grouping, min_samples_per_bucket=min_samples_per_bucket
        )
        for grouping in (
            ClimatologyGrouping.MONTH.value,
            ClimatologyGrouping.DAY_OF_YEAR.value,
        )
    }

    hour = assess_climatology_coverage(
        times, "hour_of_day", min_samples_per_bucket=min_samples_per_bucket
    )
    # Hour-of-day is a finer question than month or day-of-year: the grouping
    # can be *populated* by three days and still be meaningless, because a
    # diurnal cycle is only a climatology once many days agree about it.
    hour_supported = hour.distinct_days >= int(hour_of_day_min_days) and hour.usable
    if hour_supported:
        hour_message = ""
    elif hour.distinct_days < int(hour_of_day_min_days):
        hour_message = (
            f"Insufficient historical coverage for hour-of-day climatology: "
            f"{hour.distinct_days} distinct day(s), "
            f"{hour_of_day_min_days} required."
        )
    else:
        hour_message = (
            "Insufficient historical coverage for hour-of-day climatology: "
            f"{hour.buckets_sufficient}/{hour.buckets_expected} hours have at "
            f"least {min_samples_per_bucket} samples."
        )

    return Diagnostics(
        split=split,
        dataset_id=dataset_id,
        period_start=_iso(np.min(times)) if times.size else None,
        period_end=_iso(np.max(times)) if times.size else None,
        time_steps=int(times.size),
        spatial_points=latitude_points * longitude_points,
        variables=variables,
        missing_fraction=float(missing_fraction),
        per_variable_missing=per_variable,
        coverages=coverages,
        hour_of_day_supported=hour_supported,
        hour_of_day_message=hour_message,
        generated_at=generated_at or utc_now(),
    )

