"""Data quality gates (SIH26078, Phase 8, section 6).

Everything the pipeline ingests is checked *before* anomaly detection sees it.
The reason is that the downstream stages are optimising code: a tracking
algorithm asked to link events through a dataset with scrambled timestamps, or
an anomaly detector handed a field that is 90% NaN, will not raise - it will
happily return a confident, meaningless result.

Three verdicts
--------------
``VALID``                 every check passed
``VALID_WITH_WARNINGS``   usable, but something is off and is recorded
``REJECTED``              a blocking check failed; the data must not be processed

Blocking vs. advisory
---------------------
Checks are marked ``blocking`` or ``advisory``. Missing coordinates or a
non-monotonic time axis are blocking: the data is structurally unusable.
A slightly elevated missing-value fraction is advisory, provided it stays under
a threshold at which the anomaly statistics become meaningless. That split is
a judgement call, and it is why the threshold is a named constant rather than a
literal buried in a loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

import numpy as np
import xarray as xr

from extreme_weather.realtime.models import utc_now

__all__ = [
    "DEFAULT_MAX_MISSING_FRACTION",
    "DEFAULT_MIN_FINITE_FRACTION",
    "QualityCheck",
    "QualityGate",
    "QualityVerdict",
]

#: Advisory. Below this missing-value fraction the remaining data is still
#: usable; above it the anomaly statistics are dominated by gaps.
DEFAULT_MAX_MISSING_FRACTION = 0.5

#: Advisory. The minimum share of finite values a field must have.
DEFAULT_MIN_FINITE_FRACTION = 0.5

#: Advisory. The maximum share of time steps allowed to be entirely missing.
DEFAULT_MAX_EMPTY_STEPS_FRACTION = 0.2

VALID = "VALID"
VALID_WITH_WARNINGS = "VALID_WITH_WARNINGS"
REJECTED = "REJECTED"


@dataclass(frozen=True)
class QualityCheck:
    """One performed check.

    Attributes:
        name: Check identifier.
        passed: Whether the check succeeded.
        blocking: Whether a failure should reject the dataset.
        message: What was found, in words.
        detail: Supporting numbers.
    """

    name: str
    passed: bool
    blocking: bool
    message: str = ""
    detail: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the check."""
        return {
            "name": self.name,
            "passed": self.passed,
            "blocking": self.blocking,
            "message": self.message,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class QualityVerdict:
    """The combined outcome of all checks.

    Attributes:
        status: VALID, VALID_WITH_WARNINGS or REJECTED.
        checks: Every check performed, in order.
    """

    status: str
    checks: tuple[QualityCheck, ...] = ()

    @property
    def passed(self) -> bool:
        """True unless the dataset was rejected."""
        return self.status != REJECTED

    @property
    def blocking_failures(self) -> tuple[QualityCheck, ...]:
        """The blocking checks that failed."""
        return tuple(
            c for c in self.checks if c.blocking and not c.passed
        )

    @property
    def warnings(self) -> tuple[QualityCheck, ...]:
        """Every non-blocking check that failed."""
        return tuple(c for c in self.checks if not c.blocking and not c.passed)

    @property
    def reason(self) -> str:
        """A one-line summary of why the dataset was rejected."""
        failures = self.blocking_failures
        if not failures:
            return ""
        return "; ".join(f"{c.name}: {c.message}" for c in failures)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the verdict."""
        return {
            "status": self.status,
            "passed": self.passed,
            "reason": self.reason,
            "checks": [c.as_dict() for c in self.checks],
            "blocking_failures": [c.name for c in self.blocking_failures],
            "warnings": [c.name for c in self.warnings],
        }

    def summary(self) -> str:
        """Render a short human-readable summary."""
        if self.status == REJECTED:
            return f"REJECTED - {self.reason}"
        if self.status == VALID_WITH_WARNINGS:
            names = ", ".join(c.name for c in self.warnings) or "none"
            return f"VALID_WITH_WARNINGS - advisory checks failed: {names}"
        return f"VALID - {len(self.checks)} checks passed."


class QualityGate:
    """Validates a dataset's structure before it enters the pipeline.

    Args:
        expected_variables: Variables that must be present. Empty means "any
            variable is acceptable", which is the right default for a prototype
            that does not yet know which fields a given provider will supply.
        min_finite_fraction: Advisory floor on the share of finite values.
        max_missing_fraction: Advisory ceiling on the share of missing values.
    """

    def __init__(
        self,
        expected_variables: Sequence[str] = (),
        *,
        min_finite_fraction: float = DEFAULT_MIN_FINITE_FRACTION,
        max_missing_fraction: float = DEFAULT_MAX_MISSING_FRACTION,
    ) -> None:
        self.expected_variables = tuple(expected_variables)
        self.min_finite_fraction = float(min_finite_fraction)
        self.max_missing_fraction = float(max_missing_fraction)

    def evaluate(self, dataset: xr.Dataset) -> QualityVerdict:
        """Run every check and return the verdict.

        Returns:
            A :class:`QualityVerdict`. Structural problems are recorded as
            failed checks rather than raised, so a caller gets the complete
            picture in one call.
        """
        checks: list[QualityCheck] = [
            self._check_dimensions(dataset),
            self._check_coordinates(dataset),
            self._check_time_coordinate(dataset),
            self._check_monotonic_time(dataset),
            self._check_duplicate_times(dataset),
            self._check_temporal_coverage(dataset),
            self._check_variables(dataset),
            self._check_units(dataset),
            self._check_finite_values(dataset),
        ]
        blocking_failed = any(not c.passed and c.blocking for c in checks)
        if blocking_failed:
            status = REJECTED
        elif any(not c.passed for c in checks):
            status = VALID_WITH_WARNINGS
        else:
            status = VALID
        return QualityVerdict(status=status, checks=tuple(checks))

    def _check_dimensions(self, dataset: xr.Dataset) -> QualityCheck:
        """Check that the spatial grid is present and non-degenerate."""
        for dim in ("latitude", "longitude"):
            if dim not in dataset.dims:
                return QualityCheck(
                    name="spatial_dimensions",
                    passed=False,
                    blocking=True,
                    message=f"The dataset has no '{dim}' dimension.",
                    detail={"dims": list(dataset.dims)},
                )
            size = int(dataset.sizes[dim])
            if size == 0:
                return QualityCheck(
                    name="spatial_dimensions",
                    passed=False,
                    blocking=True,
                    message=f"'{dim}' has zero points.",
                    detail={"dims": {d: int(s) for d, s in dataset.sizes.items()}},
                )
        return QualityCheck(
            name="spatial_dimensions",
            passed=True,
            blocking=True,
            message="Spatial dimensions are present and non-empty.",
            detail={"dims": {d: int(s) for d, s in dataset.sizes.items()}},
        )

    def _check_coordinates(self, dataset: xr.Dataset) -> QualityCheck:
        """Check that latitude and longitude actually exist as coordinates."""
        missing = [c for c in ("latitude", "longitude") if c not in dataset.coords]
        if missing:
            return QualityCheck(
                name="coordinates_present",
                passed=False,
                blocking=True,
                message=f"Missing coordinate(s): {', '.join(missing)}.",
                detail={"missing": missing},
            )
        unpopulated = [
            name
            for name in ("latitude", "longitude")
            if int(np.asarray(dataset.coords[name].values).size) == 0
        ]
        if unpopulated:
            return QualityCheck(
                name="coordinates_present",
                passed=False,
                blocking=True,
                message=f"Empty coordinate(s): {', '.join(unpopulated)}.",
                detail={"empty": unpopulated},
            )
        return QualityCheck(
            name="coordinates_present",
            passed=True,
            blocking=True,
            message="Latitude and longitude coordinates are present.",
        )

    def _check_time_coordinate(self, dataset: xr.Dataset) -> QualityCheck:
        """Check that a time coordinate exists."""
        if "time" not in dataset.coords:
            return QualityCheck(
                name="time_present",
                passed=False,
                blocking=True,
                message="The dataset has no 'time' coordinate.",
            )
        size = int(dataset.sizes.get("time", 0))
        if size == 0:
            return QualityCheck(
                name="time_present",
                passed=False,
                blocking=True,
                message="The time coordinate is empty.",
            )
        return QualityCheck(
            name="time_present",
            passed=True,
            blocking=True,
            message=f"Time coordinate present with {size} steps.",
            detail={"time_steps": size},
        )

    def _check_monotonic_time(self, dataset: xr.Dataset) -> QualityCheck:
        """Check that time never runs backwards.

        Monotonic here means *non-decreasing*: an equal pair of timestamps is
        not disorder, it is duplication, and duplication is reported separately
        by :meth:`_check_duplicate_times` with a different severity. Only a real
        decrease is treated as the blocking condition, because that is what
        breaks tracking and any lag-based model.
        """
        if "time" not in dataset.coords:
            return QualityCheck(
                name="time_monotonic",
                passed=False,
                blocking=True,
                message="Cannot verify monotonic time: no 'time' coordinate.",
            )
        times = np.asarray(dataset.coords["time"].values)
        if times.size < 2:
            return QualityCheck(
                name="time_monotonic",
                passed=True,
                blocking=True,
                message="A single time step is trivially monotonic.",
                detail={"time_steps": int(times.size)},
            )
        diffs = np.diff(times.astype("datetime64[ns]").astype("int64"))
        inversions = int(np.count_nonzero(diffs < 0))
        if inversions:
            return QualityCheck(
                name="time_monotonic",
                passed=False,
                blocking=True,
                message=(
                    f"Time is not strictly increasing at {inversions} position(s). "
                    "Tracking and lag-based models require ordered time."
                ),
                detail={"inversions": inversions},
            )
        return QualityCheck(
            name="time_monotonic",
            passed=True,
            blocking=True,
            message="Time increases strictly.",
        )

    def _check_duplicate_times(self, dataset: xr.Dataset) -> QualityCheck:
        """Check for unexpected duplicate timestamps."""
        if "time" not in dataset.coords:
            return QualityCheck(
                name="no_duplicate_times",
                passed=False,
                blocking=True,
                message="Cannot check duplicates: no 'time' coordinate.",
            )
        times = np.asarray(dataset.coords["time"].values)
        if times.size < 2:
            return QualityCheck(
                name="no_duplicate_times",
                passed=True,
                blocking=True,
                message="No duplicates in a single time step.",
            )
        unique, counts = np.unique(
            times.astype("datetime64[ns]").astype("int64"), return_counts=True
        )
        duplicates = int(np.count_nonzero(counts > 1))
        if duplicates:
            repeated = [
                str(np.datetime_as_string(np.datetime64(int(v), "ns"), unit="s"))
                for v, c in zip(unique, counts)
                if c > 1
            ]
            return QualityCheck(
                name="no_duplicate_times",
                passed=False,
                blocking=False,
                message=(
                    f"{duplicates} timestamp(s) appear more than once; repeated "
                    "steps can double-count an event in tracking."
                ),
                detail={"duplicates": duplicates, "examples": repeated[:5]},
            )
        return QualityCheck(
            name="no_duplicate_times",
            passed=True,
            blocking=False,
            message="No duplicate timestamps.",
        )

    def _check_temporal_coverage(self, dataset: xr.Dataset) -> QualityCheck:
        """Check that the dataset actually covers a time span."""
        if "time" not in dataset.coords:
            return QualityCheck(
                name="temporal_coverage",
                passed=False,
                blocking=True,
                message="No 'time' coordinate, so coverage is empty.",
            )
        times = np.asarray(dataset.coords["time"].values)
        if times.size == 0:
            return QualityCheck(
                name="temporal_coverage",
                passed=False,
                blocking=True,
                message="Temporal coverage is empty.",
            )
        return QualityCheck(
            name="temporal_coverage",
            passed=True,
            blocking=True,
            message=f"Coverage spans {int(times.size)} time step(s).",
            detail={
                "time_steps": int(times.size),
                "start": str(np.datetime_as_string(times[0], unit="s")),
                "end": str(np.datetime_as_string(times[-1], unit="s")),
            },
        )

    def _check_variables(self, dataset: xr.Dataset) -> QualityCheck:
        """Check that the expected variables are present."""
        present = {str(n) for n in dataset.data_vars}
        if not present:
            return QualityCheck(
                name="variables_present",
                passed=False,
                blocking=True,
                message="The dataset contains no data variables.",
                detail={"present": []},
            )
        missing = [v for v in self.expected_variables if v not in present]
        if missing:
            return QualityCheck(
                name="variables_present",
                passed=False,
                blocking=True,
                message=(
                    f"Expected variable(s) absent: {', '.join(sorted(missing))}."
                ),
                detail={"missing": sorted(missing), "present": sorted(present)},
            )
        return QualityCheck(
            name="variables_present",
            passed=True,
            blocking=True,
            message=f"{len(present)} variable(s) present.",
            detail={"present": sorted(present)},
        )

    def _check_units(self, dataset: xr.Dataset) -> QualityCheck:
        """Check that every variable declares a plausible unit.

        A field with no ``units`` attribute is *advisory*: plenty of valid
        datasets omit it, and Phase 6's normaliser can often infer. A field with
        a unit that is not a string, or an empty one, is a genuine problem.
        """
        # A missing sentinel distinguishes "no units attribute" from
        # "units attribute present but None", which are different defects.
        absent = object()
        problems: list[str] = []
        for name, variable in dataset.data_vars.items():
            units = variable.attrs.get("units", absent)
            if units is absent:
                continue
            if not isinstance(units, str) or not str(units).strip():
                problems.append(f"{name} has an invalid unit attribute ({units!r})")
        if problems:
            return QualityCheck(
                name="units_valid",
                passed=False,
                blocking=False,
                message="; ".join(problems),
                detail={"variables": problems},
            )
        missing = [
            str(n) for n, v in dataset.data_vars.items() if "units" not in v.attrs
        ]
        if missing:
            return QualityCheck(
                name="units_valid",
                passed=False,
                blocking=False,
                message=(
                    f"{len(missing)} variable(s) declare no unit; unit conversion "
                    "may be skipped downstream."
                ),
                detail={"variables": sorted(missing)[:10]},
            )
        return QualityCheck(
            name="units_valid",
            passed=True,
            blocking=False,
            message="All variables declare a unit.",
        )

    def _check_finite_values(self, dataset: xr.Dataset) -> QualityCheck:
        """Check the share of finite values across the dataset.

        Advisory rather than blocking: a partly-missing field is often a genuine
        observation of reality. The thresholds mark the point at which the
        remaining values are too few to compute trustworthy statistics.
        """
        total = 0
        finite = 0
        per_variable: dict[str, float] = {}
        for name, variable in dataset.data_vars.items():
            values = np.asarray(variable.values, dtype="float64").ravel()
            if values.size == 0:
                continue
            good = int(np.count_nonzero(np.isfinite(values)))
            total += values.size
            finite += good
            per_variable[str(name)] = good / values.size

        if total == 0:
            return QualityCheck(
                name="finite_values",
                passed=False,
                blocking=True,
                message="The dataset has no values to assess.",
            )

        fraction = finite / total
        empty_steps = self._count_empty_steps(dataset)
        empty_fraction = empty_steps / max(1, int(dataset.sizes.get("time", 1)))

        passed = (
            fraction >= self.min_finite_fraction
            and empty_fraction <= DEFAULT_MAX_EMPTY_STEPS_FRACTION
        )
        if passed:
            message = f"{fraction * 100:.1f}% of values are finite."
        else:
            message = (
                f"Only {fraction * 100:.1f}% of values are finite, and "
                f"{empty_fraction * 100:.1f}% of time steps are entirely missing."
            )
        return QualityCheck(
            name="finite_values",
            passed=passed,
            blocking=False,
            message=message,
            detail={
                "finite_fraction": fraction,
                "missing_fraction": 1.0 - fraction,
                "empty_step_fraction": empty_fraction,
                "per_variable_finite_fraction": per_variable,
            },
        )

    @staticmethod
    def _count_empty_steps(dataset: xr.Dataset) -> int:
        """Return the number of time steps where every variable is all-NaN."""
        if "time" not in dataset.dims or not dataset.data_vars:
            return 0
        empty = np.zeros(int(dataset.sizes["time"]), dtype=bool)
        for variable in dataset.data_vars.values():
            values = np.asarray(variable.values, dtype="float64")
            if values.ndim < 1 or values.shape[0] != empty.size:
                continue
            reshaped = values.reshape(empty.size, -1)
            all_nan = ~np.isfinite(reshaped).any(axis=1)
            empty |= all_nan
        return int(np.count_nonzero(empty))

