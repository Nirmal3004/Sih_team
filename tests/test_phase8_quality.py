"""Quality-gate tests (SIH26078, Phase 8, section 6).

The required matrix: missing coordinates, duplicate timestamps, missing
variables, NaN-heavy data and invalid temporal coverage.

The property that matters most is asserted explicitly at the end: a rejected
dataset must never reach anomaly detection. The gate is only worth having if
something downstream actually stops.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pytest
import xarray as xr

from extreme_weather.realtime.quality_gate import (
    DEFAULT_MAX_MISSING_FRACTION,
    QualityGate,
    QualityVerdict,
)

from tests.phase8_support import (
    REFERENCE_NOW,
    make_dataset,
    make_dataset_without_time,
    make_duplicate_time_dataset,
    make_malformed_dataset,
    make_nan_heavy_dataset,
    make_non_monotonic_dataset,
)


def _check(verdict: QualityVerdict, name: str):
    """Return the named check, failing the test if it is absent."""
    for item in verdict.checks:
        if item.name == name:
            return item
    raise AssertionError(f"check '{name}' was not performed; ran {[c.name for c in verdict.checks]}")


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
def test_valid_dataset_passes_every_check() -> None:
    """A conforming dataset is VALID, with no failures at all."""
    verdict = QualityGate().evaluate(make_dataset())

    assert verdict.status == "VALID"
    assert verdict.passed is True
    assert all(c.passed for c in verdict.checks)


def test_every_required_check_is_performed() -> None:
    """The gate covers the checks the specification requires."""
    verdict = QualityGate().evaluate(make_dataset())
    names = {c.name for c in verdict.checks}

    assert {
        "coordinates_present",
        "time_present",
        "time_monotonic",
        "no_duplicate_times",
        "variables_present",
        "units_valid",
        "finite_values",
        "temporal_coverage",
        "spatial_dimensions",
    }.issubset(names)


# --------------------------------------------------------------------------- #
# Missing coordinates
# --------------------------------------------------------------------------- #
def test_missing_spatial_dimensions_are_rejected() -> None:
    """Data with no latitude/longitude is structurally unusable."""
    verdict = QualityGate().evaluate(make_malformed_dataset())

    assert verdict.status == "REJECTED"
    assert _check(verdict, "spatial_dimensions").blocking is True


def test_missing_coordinate_is_rejected() -> None:
    """A dimension present but the coordinate removed is also a rejection."""
    dataset = make_dataset().drop_vars("latitude")
    verdict = QualityGate().evaluate(dataset)

    assert verdict.status == "REJECTED"
    assert _check(verdict, "coordinates_present").passed is False


def test_zero_length_time_axis_is_rejected() -> None:
    """A dataset with zero steps covers nothing and cannot be processed."""
    dataset = make_dataset().isel(time=slice(0, 0))
    verdict = QualityGate().evaluate(dataset)

    assert verdict.status == "REJECTED"
    assert _check(verdict, "time_present").passed is False


# --------------------------------------------------------------------------- #
# Time
# --------------------------------------------------------------------------- #
def test_missing_time_coordinate_is_rejected() -> None:
    """Without a time axis there is no event, so there is nothing to track."""
    verdict = QualityGate().evaluate(make_dataset_without_time())

    assert verdict.status == "REJECTED"
    assert _check(verdict, "time_present").passed is False


def test_non_monotonic_time_is_rejected() -> None:
    """Tracking and lag-based models require ordered time."""
    verdict = QualityGate().evaluate(make_non_monotonic_dataset())

    assert verdict.status == "REJECTED"
    monotonic = _check(verdict, "time_monotonic")
    assert monotonic.passed is False
    assert monotonic.blocking is True


def test_duplicate_timestamps_are_a_warning_not_a_rejection() -> None:
    """A repeated step can double-count an event, but the data is still usable."""
    verdict = QualityGate().evaluate(make_duplicate_time_dataset())

    duplicates = _check(verdict, "no_duplicate_times")
    assert duplicates.passed is False
    assert duplicates.blocking is False
    assert verdict.status == "VALID_WITH_WARNINGS"
    assert verdict.passed is True


def test_duplicate_timestamps_are_counted() -> None:
    """The check reports how many repetitions were found."""
    verdict = QualityGate().evaluate(make_duplicate_time_dataset())

    assert _check(verdict, "no_duplicate_times").detail["duplicates"] >= 1


def test_duplicate_timestamps_do_not_count_as_disorder() -> None:
    """A repeated step is duplication, not inversion, so the blocking
    monotonic check must pass and only the advisory check may fail."""
    verdict = QualityGate().evaluate(make_duplicate_time_dataset())

    assert _check(verdict, "time_monotonic").passed is True
    assert _check(verdict, "no_duplicate_times").passed is False


def test_single_timestep_is_trivially_monotonic() -> None:
    """One point cannot be out of order, and must not be rejected for it."""
    verdict = QualityGate().evaluate(make_dataset(hours=1))

    assert verdict.passed is True
    assert _check(verdict, "time_monotonic").passed is True


def test_empty_temporal_coverage_is_rejected() -> None:
    """A dataset with zero time steps covers nothing."""
    dataset = make_dataset().isel(time=slice(0, 0))
    verdict = QualityGate().evaluate(dataset)

    assert verdict.status == "REJECTED"


# --------------------------------------------------------------------------- #
# Variables
# --------------------------------------------------------------------------- #
def test_missing_expected_variable_is_rejected() -> None:
    """A required field that is absent makes downstream analysis wrong."""
    gate = QualityGate(expected_variables=("precipitation", "surface_pressure"))
    verdict = gate.evaluate(make_dataset())

    assert verdict.status == "REJECTED"
    variables = _check(verdict, "variables_present")
    assert "surface_pressure" in variables.detail["missing"]


def test_present_expected_variable_passes() -> None:
    """The same gate passes when the variable is present."""
    gate = QualityGate(expected_variables=("precipitation",))
    verdict = gate.evaluate(make_dataset())

    assert verdict.passed is True


def test_dataset_with_no_variables_is_rejected() -> None:
    """An empty dataset is not weather."""
    dataset = make_dataset().drop_vars(list(make_dataset().data_vars))
    verdict = QualityGate().evaluate(dataset)

    assert verdict.status == "REJECTED"


def test_no_expected_variables_means_any_variable_is_acceptable() -> None:
    """The default is permissive, for providers whose fields are not yet fixed."""
    verdict = QualityGate().evaluate(make_dataset(variables=("snowfall",)))

    assert verdict.passed is True


# --------------------------------------------------------------------------- #
# Units and missing values
# --------------------------------------------------------------------------- #
def test_invalid_unit_attribute_is_reported() -> None:
    """A unit that is not a usable string is flagged."""
    dataset = make_dataset()
    dataset["temperature"].attrs["units"] = None  # type: ignore[assignment]
    verdict = QualityGate().evaluate(dataset)

    units = _check(verdict, "units_valid")
    assert units.passed is False
    assert units.blocking is False


def test_absent_unit_is_a_warning() -> None:
    """Missing units are common in real files, so they warn rather than reject."""
    dataset = make_dataset()
    for variable in dataset.data_vars.values():
        variable.attrs.pop("units", None)
    verdict = QualityGate().evaluate(dataset)

    assert verdict.status == "VALID_WITH_WARNINGS"
    assert verdict.passed is True


def test_nan_heavy_dataset_warns_and_can_reject() -> None:
    """A field that is 95% missing cannot support trustworthy statistics."""
    gate = QualityGate(min_finite_fraction=0.5, max_missing_fraction=DEFAULT_MAX_MISSING_FRACTION)
    verdict = gate.evaluate(make_nan_heavy_dataset(0.95))

    finite = _check(verdict, "finite_values")
    assert finite.passed is False
    assert finite.detail["finite_fraction"] < 0.5


def test_mild_gaps_still_pass() -> None:
    """A few missing values are a genuine observation of reality, not a fault."""
    gate = QualityGate(min_finite_fraction=0.5)
    verdict = gate.evaluate(make_nan_heavy_dataset(0.10))

    assert verdict.passed is True


def test_finite_fraction_is_reported_per_variable() -> None:
    """Per-variable detail is what lets an operator find the bad field."""
    gate = QualityGate()
    verdict = gate.evaluate(make_dataset())

    detail = _check(verdict, "finite_values").detail
    assert set(detail["per_variable_finite_fraction"]) == set(
        make_dataset().data_vars
    )


def test_entirely_empty_time_steps_are_counted() -> None:
    """A step where every variable is NaN carries no information."""
    dataset = make_dataset()
    for name in dataset.data_vars:
        values = np.asarray(dataset[name].values, dtype="float64").copy()
        values[0] = np.nan
        dataset[name] = xr.DataArray(values, dims=dataset[name].dims)

    detail = _check(QualityGate().evaluate(dataset), "finite_values").detail
    assert detail["empty_step_fraction"] > 0.0


# --------------------------------------------------------------------------- #
# Verdict reporting
# --------------------------------------------------------------------------- #
def test_rejection_reason_names_the_failing_checks() -> None:
    """A rejection must say which check failed and why."""
    verdict = QualityGate().evaluate(make_non_monotonic_dataset())

    assert "time_monotonic" in verdict.reason
    assert verdict.summary().startswith("REJECTED")


def test_warning_summary_lists_advisory_failures() -> None:
    """A warning verdict names the advisory checks that failed."""
    verdict = QualityGate().evaluate(make_duplicate_time_dataset())

    summary = verdict.summary()
    assert summary.startswith("VALID_WITH_WARNINGS")
    assert "no_duplicate_times" in summary


def test_blocking_failures_are_distinguished_from_warnings() -> None:
    """The two collections must not overlap, or the verdict is ambiguous."""
    verdict = QualityGate().evaluate(make_duplicate_time_dataset())

    blocking = {c.name for c in verdict.blocking_failures}
    warnings = {c.name for c in verdict.warnings}
    assert not (blocking & warnings)


def test_verdict_serialises_for_the_api() -> None:
    """A verdict is returned over HTTP, so it must be JSON-ready."""
    payload = QualityGate().evaluate(make_dataset()).as_dict()

    assert payload["status"] == "VALID"
    assert isinstance(payload["checks"], list)
    assert payload["reason"] == ""


def test_gate_is_deterministic() -> None:
    """The same input always yields the same verdict, so a run is reproducible."""
    dataset = make_dataset()
    gate = QualityGate()

    first = gate.evaluate(dataset).as_dict()
    second = gate.evaluate(dataset).as_dict()

    assert first == second
