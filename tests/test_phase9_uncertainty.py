"""Uncertainty and calibration tests (SIH26078, sections 6-7).

Required coverage: interval generation, bootstrap reproducibility, coverage
calculation and insufficient-sample handling.

Two rules are asserted throughout, because breaking either would be the most
misleading thing this project could do:

* an interval is never described as a probability;
* too few samples yields ``INSUFFICIENT_DATA``, never a number.
"""

from __future__ import annotations

import numpy as np
import pytest

from extreme_weather.evaluation.uncertainty import (
    BOOTSTRAP_METHOD,
    MIN_RESIDUALS_FOR_COVERAGE,
    MIN_RESIDUALS_FOR_INTERVALS,
    UNCERTAINTY_CAVEAT,
    evaluate_coverage,
    residual_intervals,
)

from tests.phase9_support import make_samples


def _bootstrap(count: int = 80, seed: int = 7) -> object:
    """Return a fitted residual bootstrap over ``count`` samples."""
    samples = make_samples(count, seed=seed)
    return residual_intervals(
        validation_true_latitude=samples["true_latitude"],
        validation_predicted_latitude=samples["predicted_latitude"],
        validation_true_longitude=samples["true_longitude"],
        validation_predicted_longitude=samples["predicted_longitude"],
        validation_true_intensity=samples["true_intensity"],
        validation_predicted_intensity=samples["predicted_intensity"],
        seed=seed,
    )


# --------------------------------------------------------------------------- #
# Interval generation
# --------------------------------------------------------------------------- #
def test_intervals_are_produced_from_enough_residuals() -> None:
    """A usable residual set must yield an interval per output."""
    uncertainty = _bootstrap(80).interval_for(12.5, 76.5, 30.0, nominal=0.8)

    assert uncertainty.latitude_interval is not None
    assert uncertainty.longitude_interval is not None
    assert uncertainty.intensity_interval is not None


def test_the_interval_brackets_the_point_prediction() -> None:
    """A residual interval is centred on the prediction."""
    uncertainty = _bootstrap(80).interval_for(12.5, 76.5, 30.0, nominal=0.8)
    low, high = uncertainty.latitude_interval

    assert low <= 12.5 <= high


def test_a_wider_nominal_gives_a_wider_interval() -> None:
    """A 90% interval must contain the 80% interval."""
    bootstrap = _bootstrap(80)
    narrow = bootstrap.interval_for(12.5, 76.5, 30.0, nominal=0.8)
    wide = bootstrap.interval_for(12.5, 76.5, 30.0, nominal=0.9)

    narrow_width = narrow.latitude_interval[1] - narrow.latitude_interval[0]
    wide_width = wide.latitude_interval[1] - wide.latitude_interval[0]
    assert wide_width > narrow_width


def test_the_method_is_always_named() -> None:
    """A reader must know how the interval was produced."""
    uncertainty = _bootstrap(60).interval_for(12.5, 76.5, 30.0)

    assert uncertainty.method == BOOTSTRAP_METHOD
    assert uncertainty.caveat == UNCERTAINTY_CAVEAT


def test_the_caveat_explicitly_denies_probability() -> None:
    """The single most important sentence in this module."""
    assert "not a probability" in UNCERTAINTY_CAVEAT.lower()
    assert "unverified assumption" in UNCERTAINTY_CAVEAT


def test_the_payload_names_the_calibration_dataset() -> None:
    """The interval must state which split produced its width."""
    payload = _bootstrap(60).interval_for(12.5, 76.5, 30.0).as_dict()

    assert payload["uncertainty"]["calibration_dataset"] == "validation"
    assert payload["uncertainty"]["caveat"]


def test_an_invalid_nominal_is_refused() -> None:
    """A nominal outside (0, 1) is not a coverage level."""
    with pytest.raises(ValueError, match="nominal must be"):
        _bootstrap(60).quantiles(1.5)


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #
def test_the_bootstrap_is_reproducible_for_a_fixed_seed() -> None:
    """The same seed must give the same quantiles, or nothing is reproducible."""
    first = _bootstrap(80, seed=99).quantiles(0.8)
    second = _bootstrap(80, seed=99).quantiles(0.8)

    assert first["latitude"][0] == pytest.approx(second["latitude"][0])
    assert first["latitude"][1] == pytest.approx(second["latitude"][1])


def test_different_seeds_give_different_draws() -> None:
    """A seed that changed nothing would mean the seed is being ignored."""
    first = _bootstrap(120, seed=1).quantiles(0.8)
    second = _bootstrap(120, seed=2).quantiles(0.8)

    assert first["intensity"] != second["intensity"]


def test_the_same_seed_reproduces_the_whole_interval_payload() -> None:
    """End-to-end determinism of the reported uncertainty."""
    a = _bootstrap(80, seed=5).interval_for(12.5, 76.5, 30.0, nominal=0.9).as_dict()
    b = _bootstrap(80, seed=5).interval_for(12.5, 76.5, 30.0, nominal=0.9).as_dict()

    assert a == b


def test_non_finite_residuals_are_dropped_not_propagated() -> None:
    """A NaN residual would poison every quantile downstream."""
    samples = make_samples(60)
    predicted = samples["predicted_latitude"].copy()
    predicted[0] = np.nan
    bootstrap = residual_intervals(
        validation_true_latitude=samples["true_latitude"],
        validation_predicted_latitude=predicted,
        validation_true_longitude=samples["true_longitude"],
        validation_predicted_longitude=samples["predicted_longitude"],
        validation_true_intensity=samples["true_intensity"],
        validation_predicted_intensity=samples["predicted_intensity"],
        seed=1,
    )

    assert bootstrap.residuals["latitude"].size == 59
    assert np.all(np.isfinite(bootstrap.residuals["latitude"]))


# --------------------------------------------------------------------------- #
# Insufficient data
# --------------------------------------------------------------------------- #
def test_too_few_residuals_yield_no_interval() -> None:
    """Quantiles over a handful of samples are meaningless."""
    bootstrap = _bootstrap(5)
    bounds = bootstrap.quantiles(0.8)

    assert bounds["latitude"] is None
    assert bounds["intensity"] is None


def test_insufficient_samples_are_reported_as_such() -> None:
    """The absence must be typed, not a silent null."""
    uncertainty = _bootstrap(5).interval_for(12.5, 76.5, 30.0)

    assert uncertainty.latitude_interval is None
    assert uncertainty.sample_count < MIN_RESIDUALS_FOR_INTERVALS


def test_coverage_needs_enough_test_samples() -> None:
    """A coverage fraction over too few samples is sampling noise."""
    samples = make_samples(10, seed=3)
    result = evaluate_coverage(
        _bootstrap(80),
        test_true_latitude=samples["true_latitude"],
        test_predicted_latitude=samples["predicted_latitude"],
        test_true_longitude=samples["true_longitude"],
        test_predicted_longitude=samples["predicted_longitude"],
        test_true_intensity=samples["true_intensity"],
        test_predicted_intensity=samples["predicted_intensity"],
        nominal=0.8,
    )

    assert result.status == "INSUFFICIENT_DATA"
    assert result.empirical_coverage == {}
    assert str(MIN_RESIDUALS_FOR_COVERAGE) in result.reason


def test_insufficient_coverage_produces_no_numbers() -> None:
    """Converting an unmeasured coverage to a float must fail loudly."""
    samples = make_samples(5, seed=3)
    result = evaluate_coverage(
        _bootstrap(80),
        test_true_latitude=samples["true_latitude"],
        test_predicted_latitude=samples["predicted_latitude"],
        test_true_longitude=samples["true_longitude"],
        test_predicted_longitude=samples["predicted_longitude"],
        test_true_intensity=samples["true_intensity"],
        test_predicted_intensity=samples["predicted_intensity"],
    )
    metrics = result.metrics()

    assert all(metric.available is False for metric in metrics)
    assert all(metric.status == "INSUFFICIENT_DATA" for metric in metrics)


# --------------------------------------------------------------------------- #
# Coverage measurement
# --------------------------------------------------------------------------- #
def test_coverage_is_measured_when_there_is_enough_data() -> None:
    """With enough residuals, empirical coverage is reported per output."""
    samples = make_samples(80, seed=4)
    result = evaluate_coverage(
        _bootstrap(80, seed=4),
        test_true_latitude=samples["true_latitude"],
        test_predicted_latitude=samples["predicted_latitude"],
        test_true_longitude=samples["true_longitude"],
        test_predicted_longitude=samples["predicted_longitude"],
        test_true_intensity=samples["true_intensity"],
        test_predicted_intensity=samples["predicted_intensity"],
        nominal=0.8,
    )

    assert result.status == "MEASURED"
    for value in result.empirical_coverage.values():
        assert 0.0 <= value <= 1.0


def test_coverage_records_its_sample_count_and_evaluation_split() -> None:
    """A coverage figure without its sample size is not interpretable."""
    samples = make_samples(80, seed=4)
    payload = evaluate_coverage(
        _bootstrap(80, seed=4),
        test_true_latitude=samples["true_latitude"],
        test_predicted_latitude=samples["predicted_latitude"],
        test_true_longitude=samples["true_longitude"],
        test_predicted_longitude=samples["predicted_longitude"],
        test_true_intensity=samples["true_intensity"],
        test_predicted_intensity=samples["predicted_intensity"],
    ).as_dict()

    assert payload["sample_count"] == 80
    assert payload["evaluated_on"] == "test"


def test_a_miscalibrated_interval_is_not_reported_as_calibrated() -> None:
    """Calibration is measured, not assumed.

    The interval is fitted on validation residuals and checked on a *wider*
    test residual spread, so coverage should fall short of nominal. The project
    must report that honestly rather than claim calibration.
    """
    bootstrap = _bootstrap(200, seed=8)
    generator = np.random.default_rng(99)
    n = 200
    predicted = np.zeros(n)
    result = evaluate_coverage(
        bootstrap,
        test_true_latitude=predicted + generator.normal(0, 10.0, n),
        test_predicted_latitude=predicted,
        test_true_longitude=predicted + generator.normal(0, 10.0, n),
        test_predicted_longitude=predicted,
        test_true_intensity=predicted + generator.normal(0, 10.0, n),
        test_predicted_intensity=predicted,
        nominal=0.8,
    )

    assert result.status == "MEASURED"
    # The test residuals are far wider than the validation ones, so coverage
    # must be measurably below nominal - and must be reported as such.
    assert all(value < 0.8 for value in result.empirical_coverage.values())
    assert result.is_calibrated is False


def test_coverage_metrics_are_measured_not_derived() -> None:
    """Coverage is a measured property of the interval, not an estimate."""
    samples = make_samples(80, seed=4)
    result = evaluate_coverage(
        _bootstrap(80, seed=4),
        test_true_latitude=samples["true_latitude"],
        test_predicted_latitude=samples["predicted_latitude"],
        test_true_longitude=samples["true_longitude"],
        test_predicted_longitude=samples["predicted_longitude"],
        test_true_intensity=samples["true_intensity"],
        test_predicted_intensity=samples["predicted_intensity"],
    )
    metrics = result.metrics()

    assert all(metric.status == "MEASURED" for metric in metrics)
    assert all(metric.sample_count == 80 for metric in metrics)


def test_interval_widths_are_reported_alongside_coverage() -> None:
    """A coverage number without a width says nothing about usefulness."""
    samples = make_samples(80, seed=4)
    payload = evaluate_coverage(
        _bootstrap(80, seed=4),
        test_true_latitude=samples["true_latitude"],
        test_predicted_latitude=samples["predicted_latitude"],
        test_true_longitude=samples["true_longitude"],
        test_predicted_longitude=samples["predicted_longitude"],
        test_true_intensity=samples["true_intensity"],
        test_predicted_intensity=samples["predicted_intensity"],
    ).as_dict()

    assert set(payload["interval_width"]) == {"latitude", "longitude", "intensity"}
