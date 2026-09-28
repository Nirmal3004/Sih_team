"""Metric tests (SIH26078, Phase 4).

These are agreement metrics against a **synthetic** reference. Undefined cases
must return ``None`` rather than a fabricated 0 or 1, and that distinction is
asserted explicitly.
"""

from __future__ import annotations

import numpy as np
import pytest

from extreme_weather.downscaling.metrics import (
    DEFAULT_WET_THRESHOLD,
    classification_metrics,
    error_metrics,
    evaluate,
    pearson_correlation,
)

# --- continuous error metrics -----------------------------------------------


def test_mae_of_a_known_error():
    prediction = np.array([2.0, 4.0, 6.0])
    reference = np.array([1.0, 4.0, 9.0])
    assert error_metrics("m", prediction, reference).mae == pytest.approx(4.0 / 3.0)


def test_rmse_of_a_known_error():
    prediction = np.array([2.0, 4.0, 6.0])
    reference = np.array([1.0, 4.0, 9.0])
    assert error_metrics("m", prediction, reference).rmse == pytest.approx(
        np.sqrt((1.0 + 0.0 + 9.0) / 3.0)
    )


def test_bias_is_signed():
    """Positive bias means over-prediction; the sign must be preserved."""
    over = error_metrics("m", np.array([2.0, 2.0]), np.array([1.0, 1.0]))
    under = error_metrics("m", np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    assert over.bias == pytest.approx(1.0)
    assert under.bias == pytest.approx(-1.0)


def test_metrics_are_zero_for_a_perfect_prediction():
    field = np.array([1.0, 5.0, 3.0])
    result = error_metrics("m", field, field)
    assert result.mae == 0.0
    assert result.rmse == 0.0
    assert result.bias == 0.0
    assert result.correlation == pytest.approx(1.0)
    assert result.n_points == 3


def test_rmse_is_at_least_the_mae():
    """A guaranteed relation; a violation would mean an implementation bug."""
    rng = np.random.default_rng(0)
    result = error_metrics("m", rng.normal(size=500), rng.normal(size=500))
    assert result.rmse >= result.mae


def test_metrics_reject_mismatched_shapes():
    with pytest.raises(ValueError, match="shapes differ"):
        error_metrics("m", np.zeros(4), np.zeros(5))


def test_metrics_reject_an_empty_array():
    with pytest.raises(ValueError, match="empty array"):
        error_metrics("m", np.zeros(0), np.zeros(0))


# --- correlation ------------------------------------------------------------


def test_correlation_of_a_perfect_linear_relation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson_correlation(x, 2.0 * x + 1.0) == pytest.approx(1.0)


def test_correlation_of_a_reversed_relation():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert pearson_correlation(x, -x) == pytest.approx(-1.0)


def test_correlation_is_undefined_for_a_constant_field():
    """0/0 is 'not computable', not 'uncorrelated' - so it must be None."""
    constant = np.ones(10)
    assert pearson_correlation(constant, np.arange(10.0)) is None
    assert pearson_correlation(np.arange(10.0), constant) is None


def test_correlation_is_undefined_for_non_finite_input():
    assert pearson_correlation(np.array([1.0, 2.0, np.nan, 4.0]), np.arange(4.0)) is None


# --- wet/dry classification ------------------------------------------------


def test_perfect_wet_dry_forecast():
    field = np.array([1.0, 0.0, 1.0, 0.0])
    result = classification_metrics("m", field, field)
    assert result.pod == pytest.approx(1.0)
    assert result.far == pytest.approx(0.0)
    assert result.csi == pytest.approx(1.0)
    assert result.frequency_bias == 0
    assert result.hits == 2
    assert (result.misses, result.false_alarms) == (0, 0)


def test_pod_and_far_of_a_known_contingency_table():
    # a=2 hits, b=1 false alarm, c=2 misses, d=5 correct negatives.
    reference = np.array([1.0, 1.0, 1.0, 1.0, 0, 0, 0, 0, 0, 0])
    prediction = np.array([1.0, 1.0, 0.0, 0.0, 1.0, 0, 0, 0, 0, 0])
    result = classification_metrics("m", prediction, reference, threshold=0.5)
    assert result.hits == 2
    assert result.false_alarms == 1
    assert result.misses == 2
    assert result.correct_negatives == 5
    assert result.pod == pytest.approx(2 / 4)
    assert result.far == pytest.approx(1 / 3)
    assert result.csi == pytest.approx(2 / 5)
    assert result.frequency_bias == (2 + 1) - (2 + 2)


def test_predicting_rain_everywhere_trades_pod_against_far():
    """The reason POD must never be read on its own."""
    result = classification_metrics(
        "always-wet", np.ones(4), np.array([1.0, 0.0, 0.0, 0.0]), threshold=0.5
    )
    assert result.pod == pytest.approx(1.0)
    assert result.far == pytest.approx(0.75)


def test_pod_is_none_when_no_events_are_observed():
    assert classification_metrics("m", np.ones(5), np.zeros(5), 0.5).pod is None


def test_far_is_none_when_nothing_is_predicted():
    result = classification_metrics("m", np.zeros(5), np.ones(5), 0.5)
    assert result.far is None
    assert result.pod == pytest.approx(0.0)


def test_csi_is_none_for_an_empty_contingency():
    assert classification_metrics("m", np.zeros(5), np.zeros(5), 0.5).csi is None


def test_the_threshold_is_configurable():
    field = np.array([0.05, 0.5, 0.5])
    # With a low threshold 0.05 counts as wet in both fields.
    assert classification_metrics("m", field, field, 0.01).pod == pytest.approx(1.0)
    # With a high threshold nothing is wet at all.
    assert classification_metrics("m", field, field, 1.0).pod is None


def test_default_threshold_is_documented():
    assert DEFAULT_WET_THRESHOLD == 0.1


# --- combined ---------------------------------------------------------------


def test_evaluate_returns_every_metric():
    rng = np.random.default_rng(1)
    reference = rng.random((20, 20))
    prediction = reference + rng.normal(scale=0.01, size=reference.shape)
    result = evaluate("m", prediction, reference)
    payload = result.as_dict()
    for key in ("method", "mae", "rmse", "bias", "correlation", "pod", "far", "csi"):
        assert key in payload
    assert result.classification is not None
    assert result.error.n_points == 400


def test_evaluate_can_skip_the_classification_metrics():
    result = evaluate("m", np.ones((4, 4)), np.ones((4, 4)), threshold=None)
    assert result.classification is None
    assert result.as_dict()["pod"] is None


def test_metrics_round_trip_through_a_dict():
    payload = evaluate("m", np.ones((4, 4)), np.full((4, 4), 2.0)).as_dict()
    assert payload["mae"] == pytest.approx(1.0)
    assert payload["bias"] == pytest.approx(-1.0)
