"""GNN baseline and downscaling evaluation tests (SIH26078, sections 3-5).

Required coverage: persistence, constant velocity, the GNN itself, insufficient
target handling, multi-step rollout, identical test frames, and honest handling
of an unavailable AI checkpoint.

The central honesty assertions are the ones that matter most here: a model that
cannot be scored must be reported as unavailable, never as zero, and a baseline
must not be silently dropped when its inputs are missing.
"""

from __future__ import annotations

import numpy as np
import pytest

from extreme_weather.evaluation.baselines import (
    CONSTANT_VELOCITY,
    PERSISTENCE,
    constant_velocity_baseline,
    describe_baselines,
    evaluate_baseline,
)
from extreme_weather.evaluation.evaluator import (
    evaluate_gnn_baselines,
    evaluate_multistep_rollout,
    summarise_gnn_comparison,
)
from extreme_weather.evaluation.models import INSUFFICIENT_DATA, NOT_AVAILABLE

from tests.phase9_support import make_samples


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def test_persistence_holds_the_current_state() -> None:
    """Persistence predicts the present unchanged, so its error is the movement."""
    samples = make_samples(30)
    metrics = evaluate_baseline(
        PERSISTENCE,
        current_latitude=samples["current_latitude"],
        current_longitude=samples["current_longitude"],
        current_intensity=samples["current_intensity"],
        future_latitude=samples["true_latitude"],
        future_longitude=samples["true_longitude"],
        future_intensity=samples["true_intensity"],
    )

    # The truth moves 0.10 deg per step, so persistence is off by exactly that.
    assert metrics["latitude_mae_deg"] == pytest.approx(0.10, abs=1e-6)
    assert metrics["longitude_mae_deg"] == pytest.approx(0.15, abs=1e-6)
    assert metrics["sample_count"] == 30


def test_persistence_on_a_stationary_event_is_near_zero() -> None:
    """With no movement, persistence is exact - the sanity floor of the metric."""
    lat = np.full(10, 12.0)
    lon = np.full(10, 76.0)
    intensity = np.full(10, 30.0)
    metrics = evaluate_baseline(
        PERSISTENCE,
        current_latitude=lat,
        current_longitude=lon,
        current_intensity=intensity,
        future_latitude=lat.copy(),
        future_longitude=lon.copy(),
        future_intensity=intensity.copy(),
    )

    assert metrics["latitude_mae_deg"] == pytest.approx(0.0)
    assert metrics["displacement_error_km_mean"] == pytest.approx(0.0)


def test_persistence_rejects_mismatched_shapes() -> None:
    """Scoring different sample counts against each other would be meaningless."""
    with pytest.raises(ValueError, match="same shape"):
        evaluate_baseline(
            PERSISTENCE,
            current_latitude=np.zeros(5),
            current_longitude=np.zeros(5),
            current_intensity=np.zeros(5),
            future_latitude=np.zeros(4),
            future_longitude=np.zeros(4),
            future_intensity=np.zeros(4),
        )


def test_persistence_rejects_an_empty_sample() -> None:
    """An empty baseline has no error to report."""
    empty = np.array([])
    with pytest.raises(ValueError, match="at least one sample"):
        evaluate_baseline(
            PERSISTENCE,
            current_latitude=empty,
            current_longitude=empty,
            current_intensity=empty,
            future_latitude=empty,
            future_longitude=empty,
            future_intensity=empty,
        )


# --------------------------------------------------------------------------- #
# Constant velocity
# --------------------------------------------------------------------------- #
def test_constant_velocity_extrapolates_the_last_step() -> None:
    """For steady motion, extrapolating the last step reproduces the truth."""
    samples = make_samples(30)
    metrics = constant_velocity_baseline(
        samples["current_latitude"],
        samples["current_longitude"],
        samples["current_intensity"],
        samples["previous_latitude"],
        samples["previous_longitude"],
        samples["true_latitude"],
        samples["true_longitude"],
        samples["true_intensity"],
    )

    assert metrics["latitude_mae_deg"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["longitude_mae_deg"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["displacement_error_km_mean"] == pytest.approx(0.0, abs=1e-3)


def test_constant_velocity_beats_persistence_on_moving_events() -> None:
    """The reason this baseline exists: persistence is weak for anything moving."""
    samples = make_samples(40)
    persistence = evaluate_baseline(
        PERSISTENCE,
        current_latitude=samples["current_latitude"],
        current_longitude=samples["current_longitude"],
        current_intensity=samples["current_intensity"],
        future_latitude=samples["true_latitude"],
        future_longitude=samples["true_longitude"],
        future_intensity=samples["true_intensity"],
    )
    velocity = evaluate_baseline(
        CONSTANT_VELOCITY,
        current_latitude=samples["current_latitude"],
        current_longitude=samples["current_longitude"],
        current_intensity=samples["current_intensity"],
        previous_latitude=samples["previous_latitude"],
        previous_longitude=samples["previous_longitude"],
        future_latitude=samples["true_latitude"],
        future_longitude=samples["true_longitude"],
        future_intensity=samples["true_intensity"],
    )

    assert (
        velocity["displacement_error_km_mean"] < persistence["displacement_error_km_mean"]
    )


def test_constant_velocity_holds_intensity_rather_than_extrapolating() -> None:
    """A documented choice: intensity is not trend-extrapolated."""
    samples = make_samples(20)
    metrics = constant_velocity_baseline(
        samples["current_latitude"],
        samples["current_longitude"],
        samples["current_intensity"],
        samples["previous_latitude"],
        samples["previous_longitude"],
        samples["true_latitude"],
        samples["true_longitude"],
        samples["true_intensity"],
    )
    expected = float(
        np.mean(
            np.abs(samples["true_intensity"] - samples["current_intensity"])
        )
    )

    assert metrics["intensity_mae"] == pytest.approx(expected)


def test_constant_velocity_requires_the_previous_step() -> None:
    """Without a previous position no velocity can be formed."""
    samples = make_samples(10)
    with pytest.raises(ValueError, match="previous step"):
        evaluate_baseline(
            CONSTANT_VELOCITY,
            current_latitude=samples["current_latitude"],
            current_longitude=samples["current_longitude"],
            current_intensity=samples["current_intensity"],
            future_latitude=samples["true_latitude"],
            future_longitude=samples["true_longitude"],
            future_intensity=samples["true_intensity"],
        )


def test_unknown_baseline_is_rejected() -> None:
    """A misspelled baseline must fail loudly, not score as zero."""
    samples = make_samples(5)
    with pytest.raises(ValueError, match="Unknown baseline"):
        evaluate_baseline(
            "persistance",
            current_latitude=samples["current_latitude"],
            current_longitude=samples["current_longitude"],
            current_intensity=samples["current_intensity"],
            future_latitude=samples["true_latitude"],
            future_longitude=samples["true_longitude"],
            future_intensity=samples["true_intensity"],
        )


def test_baseline_definitions_are_documented() -> None:
    """A reader must be able to see what each baseline assumes."""
    definitions = describe_baselines()

    assert set(definitions) == {PERSISTENCE, CONSTANT_VELOCITY}
    for entry in definitions.values():
        assert entry["formula"]
        assert entry["description"]
        assert entry["why"]


# --------------------------------------------------------------------------- #
# GNN versus baselines
# --------------------------------------------------------------------------- #
def test_all_three_methods_are_scored_on_identical_samples() -> None:
    """Any difference in the numbers must be the method, not the sample."""
    result = evaluate_gnn_baselines(make_samples(40), model_available=True)

    counts = {
        name: entry.get("sample_count")
        for name, entry in result["methods"].items()
    }
    assert set(counts.values()) == {40}


def test_a_skillful_model_beats_persistence() -> None:
    """A near-perfect model must be reported as improving on the baseline."""
    result = evaluate_gnn_baselines(
        make_samples(40, model_skill=0.1), model_available=True
    )
    verdict = result["comparisons"]["gnn_vs_persistence"]["displacement_error_km_mean"]

    assert verdict["verdict"] == "improves"
    assert verdict["improves_on_baseline"] is True


def test_a_useless_model_is_reported_as_worse() -> None:
    """A model that predicts noise must be reported as losing, not averaged away."""
    noisy = make_samples(40, model_skill=50.0)
    result = evaluate_gnn_baselines(noisy, model_available=True)
    verdict = result["comparisons"]["gnn_vs_persistence"]["displacement_error_km_mean"]

    assert verdict["verdict"] == "worse"
    assert verdict["improves_on_baseline"] is False


def test_an_unavailable_model_is_reported_not_simulated() -> None:
    """No checkpoint means unavailable - never a zero and never a guess."""
    result = evaluate_gnn_baselines(make_samples(20), model_available=False)
    gnn = result["methods"]["gnn"]

    assert gnn["status"] == NOT_AVAILABLE
    assert "latitude_mae_deg" not in gnn
    assert "no trained gnn checkpoint" in gnn["reason"].lower()
    assert "gnn" in result["unavailable"]


def test_baselines_are_still_scored_when_the_model_is_absent() -> None:
    """The model being unavailable does not invalidate the baselines."""
    result = evaluate_gnn_baselines(make_samples(20), model_available=False)

    assert "latitude_mae_deg" in result["methods"][PERSISTENCE]
    assert "latitude_mae_deg" in result["methods"][CONSTANT_VELOCITY]


def test_non_finite_predictions_exclude_the_model_entirely() -> None:
    """A NaN must not be scored as a small error."""
    samples = make_samples(20)
    samples["predicted_latitude"] = samples["predicted_latitude"].copy()
    samples["predicted_latitude"][3] = np.nan
    result = evaluate_gnn_baselines(samples, model_available=True)
    gnn = result["methods"]["gnn"]

    assert gnn["status"] == NOT_AVAILABLE
    assert "non-finite" in gnn["reason"]


def test_missing_previous_step_reports_constant_velocity_as_insufficient() -> None:
    """The baseline is reported as not computable, not as zero error."""
    result = evaluate_gnn_baselines(
        make_samples(20, constant_velocity=False), model_available=True
    )
    velocity = result["methods"][CONSTANT_VELOCITY]

    assert velocity["status"] == INSUFFICIENT_DATA
    assert "latitude_mae_deg" not in velocity
    assert "rather than being reported as zero error" in velocity["reason"]


def test_empty_samples_are_refused() -> None:
    """With no next-state targets there is nothing to evaluate."""
    empty = {k: np.array([]) for k in make_samples(3)}

    with pytest.raises(ValueError, match="at least one sample"):
        evaluate_gnn_baselines(empty, model_available=True)


def test_the_horizon_is_recorded() -> None:
    """A reader must know which horizon a number belongs to."""
    result = evaluate_gnn_baselines(make_samples(10), model_available=True, horizon=3)

    assert result["horizon"] == 3
    assert result["horizon_kind"] == "3_step_rollout"


def test_summary_never_claims_overall_superiority() -> None:
    """A per-metric statement must not be inflated into a general claim."""
    result = evaluate_gnn_baselines(make_samples(40, model_skill=0.1), model_available=True)
    summary = summarise_gnn_comparison(result)

    assert "not an overall claim" in summary
    assert "better than persistence" in summary


def test_summary_says_so_when_the_model_was_not_scored() -> None:
    """An unscored model must not be described in comparative terms."""
    result = evaluate_gnn_baselines(make_samples(10), model_available=False)

    assert "not scored" in summarise_gnn_comparison(result)


# --------------------------------------------------------------------------- #
# Multi-step rollout
# --------------------------------------------------------------------------- #
def test_horizons_are_reported_separately_and_never_pooled() -> None:
    """A rollout compounds its own error, so pooling would hide that."""
    rollout = evaluate_multistep_rollout(
        {
            1: make_samples(30, seed=1),
            2: make_samples(20, seed=2, model_skill=1.5),
            3: make_samples(10, seed=3, model_skill=2.5),
        },
        model_available=True,
    )

    assert set(rollout["horizons"]) == {"1", "2", "3"}
    for entry in rollout["horizons"].values():
        assert "sample_count" in entry
        assert entry["horizon_kind"] in ("one_step", "2_step_rollout", "3_step_rollout")


def test_a_horizon_without_targets_is_reported_insufficient() -> None:
    """A horizon with no data is reported, not silently omitted."""
    empty = {k: np.array([]) for k in make_samples(3)}
    rollout = evaluate_multistep_rollout({1: make_samples(20), 2: empty}, model_available=True)

    assert rollout["horizons"]["2"]["status"] == INSUFFICIENT_DATA
    assert "at least one sample" in rollout["horizons"]["2"]["reason"]


def test_worse_skill_at_longer_horizons_is_visible_not_averaged() -> None:
    """The evaluation must expose, not smooth away, horizon degradation."""
    rollout = evaluate_multistep_rollout(
        {
            1: make_samples(40, seed=1, model_skill=0.5),
            3: make_samples(40, seed=1, model_skill=3.0),
        },
        model_available=True,
    )
    one = rollout["horizons"]["1"]["methods"]["gnn"]["displacement_error_km_mean"]
    three = rollout["horizons"]["3"]["methods"]["gnn"]["displacement_error_km_mean"]

    assert three > one
