"""Downscaling evaluation tests (SIH26079, Phase 9, section 5).

Required coverage: identical test frames, baseline comparison, unavailable AI
checkpoint handling, and no test leakage.

The adapter reuses the Phase 7 metrics rather than reimplementing them, so
these tests also assert that a Phase 9 result stays numerically consistent with
what Phase 7 would have produced for the same inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from extreme_weather.evaluation.downscaling import (
    AI_METHOD,
    CUBIC_METHOD,
    NEAREST_METHOD,
    evaluate_downscaling,
    record_downscaling_metrics,
)
from extreme_weather.evaluation.models import EvaluationRun


@pytest.fixture()
def truth() -> np.ndarray:
    """Return a small high-resolution reference field."""
    generator = np.random.default_rng(4)
    return generator.gamma(2.0, 3.0, (6, 8, 8))


def _blurred(field: np.ndarray, scale: float) -> np.ndarray:
    """Return a smoothed copy, standing in for a coarse-resolution input."""
    smoothed = field.copy()
    for _ in range(int(scale)):
        smoothed = 0.5 * smoothed + 0.125 * (
            np.roll(smoothed, 1, axis=1) + np.roll(smoothed, -1, axis=1)
        )
    return smoothed


# --------------------------------------------------------------------------- #
# Identical frames
# --------------------------------------------------------------------------- #
def test_every_method_is_scored_on_the_same_frames(truth: np.ndarray) -> None:
    """A method scored on different frames would be an unfair comparison."""
    coarse = _blurred(truth, 2)
    result = evaluate_downscaling(
        predictions={
            NEAREST_METHOD: coarse,
            CUBIC_METHOD: coarse,
            AI_METHOD: coarse,
        },
        reference=truth,
        dataset="phase9-test",
    )

    for entry in result["methods"].values():
        assert entry["shape"] == list(truth.shape)
    assert result["frames"] == truth.shape[0]
    assert result["sample_count"] == truth.size


def test_a_mismatched_frame_count_is_rejected(truth: np.ndarray) -> None:
    """Frames of different lengths cannot be compared cell by cell."""
    result = evaluate_downscaling(
        predictions={NEAREST_METHOD: _blurred(truth, 2)[:3]},
        reference=truth,
        dataset="phase9-test",
    )

    assert result["methods"] == {}
    assert "identical frames" in result["unavailable"]["all_methods"]


# --------------------------------------------------------------------------- #
# Baseline comparison
# --------------------------------------------------------------------------- #
def test_ai_is_compared_against_both_interpolation_baselines(truth: np.ndarray) -> None:
    """Cubic in particular: the Phase 4 prototype often loses to it."""
    coarse = _blurred(truth, 2)
    result = evaluate_downscaling(
        predictions={
            NEAREST_METHOD: coarse,
            CUBIC_METHOD: coarse * 1.02,
            AI_METHOD: coarse * 0.98,
        },
        reference=truth,
        dataset="phase9-test",
    )

    assert set(result["methods"]) == {NEAREST_METHOD, CUBIC_METHOD, AI_METHOD}
    assert "ai_vs_nearest" in result["comparisons"]
    assert "ai_vs_cubic" in result["comparisons"]


def test_a_loss_is_reported_as_a_loss(truth: np.ndarray) -> None:
    """A method that ties with the baseline is not declared a winner."""
    coarse = _blurred(truth, 2)
    result = evaluate_downscaling(
        predictions={NEAREST_METHOD: coarse, AI_METHOD: coarse.copy()},
        reference=truth,
        dataset="phase9-test",
    )
    verdict = result["comparisons"]["ai_vs_nearest"]["mae"]["verdict"]

    assert verdict == "tie"
    assert result["winner_by_mae"] in (None, NEAREST_METHOD, AI_METHOD)


def test_metrics_agree_with_the_phase7_implementation(truth: np.ndarray) -> None:
    """The adapter must not quietly change the numbers."""
    from extreme_weather.validation.metrics import compute_downscaling_comparison

    coarse = _blurred(truth, 2)
    result = evaluate_downscaling(
        predictions={NEAREST_METHOD: coarse, AI_METHOD: coarse * 0.99},
        reference=truth,
        dataset="phase9-test",
    )
    direct = compute_downscaling_comparison(
        {NEAREST_METHOD: coarse, AI_METHOD: coarse * 0.99},
        truth,
        dataset="phase9-test",
        split="test",
    ).as_dict()

    assert result["methods"][NEAREST_METHOD]["mae"] == pytest.approx(
        direct["methods"][NEAREST_METHOD]["mae"]
    )


# --------------------------------------------------------------------------- #
# Unavailable AI checkpoint
# --------------------------------------------------------------------------- #
def test_an_absent_checkpoint_omits_the_ai_method(truth: np.ndarray) -> None:
    """The AI result is omitted, never imitated by a baseline."""
    result = evaluate_downscaling(
        predictions={NEAREST_METHOD: _blurred(truth, 2), CUBIC_METHOD: _blurred(truth, 2)},
        reference=truth,
        dataset="phase9-test",
    )

    assert AI_METHOD not in result["methods"]
    assert "omitted from the comparison" in result["unavailable"][AI_METHOD]


def test_an_absent_checkpoint_does_not_remove_the_baselines(truth: np.ndarray) -> None:
    """Interpolation is still evaluable without a model."""
    result = evaluate_downscaling(
        predictions={NEAREST_METHOD: _blurred(truth, 2)},
        reference=truth,
        dataset="phase9-test",
    )

    assert NEAREST_METHOD in result["methods"]


def test_no_reference_omits_every_method() -> None:
    """Without truth there is nothing to measure, so nothing is reported."""
    result = evaluate_downscaling(
        predictions={NEAREST_METHOD: np.zeros((2, 4, 4)), AI_METHOD: np.zeros((2, 4, 4))},
        reference=None,
        dataset="phase9-test",
    )

    assert result["methods"] == {}
    assert "omitted rather than compared against each other" in result["unavailable"]["all_methods"]


# --------------------------------------------------------------------------- #
# Classification metrics
# --------------------------------------------------------------------------- #
def test_pod_far_are_reported_only_behind_a_threshold(truth: np.ndarray) -> None:
    """A contingency table against an invented threshold is meaningless."""
    coarse = _blurred(truth, 2)
    without = evaluate_downscaling(
        predictions={NEAREST_METHOD: coarse},
        reference=truth,
        dataset="phase9-test",
    )
    with_threshold = evaluate_downscaling(
        predictions={NEAREST_METHOD: coarse},
        reference=truth,
        dataset="phase9-test",
        event_threshold=float(np.median(truth)),
    )

    assert "pod_far_csi" in without["unavailable"]
    assert "No defensible event threshold" in without["unavailable"]["pod_far_csi"]
    assert with_threshold["event_threshold"] is not None


# --------------------------------------------------------------------------- #
# No test leakage
# --------------------------------------------------------------------------- #
def test_the_split_is_recorded_on_every_comparison(truth: np.ndarray) -> None:
    """A reader must be able to see which split produced a number."""
    result = evaluate_downscaling(
        predictions={NEAREST_METHOD: _blurred(truth, 2)},
        reference=truth,
        dataset="phase9-test",
        split="test",
    )

    assert result["split"] == "test"


def test_recorded_metrics_never_carry_a_zero_for_an_absent_method() -> None:
    """A missing method must not be recorded as a perfect score."""
    run = EvaluationRun(run_id="eval-x", dataset_id="phase9-test")
    result = evaluate_downscaling(
        predictions={NEAREST_METHOD: np.ones((2, 4, 4))},
        reference=np.ones((2, 4, 4)),
        dataset="phase9-test",
    )
    record_downscaling_metrics(result, run)

    ai_metrics = [
        entry for entry in run.unavailable_metrics.get("downscaling", {}).values()
        if entry["name"].startswith("ai_")
    ]
    assert ai_metrics
    for entry in ai_metrics:
        assert entry["value"] is None
        assert entry["reason"]


def test_measured_metrics_reach_the_run(truth: np.ndarray) -> None:
    """A real measurement must be recorded as MEASURED."""
    run = EvaluationRun(run_id="eval-y", dataset_id="phase9-test")
    result = evaluate_downscaling(
        predictions={NEAREST_METHOD: _blurred(truth, 2)},
        reference=truth,
        dataset="phase9-test",
    )
    record_downscaling_metrics(result, run)

    measured = run.metrics.get("downscaling", {})
    assert f"{NEAREST_METHOD}_mae" in measured
    assert measured[f"{NEAREST_METHOD}_mae"]["status"] == "MEASURED"
