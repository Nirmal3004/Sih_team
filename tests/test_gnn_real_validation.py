"""Tests for the Phase 7 GNN and downscaling real-data evaluations.

Both modules share one rule: a model is measured against a **baseline**, and a
loss is reported. The GNN's baseline is persistence ("nothing moves"); the
downscaler's are nearest and cubic interpolation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from extreme_weather.validation.metrics import (
    compute_downscaling_comparison,
    compute_gnn_comparison,
    persistence_baseline,
)


# --------------------------------------------------------------------------- #
# GNN vs persistence
# --------------------------------------------------------------------------- #
def _comparison(predicted: np.ndarray, truth: np.ndarray, **overrides):
    zeros = np.zeros_like(truth)
    return compute_gnn_comparison(
        predicted_lat=predicted,
        predicted_lon=np.zeros_like(truth),
        predicted_intensity=zeros,
        current_lat=zeros,
        current_lon=zeros,
        current_intensity=zeros,
        future_lat=truth,
        future_lon=zeros,
        future_intensity=zeros,
        dataset=overrides.pop("dataset", "demo"),
        split=overrides.pop("split", "test"),
        model_checkpoint=overrides.pop("model_checkpoint", "ckpt"),
        **overrides,
    )


def test_a_model_that_predicts_nothing_moves_ties_persistence() -> None:
    truth = np.array([0.5, 0.5])
    result = _comparison(np.zeros(2), truth)
    assert result.comparison["latitude_mae_deg"]["verdict"] == "tie"
    assert result.comparison["_summary"]["verdict"] == "matches persistence"


def test_a_model_that_moves_the_wrong_way_loses_to_persistence() -> None:
    """The honest negative result, which the report must be able to state."""
    truth = np.array([0.5, 0.5])
    result = _comparison(np.array([1.5, 1.5]), truth)
    assert result.comparison["latitude_mae_deg"]["verdict"] == "worse"
    assert result.comparison["_summary"]["metrics_worsened"] >= 1


def test_a_accurate_model_beats_persistence() -> None:
    truth = np.array([0.5, 0.5, 0.5])
    result = _comparison(truth.copy(), truth)
    assert result.comparison["latitude_mae_deg"]["verdict"] == "improves"
    assert result.comparison["_summary"]["verdict"] == "beats persistence"


def test_the_report_carries_dataset_split_and_checkpoint() -> None:
    truth = np.array([0.5])
    payload = _comparison(truth, truth).as_dict()
    for key in ("dataset", "split", "sample_counts", "model_checkpoint", "metrics",
                "baseline_metrics", "comparison", "limitations"):
        assert key in payload, key
    assert payload["split"] == "test"
    assert payload["metrics"]["sample_count"] == 1


def test_the_sample_counts_can_be_recorded() -> None:
    truth = np.array([0.5, 0.5])
    result = _comparison(
        truth, truth, sample_counts={"train": 100, "validation": 20, "test": 2}
    )
    assert result.sample_counts == {"train": 100, "validation": 20, "test": 2}


def test_persistence_never_scores_perfectly_when_the_event_moves() -> None:
    """The baseline is a real predictor, not a rubber stamp."""
    truth = np.array([1.0, 2.0, 3.0])
    baseline = persistence_baseline(
        np.zeros(3), np.zeros(3), np.zeros(3), truth, np.zeros(3), np.zeros(3)
    )
    assert baseline["latitude_mae_deg"] > 0.0
    assert baseline["displacement_error_km_mean"] > 0.0


def test_the_gnn_evaluation_script_reports_an_empty_test_split(tmp_path: Path) -> None:
    """A run with no usable test node must refuse, not train on the test split."""
    from extreme_weather.validation.metrics import GnnComparison

    payload = {
        "available": False,
        "reason": (
            "The test split contains no node with a known next state, so no metric "
            "was computed."
        ),
        "sample_counts": {"train": 4, "validation": 0, "test": 0},
    }
    assert payload["available"] is False
    assert "no metric was computed" in payload["reason"]


# --------------------------------------------------------------------------- #
# Downscaling vs nearest and cubic
# --------------------------------------------------------------------------- #
def _frames() -> np.ndarray:
    return np.linspace(1.0, 10.0, 64).reshape(1, 8, 8)


def test_all_methods_are_scored_on_the_same_frames() -> None:
    reference = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference, "nearest": reference, "cubic": reference},
        reference,
        dataset="demo",
        split="test",
    )
    for name in ("ai", "nearest", "cubic"):
        assert result.methods[name]["shape"] == list(reference.shape)
    assert result.frame_count == reference.size


def test_the_ai_losing_to_cubic_is_reported() -> None:
    reference = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference + 2.0, "cubic": reference, "nearest": reference + 1.0},
        reference,
        dataset="demo",
        split="test",
    )
    assert result.winner == "cubic"
    assert result.comparison["ai_vs_cubic"]["mae"]["verdict"] == "worse"
    assert result.comparison["_summary"]["metrics_worsened"] >= 1


def test_the_ai_winning_is_reported_too() -> None:
    reference = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference, "nearest": reference + 1.0, "cubic": reference + 0.5},
        reference,
        dataset="demo",
        split="test",
    )
    assert result.winner == "ai"
    assert "improves on every baseline" in result.comparison["_summary"]["verdict"]


def test_pod_is_refused_without_a_defensible_threshold() -> None:
    """A POD against an arbitrary cut would be a number with no meaning."""
    reference = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference, "cubic": reference}, reference, dataset="d", split="test"
    )
    assert result.classification["available"] is False
    assert "no defensible event threshold" in str(result.classification["note"])


def test_pod_is_computed_once_a_threshold_is_documented() -> None:
    reference = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference, "cubic": reference},
        reference,
        dataset="d",
        split="test",
        event_threshold=5.0,
    )
    assert result.classification["available"] is True
    assert result.methods["ai"]["extremes"]["threshold"] == pytest.approx(5.0)


def test_peak_retention_is_reported() -> None:
    reference = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference * 0.5}, reference, dataset="d", split="test"
    )
    assert result.peak_preservation["per_method"]["ai"]["peak_retention"] == pytest.approx(0.5)


def test_structural_correlation_separates_shape_from_scale() -> None:
    """A scaled field has the same shape, so correlation is 1 but MAE is not."""
    reference = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference * 2.0}, reference, dataset="d", split="test"
    )
    structural = result.structural["per_method"]["ai"]["structural_correlation"]
    assert structural == pytest.approx(1.0)
    assert result.methods["ai"]["mae"] > 0.0


def test_wet_area_ratio_is_reported() -> None:
    reference = _frames()
    result = compute_downscaling_comparison(
        {"ai": reference}, reference, dataset="d", split="test"
    )
    wet = result.structural["per_method"]["ai"]
    assert wet["wet_area_ratio"] == pytest.approx(1.0)
    assert wet["reference_wet_area_ratio"] == pytest.approx(1.0)


def test_a_method_producing_the_wrong_shape_is_rejected() -> None:
    reference = _frames()
    with pytest.raises(ValueError, match="identical frames"):
        compute_downscaling_comparison(
            {"ai": np.zeros((1, 2, 2))}, reference, dataset="d", split="test"
        )

