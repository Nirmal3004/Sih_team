"""Dataset, split and normalisation tests (SIH26078, Phase 4).

The leakage tests here are the important ones. Adjacent weather frames are nearly
identical, so a shuffled split would leak the answer into training and the
resulting numbers would look excellent while meaning nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

from extreme_weather.downscaling.dataset import (
    NormalizationStats,
    SplitConfig,
    SplitIndices,
    bilinear_baseline,
    build_downscaling_dataset,
    fit_normalization,
    load_coarse_dataset,
)
from extreme_weather.downscaling.features import FEATURE_COUNT

from .downscaling_fixtures import FINE_RESOLUTION, make_coarse_field

# --- shapes -----------------------------------------------------------------


def test_dataset_array_shapes(tiny_downscaling_dataset):
    ds = tiny_downscaling_dataset
    n = ds.sample_count
    assert ds.features.shape == (n, FEATURE_COUNT, ds.input_shape[0], ds.input_shape[1])
    assert ds.targets.shape == (n, 1, ds.output_shape[0], ds.output_shape[1])
    assert ds.references.shape == (n, ds.output_shape[0], ds.output_shape[1])
    assert ds.baselines.shape == ds.references.shape
    assert ds.coarse.shape == (n, ds.input_shape[0], ds.input_shape[1])


def test_dataset_resolutions_are_detected(tiny_downscaling_dataset):
    assert tiny_downscaling_dataset.input_resolution_deg == pytest.approx(0.125)
    assert tiny_downscaling_dataset.output_resolution_deg == pytest.approx(FINE_RESOLUTION, abs=1e-6)
    assert tiny_downscaling_dataset.scale_factor == pytest.approx(2.5, abs=1e-6)


def test_targets_are_non_negative_residuals(tiny_downscaling_dataset):
    """The target matches the ReLU constraint of the model output."""
    assert (tiny_downscaling_dataset.targets >= 0.0).all()


def test_target_equals_reference_minus_baseline(tiny_downscaling_dataset):
    ds = tiny_downscaling_dataset
    expected = np.maximum(ds.references - ds.baselines, 0.0)
    assert ds.targets[:, 0] == pytest.approx(expected, abs=1e-6)


def test_dataset_rejects_inconsistent_inputs():
    field, latitude, longitude, timestamps = make_coarse_field(time_steps=3, size=5)
    with pytest.raises(ValueError, match="At least 3 time steps"):
        build_downscaling_dataset(
            field[:2], latitude, longitude, timestamps[:2], FINE_RESOLUTION
        )
    with pytest.raises(ValueError, match="timestamps"):
        build_downscaling_dataset(field, latitude, longitude, timestamps[:2], FINE_RESOLUTION)
    with pytest.raises(ValueError, match="do not match"):
        build_downscaling_dataset(
            field, latitude[:-1], longitude, timestamps, FINE_RESOLUTION
        )
    with pytest.raises(ValueError, match=r"\(time, lat, lon\)"):
        build_downscaling_dataset(
            field[0], latitude, longitude, timestamps[:1], FINE_RESOLUTION
        )


# --- chronological split ---------------------------------------------------


def test_split_is_chronological_not_shuffled(tiny_downscaling_dataset):
    """The core invariant: every training frame precedes every test frame."""
    ds = tiny_downscaling_dataset
    assert ds.split.train == sorted(ds.split.train)
    assert ds.split.validation == sorted(ds.split.validation)
    assert ds.split.test == sorted(ds.split.test)
    assert max(ds.split.train) < min(ds.split.validation)
    assert max(ds.split.validation) < min(ds.split.test)


def test_split_matches_the_timestamps(tiny_downscaling_dataset):
    """The index blocks must follow the valid times, not the sample list."""
    assert tiny_downscaling_dataset.split.is_chronological(tiny_downscaling_dataset.timestamps)


def test_split_blocks_do_not_overlap(tiny_downscaling_dataset):
    ds = tiny_downscaling_dataset
    flat = ds.split.train + ds.split.validation + ds.split.test
    assert len(flat) == len(set(flat))
    assert sorted(flat) == list(range(ds.sample_count))


def test_every_block_is_populated(tiny_downscaling_dataset):
    sizes = tiny_downscaling_dataset.split.sizes
    assert sizes["train"] > 0
    assert sizes["test"] > 0


def test_split_fractions_are_validated():
    assert SplitConfig().train_end == pytest.approx(0.6)
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitConfig(train=0.5, validation=0.2, test=0.2)
    with pytest.raises(ValueError, match=">= 0"):
        SplitConfig(train=-0.1, validation=0.5, test=0.6)
    with pytest.raises(ValueError, match="training fraction"):
        SplitConfig(train=0.0, validation=0.5, test=0.5)
    with pytest.raises(ValueError, match="test fraction"):
        SplitConfig(train=0.8, validation=0.2, test=0.0)


def test_split_indices_rejects_an_unknown_block():
    with pytest.raises(ValueError, match="Unknown split block"):
        SplitIndices(train=[0]).of("holdout")


def test_custom_fractions_shift_the_boundary():
    field, latitude, longitude, timestamps = make_coarse_field(time_steps=20, size=5)
    generous = build_downscaling_dataset(
        field, latitude, longitude, timestamps, FINE_RESOLUTION,
        split_config=SplitConfig(train=0.8, validation=0.1, test=0.1),
    )
    modest = build_downscaling_dataset(
        field, latitude, longitude, timestamps, FINE_RESOLUTION,
        split_config=SplitConfig(train=0.4, validation=0.3, test=0.3),
    )
    assert generous.sizes["train"] > modest.sizes["train"]
    assert modest.sizes["test"] > generous.sizes["test"]


def test_time_subset_is_honoured():
    field, latitude, longitude, timestamps = make_coarse_field(time_steps=12, size=5)
    subset = build_downscaling_dataset(
        field, latitude, longitude, timestamps, FINE_RESOLUTION, time_subset=slice(0, 6)
    )
    assert subset.sample_count == 6


# --- normalisation ---------------------------------------------------------


def test_normalization_uses_training_data_only(tiny_downscaling_dataset):
    """The statistics must come from the training block and nothing else."""
    ds = tiny_downscaling_dataset
    refitted = fit_normalization(ds.features, ds.split.train)
    assert np.allclose(refitted.mean, ds.normalization.mean, atol=1e-6)
    assert np.allclose(refitted.std, ds.normalization.std, atol=1e-6)
    assert "training split only" in ds.normalization.source


def test_normalization_differs_when_the_test_block_is_included(tiny_downscaling_dataset):
    """Including the test block changes the statistics - proof there is no leak."""
    ds = tiny_downscaling_dataset
    training_only = fit_normalization(ds.features, ds.split.train)
    everything = fit_normalization(ds.features, list(range(ds.sample_count)))
    assert not np.allclose(training_only.mean, everything.mean, atol=1e-4)


def test_normalization_standardises_a_batch(tiny_downscaling_dataset):
    normalised = tiny_downscaling_dataset.normalized_features()
    assert normalised.shape == tiny_downscaling_dataset.features.shape
    assert np.all(np.isfinite(normalised))


def test_normalization_works_on_a_single_sample(tiny_downscaling_dataset):
    single = tiny_downscaling_dataset.normalization.apply(tiny_downscaling_dataset.features[0])
    assert single.shape == tiny_downscaling_dataset.features[0].shape


def test_normalization_rejects_a_channel_mismatch(tiny_downscaling_dataset):
    with pytest.raises(ValueError, match="channels"):
        tiny_downscaling_dataset.normalization.apply(np.zeros((3, 5, 5), dtype="float32"))
    with pytest.raises(ValueError, match="3-D|4-D"):
        tiny_downscaling_dataset.normalization.apply(np.zeros((5,), dtype="float32"))


def test_normalization_rejects_non_positive_std():
    with pytest.raises(ValueError, match="std"):
        NormalizationStats(mean=(0.0, 1.0), std=(0.0, 1.0))
    with pytest.raises(ValueError, match="same length"):
        NormalizationStats(mean=(0.0,), std=(1.0, 2.0))


def test_fit_normalization_validates_its_indices(tiny_downscaling_dataset):
    with pytest.raises(ValueError, match="empty index set"):
        fit_normalization(tiny_downscaling_dataset.features, [])
    with pytest.raises(ValueError, match="out of range"):
        fit_normalization(tiny_downscaling_dataset.features, [9999])
    with pytest.raises(ValueError, match=r"\[samples, C, H, W\]"):
        fit_normalization(tiny_downscaling_dataset.features[0], [0])


def test_normalization_round_trips_through_a_dict(tiny_downscaling_dataset):
    payload = tiny_downscaling_dataset.normalization.as_dict()
    restored = NormalizationStats.from_dict(payload)
    assert restored.mean == tiny_downscaling_dataset.normalization.mean
    assert restored.std == tiny_downscaling_dataset.normalization.std
    assert restored.source == tiny_downscaling_dataset.normalization.source


def test_block_accessor_returns_normalised_features(tiny_downscaling_dataset):
    block = tiny_downscaling_dataset.block("train")
    assert block.shape[0] == len(tiny_downscaling_dataset.split.train)


# --- misc -------------------------------------------------------------------


def test_bilinear_baseline_preserves_the_field_mean():
    baseline = bilinear_baseline(np.ones((9, 9)) * 3.0, (21, 21))
    assert baseline.shape == (21, 21)
    assert baseline.mean() == pytest.approx(3.0)


def test_dataset_summary_is_json_serialisable(tiny_downscaling_dataset):
    summary = tiny_downscaling_dataset.summary()
    assert summary["variable"] == "precipitation"
    assert summary["split"]["train"] > 0
    assert "SYNTHETIC" in summary["reference_disclaimer"]


def test_load_coarse_dataset_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_coarse_dataset(tmp_path / "absent.nc")
