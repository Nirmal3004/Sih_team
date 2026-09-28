"""Tests for Phase 7 temporal splitting and leakage prevention.

The invariant under test is the one the whole phase rests on::

    max(train_time) < min(validation_time) < min(test_time)

and no timestamp in two splits. These tests exist to make a future change that
breaks it fail loudly rather than silently produce meaningless numbers.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from extreme_weather.validation.splits import (
    SplitError,
    assert_no_leakage,
    split_dataset,
    split_timestamps,
)

from tests.validation_fixtures import FULL_STEPS, make_times


def _times(count: int = FULL_STEPS) -> np.ndarray:
    return make_times(count)


# --------------------------------------------------------------------------- #
# The core invariant
# --------------------------------------------------------------------------- #
def test_the_three_splits_are_strictly_ordered() -> None:
    summary = split_timestamps(_times())
    assert np.max(summary.train.timestamps) < np.min(summary.validation.timestamps)
    assert np.max(summary.validation.timestamps) < np.min(summary.test.timestamps)


def test_no_timestamp_appears_in_two_splits() -> None:
    summary = split_timestamps(_times())
    all_times = np.concatenate([s.timestamps for s in summary.as_list()])
    assert np.unique(all_times).size == all_times.size


def test_every_timestamp_is_assigned_exactly_once() -> None:
    times = _times()
    summary = split_timestamps(times)
    assert sum(len(s) for s in summary.as_list()) == len(times)


def test_the_default_ratios_are_respected() -> None:
    summary = split_timestamps(_times(), train_ratio=0.6, validation_ratio=0.2)
    assert (len(summary.train), len(summary.validation), len(summary.test)) == (60, 20, 20)


def test_custom_ratios_are_respected() -> None:
    summary = split_timestamps(_times(), train_ratio=0.5, validation_ratio=0.25)
    assert (len(summary.train), len(summary.validation), len(summary.test)) == (50, 25, 25)


def test_the_split_is_deterministic() -> None:
    """The same input must always produce the same split, or nothing is reproducible."""
    first = split_timestamps(_times())
    second = split_timestamps(_times())
    assert [s.indices.tolist() for s in first.as_list()] == [
        s.indices.tolist() for s in second.as_list()
    ]


def test_an_unsorted_input_is_split_by_value_not_position() -> None:
    """Indices point at the original array, so a shuffled axis is handled."""
    shuffled = _times(20)[::-1].copy()
    summary = split_timestamps(shuffled, train_ratio=0.5, validation_ratio=0.25)
    assert np.max(summary.train.timestamps) < np.min(summary.validation.timestamps)
    assert all(0 <= i < len(shuffled) for i in summary.train.indices)


# --------------------------------------------------------------------------- #
# Rejections - each of these would corrupt a validation run
# --------------------------------------------------------------------------- #
def test_duplicate_timestamps_are_rejected() -> None:
    times = _times(20)
    duplicated = np.array([times[0], times[0], *times[1:19]], dtype="datetime64[ns]")
    with pytest.raises(SplitError, match="duplicate timestamp"):
        split_timestamps(duplicated)


def test_an_empty_axis_is_rejected() -> None:
    with pytest.raises(SplitError, match="empty"):
        split_timestamps(np.array([], dtype="datetime64[ns]"))


def test_a_non_datetime_axis_is_rejected() -> None:
    with pytest.raises(SplitError, match="datetime64"):
        split_timestamps(np.arange(20, dtype="float64"))


def test_a_too_short_series_is_rejected() -> None:
    with pytest.raises(SplitError, match="at least"):
        split_timestamps(_times(6))


def test_ratios_that_leave_no_test_period_are_rejected() -> None:
    with pytest.raises(SplitError, match="test period"):
        split_timestamps(_times(), train_ratio=0.8, validation_ratio=0.3)


def test_out_of_range_ratios_are_rejected() -> None:
    with pytest.raises(SplitError, match="train_ratio"):
        split_timestamps(_times(), train_ratio=0.0)
    with pytest.raises(SplitError, match="validation_ratio"):
        split_timestamps(_times(), validation_ratio=1.0)


def test_a_multi_dimensional_axis_is_rejected() -> None:
    with pytest.raises(SplitError, match="1-D"):
        split_timestamps(np.zeros((3, 4), dtype="datetime64[ns]"))


# --------------------------------------------------------------------------- #
# Leakage detection itself
# --------------------------------------------------------------------------- #
def test_assert_no_leakage_rejects_overlapping_splits() -> None:
    """The guard is not decorative: it must reject a hand-built leak.

    Here the validation block starts *before* training ends, which is exactly the
    mistake an accidentally shuffled split would make.
    """
    from extreme_weather.validation.splits import SplitSummary, TemporalSplit

    times = _times(10)
    summary = SplitSummary(
        # Training runs to index 6 ...
        train=TemporalSplit("train", np.arange(7), times[0:7]),
        # ... but validation starts at index 5, inside it.
        validation=TemporalSplit("validation", np.array([5, 6]), times[5:7]),
        test=TemporalSplit("test", np.array([8, 9]), times[8:10]),
    )
    with pytest.raises(SplitError, match="leakage"):
        assert_no_leakage(summary)


def test_assert_no_leakage_rejects_an_empty_split() -> None:
    from extreme_weather.validation.splits import SplitSummary, TemporalSplit

    times = _times(10)
    summary = SplitSummary(
        train=TemporalSplit("train", np.array([]), times[:0]),
        validation=TemporalSplit("validation", np.array([1, 2]), times[1:3]),
        test=TemporalSplit("test", np.array([3, 4]), times[3:5]),
    )
    with pytest.raises(SplitError, match="empty"):
        assert_no_leakage(summary)


def test_a_valid_split_passes_the_guard() -> None:
    assert_no_leakage(split_timestamps(_times()))


# --------------------------------------------------------------------------- #
# Dataset splitting
# --------------------------------------------------------------------------- #
def test_split_dataset_keeps_every_block_the_same_spatial_shape(
    validation_dataset: xr.Dataset,
) -> None:
    blocks = split_dataset(validation_dataset)
    assert set(blocks) == {"train", "validation", "test"}
    for name, block in blocks.items():
        assert block.sizes["latitude"] == validation_dataset.sizes["latitude"], name
        assert block.sizes["longitude"] == validation_dataset.sizes["longitude"], name
    total = sum(int(block.sizes["time"]) for block in blocks.values())
    assert total == int(validation_dataset.sizes["time"])


def test_split_dataset_blocks_are_disjoint_and_ordered(
    validation_dataset: xr.Dataset,
) -> None:
    blocks = split_dataset(validation_dataset)
    train_end = np.max(blocks["train"].coords["time"].values)
    validation_start = np.min(blocks["validation"].coords["time"].values)
    validation_end = np.max(blocks["validation"].coords["time"].values)
    test_start = np.min(blocks["test"].coords["time"].values)
    assert train_end < validation_start < validation_end < test_start


def test_split_dataset_preserves_the_variables(
    validation_dataset: xr.Dataset,
) -> None:
    blocks = split_dataset(validation_dataset)
    assert set(blocks["train"].data_vars) == set(validation_dataset.data_vars)


def test_a_dataset_without_a_time_axis_is_rejected() -> None:
    dataset = xr.Dataset({"x": ("latitude", [1.0, 2.0])})
    with pytest.raises(SplitError, match="1-D 'time'"):
        split_dataset(dataset)


def test_a_non_dataset_is_rejected() -> None:
    with pytest.raises(SplitError, match="xarray.Dataset"):
        split_dataset([1, 2, 3])  # type: ignore[arg-type]


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def test_the_summary_serialises_every_split() -> None:
    payload = split_timestamps(_times()).as_dict()
    assert [entry["name"] for entry in payload["splits"]] == [
        "train",
        "validation",
        "test",
    ]
    for entry in payload["splits"]:
        assert entry["start"] and entry["end"]
        assert entry["size"] > 0


def test_a_split_can_be_fetched_by_name() -> None:
    summary = split_timestamps(_times())
    assert summary.get("test") is summary.test
    with pytest.raises(SplitError, match="Unknown split"):
        summary.get("holdout")

