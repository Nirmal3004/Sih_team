"""Evaluation-dataset tests (SIH26078, Phase 9, section 2).

The required coverage: chronological split, no overlap, no future leakage,
provenance and a deterministic dataset id.

The leakage tests are written to fail loudly if the guards are weakened: each
one constructs a split that *must* be rejected and asserts the rejection.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import xarray as xr

from extreme_weather.evaluation.dataset import (
    EvaluationSplit,
    LeakageError,
    assert_chronological,
    build_evaluation_dataset,
    fit_train_only_scaler,
)

UTC = timezone.utc

START = datetime(2024, 6, 1, tzinfo=UTC)


def make_record(steps: int = 100, seed: int = 5) -> xr.Dataset:
    """Return a small, valid, deterministic weather record."""
    stamps = np.array(
        [np.datetime64(START.replace(tzinfo=None)) + np.timedelta64(h, "h") for h in range(steps)],
        dtype="datetime64[ns]",
    )
    generator = np.random.default_rng(seed)
    return xr.Dataset(
        {
            "precipitation": (("time", "latitude", "longitude"), generator.gamma(2.0, 3.0, (steps, 3, 3))),
            "temperature": (("time", "latitude", "longitude"), 300.0 + generator.normal(0, 1, (steps, 3, 3))),
        },
        coords={
            "time": stamps,
            "latitude": [12.0, 12.5, 13.0],
            "longitude": [76.0, 76.5, 77.0],
        },
    )


# --------------------------------------------------------------------------- #
# Chronological split
# --------------------------------------------------------------------------- #
def test_split_is_chronological() -> None:
    """Train must precede validation, which must precede test."""
    dataset = build_evaluation_dataset(make_record(100))

    assert dataset.split.train[-1] < dataset.split.validation[0]
    assert dataset.split.validation[-1] < dataset.split.test[0]


def test_split_blocks_do_not_overlap() -> None:
    """No time step may appear in two blocks."""
    dataset = build_evaluation_dataset(make_record(100))

    train, validation, test = (
        set(dataset.split.train),
        set(dataset.split.validation),
        set(dataset.split.test),
    )
    assert not (train & validation)
    assert not (train & test)
    assert not (validation & test)


def test_split_covers_every_step_exactly_once() -> None:
    """No step may be dropped at a boundary."""
    dataset = build_evaluation_dataset(make_record(100))
    sizes = dataset.split.sizes()

    assert sizes["total"] == 100
    assert sizes["train"] + sizes["validation"] + sizes["test"] == 100


def test_split_is_not_shuffled() -> None:
    """The manifest must state the ordering, so a reader can verify it."""
    dataset = build_evaluation_dataset(make_record(100))
    payload = dataset.split.as_dict()

    assert payload["ordering"] == "chronological"
    assert payload["shuffled"] is False
    assert list(payload["train_indices"]) == sorted(payload["train_indices"])


def test_boundaries_are_recorded() -> None:
    """The report must show where each block starts and ends."""
    dataset = build_evaluation_dataset(make_record(100))

    assert dataset.split.train_end < str(dataset.split.validation_start)
    assert dataset.split.validation_end < dataset.split.test_start


# --------------------------------------------------------------------------- #
# Leakage guards
# --------------------------------------------------------------------------- #
def test_overlapping_blocks_are_rejected() -> None:
    """A step in both train and test is leakage and must be refused.

    Two independent guards cover this: the ordering check and the explicit
    set-overlap check. For chronological blocks the ordering check normally
    fires first, so the test asserts the requirement - the split is refused as
    a leak - rather than pinning which guard caught it.
    """
    times = np.asarray(make_record(10).coords["time"].values)
    bad = EvaluationSplit(
        train=(0, 1, 2, 3, 4, 5),
        validation=(6, 7),
        test=(5, 8, 9),  # index 5 is also in train
        train_end="x",
        validation_start="x",
        validation_end="x",
        test_start="x",
    )

    with pytest.raises(LeakageError) as caught:
        assert_chronological(times, bad)
    message = str(caught.value)
    assert "not before" in message or "share at least one time step" in message


def test_the_overlap_guard_catches_an_overlap_that_ordering_misses() -> None:
    """The explicit overlap check is exercised directly, in isolation.

    Constructing a chronological split that overlaps is impossible - ordering
    implies disjointness - so the overlap guard is verified against a
    degenerate empty validation block, where ordering has nothing to compare.
    """
    times = np.asarray(make_record(10).coords["time"].values)
    overlapping = EvaluationSplit(
        train=(0, 1, 2, 3),
        validation=(),
        test=(3, 4, 5),  # index 3 appears in both train and test
        train_end="x",
        validation_start=None,
        validation_end=None,
        test_start="x",
    )

    with pytest.raises(LeakageError, match="share at least one time step"):
        assert_chronological(times, overlapping)


def test_inverted_ordering_is_rejected() -> None:
    """Test data before training data is leakage and must be refused."""
    times = np.asarray(make_record(10).coords["time"].values)
    bad = EvaluationSplit(
        train=(5, 6, 7),
        validation=(8,),
        test=(0, 1, 2),
        train_end="x",
        validation_start="x",
        validation_end="x",
        test_start="x",
    )

    with pytest.raises(LeakageError, match="not before"):
        assert_chronological(times, bad)


def test_empty_test_block_is_rejected() -> None:
    """An evaluation needs something to score."""
    times = np.asarray(make_record(10).coords["time"].values)
    bad = EvaluationSplit(
        train=(0, 1, 2, 3),
        validation=(4, 5),
        test=(),
        train_end="x",
        validation_start="x",
        validation_end="x",
        test_start="x",
    )

    with pytest.raises(LeakageError, match="non-empty"):
        assert_chronological(times, bad)


def test_unsorted_time_axis_is_rejected() -> None:
    """A scrambled axis makes chronological ordering unverifiable."""
    stamps = np.asarray(make_record(6).coords["time"].values).copy()
    stamps[[1, 2]] = stamps[[2, 1]]
    split = EvaluationSplit(
        train=(0, 1),
        validation=(2,),
        test=(3, 4, 5),
        train_end="x",
        validation_start="x",
        validation_end="x",
        test_start="x",
    )

    with pytest.raises(LeakageError, match="not strictly increasing"):
        assert_chronological(stamps, split)


def test_out_of_range_index_is_rejected() -> None:
    """An index beyond the axis is a bug, not a split."""
    times = np.asarray(make_record(6).coords["time"].values)
    split = EvaluationSplit(
        train=(0, 1),
        validation=(2,),
        test=(3, 99),
        train_end="x",
        validation_start="x",
        validation_end="x",
        test_start="x",
    )

    with pytest.raises(LeakageError, match="outside"):
        assert_chronological(times, split)


def test_constructor_verifies_leakage_eagerly() -> None:
    """A leaking dataset must not be constructible in the first place."""
    dataset = build_evaluation_dataset(make_record(100))
    dataset.assert_no_leakage()  # must not raise

    # Corrupt the split behind the guard's back, then re-verify.
    object.__setattr__(dataset.split, "test", (0, 1, 2))
    with pytest.raises(LeakageError):
        dataset.assert_no_leakage()


def test_ratios_must_leave_room_for_a_test_block() -> None:
    """A split that allocates 100% to train and validation is refused."""
    with pytest.raises(LeakageError, match="room for a test block"):
        build_evaluation_dataset(make_record(50), train_ratio=0.7, validation_ratio=0.3)


def test_a_single_step_record_is_refused() -> None:
    """One step cannot be split chronologically."""
    with pytest.raises(LeakageError, match="at least 2 time steps"):
        build_evaluation_dataset(make_record(1))


# --------------------------------------------------------------------------- #
# Train-only statistics
# --------------------------------------------------------------------------- #
def test_scaler_is_fitted_on_training_data_only() -> None:
    """Fitting on the whole record is the classic temporal leak."""
    record = make_record(100)
    dataset = build_evaluation_dataset(record)

    expected = float(
        np.nanmean(
            np.asarray(record["precipitation"].isel(time=list(dataset.split.train)).values)
        )
    )
    assert dataset.scaler.mean["precipitation"] == pytest.approx(expected)
    assert dataset.scaler.fitted_on == len(dataset.split.train)


def test_scaler_ignores_the_test_block() -> None:
    """Changing only the test period must not move the fitted constants."""
    first = build_evaluation_dataset(make_record(100))
    altered = make_record(100)
    altered["precipitation"][-1] = altered["precipitation"][-1] + 1000.0
    second = build_evaluation_dataset(altered)

    assert first.scaler.mean["precipitation"] == pytest.approx(second.scaler.mean["precipitation"])


def test_scaler_declares_its_provenance() -> None:
    """The scaler must state that it was fitted on training data only."""
    dataset = build_evaluation_dataset(make_record(100))
    payload = dataset.scaler.as_dict()

    assert payload["fitted_on"] == "train_only"
    assert payload["fitted_on_training_steps"] == len(dataset.split.train)


def test_scaler_needs_training_indices() -> None:
    """Fitting on nothing is a configuration error."""
    with pytest.raises(LeakageError, match="at least one training step"):
        fit_train_only_scaler(make_record(20), [])


def test_zero_variance_variable_gets_an_identity_transform() -> None:
    """A constant field must not divide by zero."""
    record = make_record(20)
    record["precipitation"] = xr.DataArray(
        np.full((20, 3, 3), 5.0),
        dims=("time", "latitude", "longitude"),
    )
    scaler = fit_train_only_scaler(record, list(range(12)))

    assert scaler.scale["precipitation"] == 1.0


# --------------------------------------------------------------------------- #
# Provenance and identity
# --------------------------------------------------------------------------- #
def test_dataset_id_is_deterministic() -> None:
    """The same content must always name the same dataset."""
    first = build_evaluation_dataset(make_record(60))
    second = build_evaluation_dataset(make_record(60))

    assert first.dataset_id == second.dataset_id
    assert first.checksum == second.checksum


def test_dataset_id_changes_with_the_content() -> None:
    """Different data must not collide under the same id."""
    first = build_evaluation_dataset(make_record(60))
    second = build_evaluation_dataset(make_record(60, seed=6))

    assert first.dataset_id != second.dataset_id


def test_manifest_records_everything_needed_to_reproduce() -> None:
    """A reader must be able to rebuild the evaluation from the manifest."""
    dataset = build_evaluation_dataset(make_record(100), source="era5-subset", resolution_deg=0.1)
    manifest = dataset.as_dict()

    assert manifest["dataset_id"]
    assert manifest["checksum"]
    assert manifest["source"] == "era5-subset"
    assert manifest["source_period"]["start"]
    assert manifest["source_period"]["end"]
    assert "precipitation" in manifest["variables"]
    assert manifest["grid"]["time"] == 100
    assert manifest["resolution_deg"] == 0.1
    assert manifest["preprocessing_version"]
    assert manifest["split"]["boundaries"]["train_end"]


def test_provenance_carries_the_split_and_checksum() -> None:
    """Provenance must bind the numbers to the data that produced them."""
    dataset = build_evaluation_dataset(make_record(100))
    provenance = dataset.provenance()

    assert provenance.dataset_checksum == dataset.checksum
    assert provenance.split_boundaries["train_end"] == dataset.split.train_end


def test_missing_time_coordinate_is_refused() -> None:
    """Without a time axis there is nothing to split."""
    record = make_record(20).drop_vars("time")

    with pytest.raises(ValueError, match="time"):
        build_evaluation_dataset(record)


def test_split_ratios_are_honoured() -> None:
    """The requested proportions must actually shape the split."""
    dataset = build_evaluation_dataset(make_record(100), train_ratio=0.5, validation_ratio=0.25)
    sizes = dataset.split.sizes()

    assert sizes["train"] == 50
    assert sizes["validation"] == 25
    assert sizes["test"] == 25
