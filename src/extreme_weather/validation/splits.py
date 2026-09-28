"""Strict chronological splitting of weather time series (SIH26078, Phase 7).

Weather data is a time series. Shuffling it before splitting produces a model
that has seen next week's rain while learning about today's, and every number it
then reports is meaningless. This module makes that impossible to do by
accident:

* splits are **chronological only** - there is no shuffle option and no random
  seed, because a random split of a time series is always wrong;
* the boundaries are ``max(train) < min(validation) < min(test)``, with no
  overlap and no sample in two splits;
* the *climatology* is built from the training slice alone, so a test-period
  value can never influence a threshold or a score.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import xarray as xr

__all__ = [
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VALIDATION_RATIO",
    "MIN_SPLIT_STEPS",
    "SplitError",
    "SplitSummary",
    "TemporalSplit",
    "assert_no_leakage",
    "split_dataset",
    "split_timestamps",
]


#: Fraction of the period assigned to training when the caller states none.
DEFAULT_TRAIN_RATIO: float = 0.6

#: Fraction assigned to validation; the test split takes the remainder.
DEFAULT_VALIDATION_RATIO: float = 0.2

#: Every split needs at least this many steps to be measurable at all.
MIN_SPLIT_STEPS: int = 3


class SplitError(ValueError):
    """Raised when a split cannot be produced from the given timestamps."""


@dataclass(frozen=True)
class TemporalSplit:
    """The timestamps belonging to one split.

    Attributes:
        name: ``"train"``, ``"validation"`` or ``"test"``.
        indices: Positional indices into the source time axis.
        timestamps: The ``datetime64`` values themselves.
    """

    name: str
    indices: np.ndarray
    timestamps: np.ndarray

    def __len__(self) -> int:
        """Number of timestamps in the split."""
        return int(self.indices.size)

    @property
    def start(self) -> str | None:
        """ISO 8601 start of the split, or ``None`` when it is empty."""
        return _iso(self.timestamps[0]) if self.timestamps.size else None

    @property
    def end(self) -> str | None:
        """ISO 8601 end of the split, or ``None`` when it is empty."""
        return _iso(self.timestamps[-1]) if self.timestamps.size else None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary."""
        return {
            "name": self.name,
            "size": len(self),
            "start": self.start,
            "end": self.end,
        }


@dataclass(frozen=True)
class SplitSummary:
    """All three splits plus the ratios that produced them."""

    train: TemporalSplit
    validation: TemporalSplit
    test: TemporalSplit
    train_ratio: float = DEFAULT_TRAIN_RATIO
    validation_ratio: float = DEFAULT_VALIDATION_RATIO

    def as_list(self) -> list[TemporalSplit]:
        """Return the splits in chronological order."""
        return [self.train, self.validation, self.test]

    def get(self, name: str) -> TemporalSplit:
        """Return one split by name.

        Raises:
            SplitError: If the name is not one of the three splits.
        """
        for split in self.as_list():
            if split.name == name:
                return split
        raise SplitError(
            f"Unknown split '{name}'. Use 'train', 'validation' or 'test'."
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary of every split."""
        return {
            "train_ratio": self.train_ratio,
            "validation_ratio": self.validation_ratio,
            "total": len(self.train) + len(self.validation) + len(self.test),
            "splits": [split.as_dict() for split in self.as_list()],
        }


def _iso(value: Any) -> str:
    """Format a ``datetime64`` scalar as a compact ISO 8601 string."""
    return str(np.datetime_as_string(np.asarray(value), unit="s"))


def _validate_ratios(train_ratio: float, validation_ratio: float) -> tuple[float, float]:
    """Check the requested ratios and return them as floats.

    Raises:
        SplitError: If a ratio is outside ``(0, 1)``, or the two leave no test
            period.
    """
    train = float(train_ratio)
    validation = float(validation_ratio)
    if not 0.0 < train < 1.0:
        raise SplitError(f"train_ratio must be in (0, 1); got {train_ratio}.")
    if not 0.0 < validation < 1.0:
        raise SplitError(f"validation_ratio must be in (0, 1); got {validation_ratio}.")
    if train + validation >= 1.0:
        raise SplitError(
            "train_ratio + validation_ratio must leave a test period; got "
            f"{train} + {validation} = {train + validation}."
        )
    return train, validation


def _sorted_unique(times: np.ndarray) -> np.ndarray:
    """Return the timestamps sorted, rejecting duplicates.

    Raises:
        SplitError: If the axis is not 1-D, is empty, is not datetime-like, or
            contains duplicates. Duplicates are rejected rather than collapsed
            because a duplicated timestamp means the *source* is ambiguous, and
            silently picking one would hide that.
    """
    values = np.asarray(times)
    if values.ndim != 1:
        raise SplitError(f"time must be 1-D; got shape {values.shape}.")
    if values.size == 0:
        raise SplitError("time is empty; there is nothing to split.")
    if not np.issubdtype(values.dtype, np.datetime64):
        raise SplitError(f"time must be datetime64; got dtype {values.dtype}.")
    if np.unique(values).size != values.size:
        duplicates = int(values.size - np.unique(values).size)
        raise SplitError(
            f"time contains {duplicates} duplicate timestamp(s). Resolve them in "
            "the source data before splitting."
        )
    return np.sort(values)


def split_timestamps(
    times: Sequence[np.datetime64] | np.ndarray,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
) -> SplitSummary:
    """Split timestamps into train / validation / test, chronologically.

    The cut points are computed from the sorted axis and applied so every
    timestamp lands in exactly one split:

    * ``train`` takes the first ``round(train_ratio * n)`` steps,
    * ``validation`` the next ``round(validation_ratio * n)``,
    * ``test`` everything after that.

    Returned indices are positions in the **original** array, so the caller can
    slice a dataset without re-sorting it.

    Args:
        times: A 1-D array of ``datetime64`` values.
        train_ratio: Fraction of steps for training, in ``(0, 1)``.
        validation_ratio: Fraction of steps for validation, in ``(0, 1)``.

    Returns:
        A :class:`SplitSummary` whose three splits are strictly ordered and
        disjoint.

    Raises:
        SplitError: If the ratios are invalid, the axis is unusable, or the
            series is too short to give every split
            :data:`MIN_SPLIT_STEPS` samples.
    """
    train_fraction, validation_fraction = _validate_ratios(
        train_ratio, validation_ratio
    )
    original = np.asarray(times)
    ordered = _sorted_unique(original)

    total = int(ordered.size)
    if total < MIN_SPLIT_STEPS * 3:
        raise SplitError(
            f"Need at least {MIN_SPLIT_STEPS * 3} timestamps to give each of the "
            f"three splits {MIN_SPLIT_STEPS} samples; got {total}."
        )

    train_end = int(round(train_fraction * total))
    validation_end = train_end + int(round(validation_fraction * total))
    # Never let rounding eat a split: push the boundaries apart if needed.
    train_end = min(max(train_end, MIN_SPLIT_STEPS), total - 2 * MIN_SPLIT_STEPS)
    validation_end = min(
        max(validation_end, train_end + MIN_SPLIT_STEPS), total - MIN_SPLIT_STEPS
    )

    bounds = {
        "train": ordered[:train_end],
        "validation": ordered[train_end:validation_end],
        "test": ordered[validation_end:],
    }

    splits: list[TemporalSplit] = []
    for name in ("train", "validation", "test"):
        values = bounds[name]
        # Map back to positions in the original array, preserving its order.
        positions = np.flatnonzero(np.isin(original, values))
        splits.append(
            TemporalSplit(
                name=name, indices=positions, timestamps=original[positions]
            )
        )

    summary = SplitSummary(
        train=splits[0],
        validation=splits[1],
        test=splits[2],
        train_ratio=train_fraction,
        validation_ratio=validation_fraction,
    )
    assert_no_leakage(summary)
    return summary



def assert_no_leakage(summary: SplitSummary) -> None:
    """Verify the three splits are ordered, disjoint and cover the input.

    This is the invariant the whole of Phase 7 rests on, so it is checked
    explicitly rather than assumed - including by the tests.

    Raises:
        SplitError: If any timestamp appears in two splits, or the splits are
            not in increasing chronological order.
    """
    for split in summary.as_list():
        if len(split) == 0:
            raise SplitError(f"The '{split.name}' split is empty.")

    train_end = np.max(summary.train.timestamps)
    validation_start = np.min(summary.validation.timestamps)
    validation_end = np.max(summary.validation.timestamps)
    test_start = np.min(summary.test.timestamps)

    if not train_end < validation_start:
        raise SplitError(
            "Temporal leakage: max(train) is not before min(validation) "
            f"({_iso(train_end)} >= {_iso(validation_start)})."
        )
    if not validation_end < test_start:
        raise SplitError(
            "Temporal leakage: max(validation) is not before min(test) "
            f"({_iso(validation_end)} >= {_iso(test_start)})."
        )

    total = sum(len(split) for split in summary.as_list())
    unique = np.unique(np.concatenate([split.timestamps for split in summary.as_list()]))
    if unique.size != total:
        raise SplitError(
            f"Splits overlap: {total} assignments but only {unique.size} unique "
            "timestamps."
        )


def split_dataset(
    dataset: xr.Dataset,
    *,
    train_ratio: float = DEFAULT_TRAIN_RATIO,
    validation_ratio: float = DEFAULT_VALIDATION_RATIO,
) -> dict[str, xr.Dataset]:
    """Split a gridded dataset into train / validation / test datasets.

    Only the ``time`` axis is subset, so every split keeps the same spatial grid
    and variables. The climatology for the whole experiment is then built from
    ``result["train"]`` alone.

    Args:
        dataset: A dataset with a ``time`` dimension.
        train_ratio: Fraction of steps for training.
        validation_ratio: Fraction of steps for validation.

    Returns:
        ``{"train": ..., "validation": ..., "test": ...}``.

    Raises:
        SplitError: If the dataset has no 1-D ``time`` axis, or the split is
            impossible.
    """
    if not isinstance(dataset, xr.Dataset):
        raise SplitError(f"Expected an xarray.Dataset, got {type(dataset).__name__}.")
    if "time" not in dataset.coords or dataset.coords["time"].ndim != 1:
        raise SplitError("Dataset must carry a 1-D 'time' coordinate to be split.")

    summary = split_timestamps(
        dataset.coords["time"].values,
        train_ratio=train_ratio,
        validation_ratio=validation_ratio,
    )
    return {split.name: dataset.isel(time=split.indices) for split in summary.as_list()}

