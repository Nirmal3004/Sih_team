"""The Phase 9 evaluation dataset (SIH26078, section 2).

This module builds the one artifact everything else in Phase 9 is measured on,
and its main job is to make leakage **structurally difficult** rather than
merely discouraged.

The rules
---------
1. **Chronological, never shuffled.** Train precedes validation precedes test.
2. **No overlap.** The split is asserted, not assumed.
3. **Learned statistics come from train only.** Feature scaling and any
   normalisation are fitted on the training block and then *applied* unchanged
   to validation and test.
4. **Deterministic identity.** The dataset id is a hash of the content, so the
   same inputs always name the same dataset.

The leakage guards
------------------
:meth:`EvaluationDataset.assert_no_leakage` re-derives the ordering from the
timestamps themselves and raises if it does not hold. It is called during
construction, so a dataset that would leak cannot be built in the first place -
rather than being caught later by a test that someone might delete.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from extreme_weather.evaluation.models import Provenance, utc_now

__all__ = [
    "EvaluationDataset",
    "EvaluationSplit",
    "LeakageError",
    "assert_chronological",
    "build_evaluation_dataset",
    "fit_train_only_scaler",
]

UTC = timezone.utc


class LeakageError(ValueError):
    """Raised when a split or scaling would leak information.

    A dedicated type so the API can map it to 422 and the CLI can explain it,
    rather than letting a leakage problem surface as a confusing numeric error.
    """


@dataclass(frozen=True)
class EvaluationSplit:
    """Chronological train/validation/test boundaries.

    Attributes:
        train: Indices into the time axis for training.
        validation: Indices for validation.
        test: Indices for testing.
        train_end: Last training timestamp, inclusive.
        validation_start: First validation timestamp.
        validation_end: Last validation timestamp, inclusive.
        test_start: First test timestamp.
        ratios: The fractions requested.
    """

    train: tuple[int, ...]
    validation: tuple[int, ...]
    test: tuple[int, ...]
    train_end: str
    validation_start: str | None
    validation_end: str | None
    test_start: str
    ratios: dict[str, float] = field(default_factory=dict)

    def sizes(self) -> dict[str, int]:
        """Return the number of time steps in each block."""
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
            "total": len(self.train) + len(self.validation) + len(self.test),
        }

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record of the split."""
        payload = {
            "train_indices": list(self.train),
            "validation_indices": list(self.validation),
            "test_indices": list(self.test),
            "sizes": self.sizes(),
            "ratios": dict(self.ratios),
            "boundaries": {
                "train_end": self.train_end,
                "validation_start": self.validation_start,
                "validation_end": self.validation_end,
                "test_start": self.test_start,
            },
            "ordering": "chronological",
            "shuffled": False,
        }
        return payload


def _iso(value: np.datetime64) -> str:
    """Return an ISO string for a numpy timestamp."""
    return str(np.datetime_as_string(value, unit="s"))


def assert_chronological(
    times: Sequence[np.datetime64], split: EvaluationSplit
) -> None:
    """Verify the split is chronological and disjoint, or raise.

    Args:
        times: The full, sorted time axis.
        split: The split to verify.

    Raises:
        LeakageError: If the blocks are out of order, overlap, or the axis is
            not sorted. Any of those would leak future information.
    """
    axis = np.asarray(times)
    if axis.size and np.any(np.diff(axis.astype("datetime64[ns]").astype("int64")) <= 0):
        raise LeakageError(
            "The time axis is not strictly increasing, so a chronological split "
            "cannot be verified. Sort and de-duplicate the axis first."
        )

    train = np.asarray(split.train, dtype="int64")
    validation = np.asarray(split.validation, dtype="int64")
    test = np.asarray(split.test, dtype="int64")

    if train.size == 0 or test.size == 0:
        raise LeakageError(
            "A chronological evaluation needs a non-empty training block and a "
            f"non-empty test block (got train={train.size}, test={test.size})."
        )

    if train.size and validation.size and train.max() >= validation.min():
        raise LeakageError(
            f"Training index {int(train.max())} is not before validation index "
            f"{int(validation.min())}: the test period would leak into training."
        )
    if validation.size and test.size and validation.max() >= test.min():
        raise LeakageError(
            f"Validation index {int(validation.max())} is not before test index "
            f"{int(test.min())}: the test period would leak into model selection."
        )

    for name, block in (("train", train), ("validation", validation), ("test", test)):
        if block.size == 0:
            continue
        if block.min() < 0 or block.max() >= axis.size:
            raise LeakageError(
                f"The {name} block references index {int(block.max())}, outside "
                f"the {axis.size}-step time axis."
            )

    # Overlap check on the index sets, independent of ordering.
    if train.size and validation.size and set(train.tolist()) & set(validation.tolist()):
        raise LeakageError("Training and validation share at least one time step.")
    if train.size and test.size and set(train.tolist()) & set(test.tolist()):
        raise LeakageError("Training and test share at least one time step.")
    if validation.size and test.size and set(validation.tolist()) & set(test.tolist()):
        raise LeakageError("Validation and test share at least one time step.")


def _chronological_indices(
    count: int, train_ratio: float, validation_ratio: float
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Split ``range(count)`` into consecutive blocks.

    The cut points are computed once and applied to all three blocks, so no
    step can be dropped or double-counted at a boundary.
    """
    if count < 2:
        raise LeakageError(
            f"A chronological evaluation needs at least 2 time steps, got {count}."
        )
    if not 0.0 < train_ratio < 1.0:
        raise LeakageError("train_ratio must be strictly between 0 and 1.")
    if not 0.0 <= validation_ratio < 1.0:
        raise LeakageError("validation_ratio must be in [0, 1).")
    if train_ratio + validation_ratio >= 1.0:
        raise LeakageError(
            "train_ratio + validation_ratio must leave room for a test block."
        )

    train_end = int(np.floor(count * train_ratio))
    validation_end = int(np.floor(count * (train_ratio + validation_ratio)))
    train_end = max(1, min(train_end, count - 1))
    validation_end = max(train_end, min(validation_end, count))

    return (
        tuple(range(0, train_end)),
        tuple(range(train_end, validation_end)),
        tuple(range(validation_end, count)),
    )


@dataclass
class TrainOnlyScaler:
    """Standardisation constants fitted on the training block only.

    Attributes:
        mean: Per-variable training mean.
        scale: Per-variable training standard deviation.
        variables: Variable names, in order.
        fitted_on: The number of training steps used.
    """

    mean: dict[str, float] = field(default_factory=dict)
    scale: dict[str, float] = field(default_factory=dict)
    variables: tuple[str, ...] = ()
    fitted_on: int = 0

    def transform(self, dataset: xr.Dataset) -> xr.Dataset:
        """Apply the training constants to a dataset, unchanged."""
        result = dataset
        for name in self.variables:
            if name not in result.data_vars:
                continue
            values = np.asarray(result[name].values, dtype="float64")
            mean = self.mean.get(name, 0.0)
            scale = self.scale.get(name, 1.0) or 1.0
            result[name] = xr.DataArray(
                (values - mean) / scale,
                dims=result[name].dims,
                attrs={**result[name].attrs, "standardised_with": "train-only"},
            )
        return result

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable record."""
        return {
            "variables": list(self.variables),
            "mean": {k: float(v) for k, v in self.mean.items()},
            "scale": {k: float(v) for k, v in self.scale.items()},
            "fitted_on_training_steps": self.fitted_on,
            "fitted_on": "train_only",
        }


def fit_train_only_scaler(dataset: xr.Dataset, train_indices: Sequence[int]) -> TrainOnlyScaler:
    """Fit standardisation constants on the training block alone.

    Fitting on the whole record is the most common way a temporal evaluation
    leaks: the test period's mean and spread inform the transform applied to it.
    This function takes the indices explicitly so that mistake is not
    available.

    Args:
        dataset: The dataset to fit.
        train_indices: Indices into the time axis to fit on.

    Returns:
        A :class:`TrainOnlyScaler` fitted on the training block only.

    Raises:
        LeakageError: If no training indices are supplied.
    """
    indices = list(train_indices)
    if not indices:
        raise LeakageError("A train-only scaler needs at least one training step.")

    mean: dict[str, float] = {}
    scale: dict[str, float] = {}
    for name in sorted(str(v) for v in dataset.data_vars):
        if name not in dataset.data_vars or "time" not in dataset[name].dims:
            continue
        values = np.asarray(dataset[name].isel(time=list(indices)).values, dtype="float64")
        mean[name] = float(np.nanmean(values)) if values.size else 0.0
        deviation = float(np.nanstd(values)) if values.size else 0.0
        # A zero-variance training variable would divide by zero; 1.0 makes the
        # transform an identity for it, which is the honest outcome.
        scale[name] = deviation if deviation > 0.0 else 1.0

    return TrainOnlyScaler(
        mean=mean, scale=scale, variables=tuple(sorted(mean)), fitted_on=len(indices)
    )


@dataclass
class EvaluationDataset:
    """The dataset every Phase 9 measurement is taken on.

    Attributes:
        dataset_id: Content-derived deterministic identifier.
        raw: The full dataset.
        train: Training block.
        validation: Validation block.
        test: Test block.
        split: The boundaries and index sets.
        checksum: SHA-256 of the raw content.
        scaler: The train-only standardisation constants.
        source: Where the data came from.
        source_period: The period covered.
        variables: Variable names.
        grid: Grid dimensions.
        resolution: Nominal resolution.
    """

    dataset_id: str
    raw: xr.Dataset
    train: xr.Dataset
    validation: xr.Dataset
    test: xr.Dataset
    split: EvaluationSplit
    checksum: str
    scaler: TrainOnlyScaler
    source: str
    source_period: dict[str, str]
    variables: tuple[str, ...]
    grid: dict[str, int]
    resolution_deg: float | None = None
    preprocessing_version: str = "phase0-preprocess-v1"
    created_at: datetime = field(default_factory=utc_now)
    origin: str = "HISTORICAL"

    def assert_no_leakage(self) -> None:
        """Re-verify the split, raising :class:`LeakageError` on any problem."""
        assert_chronological(np.asarray(self.raw.coords["time"].values), self.split)

    def provenance(self) -> Provenance:
        """Return a reproducibility record for this dataset."""
        return Provenance(
            dataset_id=self.dataset_id,
            dataset_checksum=self.checksum,
            split_boundaries=self.split.as_dict()["boundaries"],
            configuration={
                "source": self.source,
                "source_period": dict(self.source_period),
                "variables": list(self.variables),
                "grid": dict(self.grid),
                "resolution_deg": self.resolution_deg,
                "preprocessing_version": self.preprocessing_version,
                "origin": self.origin,
                "scaler": self.scaler.as_dict(),
            },
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable manifest (no array data)."""
        return {
            "dataset_id": self.dataset_id,
            "checksum": self.checksum,
            "source": self.source,
            "source_period": dict(self.source_period),
            "origin": self.origin,
            "variables": list(self.variables),
            "grid": dict(self.grid),
            "resolution_deg": self.resolution_deg,
            "preprocessing_version": self.preprocessing_version,
            "split": self.split.as_dict(),
            "scaler": self.scaler.as_dict(),
            "created_at": self.created_at.isoformat(),
            "time_steps": int(self.raw.sizes.get("time", 0)),
        }


def _content_checksum(dataset: xr.Dataset) -> str:
    """Return a stable checksum of a dataset's values and coordinates."""
    from extreme_weather.ingestion.manifest import content_hash

    return content_hash(dataset)


def build_evaluation_dataset(
    dataset: xr.Dataset,
    *,
    dataset_id: str | None = None,
    source: str = "unknown",
    train_ratio: float = 0.6,
    validation_ratio: float = 0.2,
    resolution_deg: float | None = None,
    origin: str = "HISTORICAL",
    preprocessing_version: str = "phase0-preprocess-v1",
) -> EvaluationDataset:
    """Build a reproducible, leak-checked evaluation dataset.

    Args:
        dataset: The dataset to evaluate on. Must have a ``time`` coordinate.
        dataset_id: Explicit identifier; a content hash is used when omitted.
        source: Where the data came from, recorded in the manifest.
        train_ratio: Fraction of the record assigned to training.
        validation_ratio: Fraction assigned to validation. May be ``0``.
        resolution_deg: Nominal grid resolution, for the manifest.
        origin: Provenance class, e.g. ``HISTORICAL`` or ``SYNTHETIC``.
        preprocessing_version: Identifier of the preprocessing applied.

    Returns:
        An :class:`EvaluationDataset` whose split has been asserted.

    Raises:
        LeakageError: If the split is not chronological, overlaps, or is empty.
        ValueError: If the dataset has no time coordinate.
    """
    if "time" not in dataset.coords:
        raise ValueError(
            "An evaluation dataset needs a 'time' coordinate; none was found."
        )

    times = np.asarray(dataset.coords["time"].values)
    if times.size < 2:
        raise LeakageError(
            f"An evaluation needs at least 2 time steps, got {times.size}."
        )

    train_idx, validation_idx, test_idx = _chronological_indices(
        int(times.size), float(train_ratio), float(validation_ratio)
    )
    split = EvaluationSplit(
        train=train_idx,
        validation=validation_idx,
        test=test_idx,
        train_end=_iso(times[train_idx[-1]]),
        validation_start=_iso(times[validation_idx[0]]) if validation_idx else None,
        validation_end=_iso(times[validation_idx[-1]]) if validation_idx else None,
        test_start=_iso(times[test_idx[0]]),
        ratios={
            "train": float(train_ratio),
            "validation": float(validation_ratio),
            "test": round(1.0 - float(train_ratio) - float(validation_ratio), 10),
        },
    )
    # Verified here, not in a test: a leaking dataset must not be constructible.
    assert_chronological(times, split)

    scaler = fit_train_only_scaler(dataset, train_idx)
    checksum = _content_checksum(dataset)
    identifier = dataset_id or f"eval-{checksum[:16]}"

    grid = {
        "time": int(times.size),
        "latitude": int(dataset.sizes.get("latitude", 0)),
        "longitude": int(dataset.sizes.get("longitude", 0)),
    }
    variables = tuple(sorted(str(v) for v in dataset.data_vars))

    result = EvaluationDataset(
        dataset_id=identifier,
        raw=dataset,
        train=dataset.isel(time=list(train_idx)),
        validation=dataset.isel(time=list(validation_idx)),
        test=dataset.isel(time=list(test_idx)),
        split=split,
        checksum=checksum,
        scaler=scaler,
        source=source,
        source_period={
            "start": _iso(times[0]),
            "end": _iso(times[-1]),
        },
        variables=variables,
        grid=grid,
        resolution_deg=resolution_deg,
        preprocessing_version=preprocessing_version,
        origin=origin,
    )
    result.assert_no_leakage()
    return result
