"""Downscaling dataset, normalisation and the chronological split (Phase 4).

Two properties here are safety-critical, and both are covered by tests.

1. **No temporal leakage.** Samples are split by time, never shuffled: the
   earliest 60 % of frames are training, the next 20 % validation, the latest 20 %
   test. Mixing adjacent frames across the boundary would let the model learn
   from its own future, because consecutive weather frames are nearly identical -
   the leak would be severe and invisible in the metrics.

2. **Normalisation from training data only.** The feature mean and standard
   deviation are computed on the **training block alone**. Using the whole
   dataset would leak test-set statistics into training and bias every reported
   number. :func:`fit_normalization` therefore takes an explicit block of
   indices, and a test asserts the statistics change when the test block is
   included.

Target
------
The target is the **residual** of the high-resolution reference against the
bilinear baseline, not the reference itself:

```text
target = reference - bilinear(coarse)
```

Training on the residual matches the model architecture (which predicts a
residual) and keeps the learning problem small and zero-centred.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.ndimage import zoom

from extreme_weather.downscaling.features import (
    FEATURE_NAMES,
    FeatureConfig,
    build_feature_stack,
)
from extreme_weather.downscaling.reference import (
    REFERENCE_DISCLAIMER,
    ReferenceConfig,
    detect_resolution,
    generate_reference,
)

__all__ = [
    "DownscalingDataset",
    "NormalizationStats",
    "SplitConfig",
    "SplitIndices",
    "bilinear_baseline",
    "build_downscaling_dataset",
    "fit_normalization",
    "load_coarse_dataset",
]
DEFAULT_VARIABLE = "precipitation"


def bilinear_baseline(coarse: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Bilinearly upsample a coarse field to ``target_shape``.

    This is the baseline the learned residual is added to, and the same operator
    the training target is defined against.
    """
    values = np.asarray(coarse, dtype="float64")
    factors = (target_shape[0] / values.shape[0], target_shape[1] / values.shape[1])
    return zoom(values, factors, order=1, mode="nearest")


@dataclass(frozen=True)
class NormalizationStats:
    """Per-channel mean and standard deviation used to standardise inputs.

    Attributes:
        mean: Per-channel mean, computed on the **training block only**.
        std: Per-channel standard deviation, floored at a small positive value so
            a constant channel cannot divide by zero.
        feature_names: Channel names the statistics belong to.
        source: Description of the data the statistics came from, stored in the
            checkpoint for auditability.
    """

    mean: tuple[float, ...]
    std: tuple[float, ...]
    feature_names: tuple[str, ...] = FEATURE_NAMES
    source: str = "training split only"

    def __post_init__(self) -> None:
        if len(self.mean) != len(self.std):
            raise ValueError("mean and std must have the same length.")
        if any(value <= 0.0 for value in self.std):
            raise ValueError(f"std entries must be > 0, got {self.std}.")

    def apply(self, stack: np.ndarray) -> np.ndarray:
        """Standardise a feature array, per channel, on the spatial axes.

        Accepts either ``(channels, lat, lon)`` (one sample) or
        ``(samples, channels, lat, lon)`` (a batch). The channel axis is always
        axis 0 for a single sample and axis 1 for a batch, and the statistics are
        broadcast over whatever remains.

        Raises:
            ValueError: If the channel axis does not match the statistics.
        """
        values = np.asarray(stack, dtype="float32")
        if values.ndim == 3:
            channel_axis = 0
        elif values.ndim == 4:
            channel_axis = 1
        else:
            raise ValueError(
                f"Expected a 3-D (C, H, W) or 4-D (N, C, H, W) stack, got {values.shape}."
            )
        if values.shape[channel_axis] != len(self.mean):
            raise ValueError(
                f"Expected {len(self.mean)} channels, got {values.shape[channel_axis]}."
            )
        mean = np.asarray(self.mean, dtype="float32")
        std = np.asarray(self.std, dtype="float32")
        # Reshape to 1 for every axis except the channel axis, so the statistics
        # broadcast over samples and both spatial dimensions.
        shape = [1] * values.ndim
        shape[channel_axis] = -1
        return ((values - mean.reshape(shape)) / std.reshape(shape)).astype("float32")

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view stored in the checkpoint."""
        return {
            "mean": list(self.mean),
            "std": list(self.std),
            "feature_names": list(self.feature_names),
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NormalizationStats":
        """Rebuild from :meth:`as_dict` output."""
        return cls(
            mean=tuple(float(value) for value in payload["mean"]),
            std=tuple(float(value) for value in payload["std"]),
            feature_names=tuple(payload.get("feature_names", FEATURE_NAMES)),
            source=str(payload.get("source", "unknown")),
        )


def fit_normalization(
    features: np.ndarray,
    indices: Sequence[int],
    *,
    source: str = "training split only",
) -> NormalizationStats:
    """Compute per-channel mean/std over ``indices`` **only**.

    Args:
        features: ``(samples, channels, lat, lon)`` stack of all samples.
        indices: The sample indices the statistics may use - the training block.
        source: Label stored with the statistics.

    Returns:
        A :class:`NormalizationStats`.

    Raises:
        ValueError: If ``indices`` is empty or out of range.
    """
    values = np.asarray(features, dtype="float64")
    if values.ndim != 4:
        raise ValueError(f"features must be [samples, C, H, W]; got {values.shape}.")
    selected = [int(index) for index in indices]
    if not selected:
        raise ValueError("Cannot fit normalization on an empty index set.")
    if min(selected) < 0 or max(selected) >= values.shape[0]:
        raise ValueError(
            f"Normalization indices out of range: min {min(selected)}, max {max(selected)}, "
            f"n_samples {values.shape[0]}."
        )
    block = values[selected]
    mean = block.mean(axis=(0, 2, 3))
    # A constant channel would divide by zero; floor the deviation instead.
    std = np.maximum(block.std(axis=(0, 2, 3)), 1e-6)
    return NormalizationStats(
        mean=tuple(float(value) for value in mean),
        std=tuple(float(value) for value in std),
        feature_names=tuple(FEATURE_NAMES),
        source=source,
    )


@dataclass(frozen=True)
class SplitConfig:
    """Fractions of the time axis assigned to each block.

    Attributes:
        train: Fraction of the *time span* used for training (default 0.6).
        validation: Fraction used for validation (default 0.2).
        test: Fraction used for testing (default 0.2).
    """

    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2

    def __post_init__(self) -> None:
        if any(value < 0.0 for value in (self.train, self.validation, self.test)):
            raise ValueError("Split fractions must be >= 0.")
        total = self.train + self.validation + self.test
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0, got {total}.")
        if self.train <= 0.0:
            raise ValueError("The training fraction must be > 0.")
        if self.test <= 0.0:
            raise ValueError("The test fraction must be > 0.")

    @property
    def train_end(self) -> float:
        """Cumulative fraction marking the end of the training period."""
        return self.train

    @property
    def validation_end(self) -> float:
        """Cumulative fraction marking the end of the validation period."""
        return self.train + self.validation


@dataclass(frozen=True)
class SplitIndices:
    """Contiguous, non-overlapping index blocks ordered in time.

    Attributes:
        train: Earliest sample indices (training).
        validation: The block after training.
        test: The latest sample indices.
    """

    train: list[int] = field(default_factory=list)
    validation: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)

    @property
    def sizes(self) -> dict[str, int]:
        """Sample count per block."""
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def of(self, block: str) -> list[int]:
        """Indices of one block.

        Raises:
            ValueError: If ``block`` is not ``train``/``validation``/``test``.
        """
        if block == "train":
            return self.train
        if block == "validation":
            return self.validation
        if block == "test":
            return self.test
        raise ValueError(f"Unknown split block {block!r}.")

    def is_chronological(self, timestamps: Sequence[Any]) -> bool:
        """True when every block lies entirely before the next one."""
        def last(block: list[int]) -> Any:
            return timestamps[max(block)] if block else None

        def first(block: list[int]) -> Any:
            return timestamps[min(block)] if block else None

        if self.train and self.validation and last(self.train) > first(self.validation):
            return False
        if self.validation and self.test and last(self.validation) > first(self.test):
            return False
        return True


@dataclass
class DownscalingDataset:
    """Everything a training run needs: features, targets, split, statistics.

    Attributes:
        features: ``(samples, channels, lat, lon)`` **un-normalised** features.
        targets: ``(samples, 1, lat_target, lon_target)`` residual targets
            (reference minus bilinear baseline).
        references: ``(samples, lat_target, lon_target)`` the synthetic reference
            itself, needed for evaluation.
        baselines: ``(samples, lat_target, lon_target)`` the bilinear baseline of
            each sample.
        coarse: ``(samples, lat, lon)`` the coarse input fields.
        timestamps: Valid time of each sample, used by the chronological split.
        normalization: Statistics fitted on the **training block only**.
        split: The chronological block assignment.
        input_resolution_deg / output_resolution_deg: Detected from the data.
        input_shape / output_shape: ``(lat, lon)`` grid sizes.
        variable: Name of the downscaled variable.
        metadata: Provenance, including the reference disclaimer.
    """

    features: np.ndarray
    targets: np.ndarray
    references: np.ndarray
    baselines: np.ndarray
    coarse: np.ndarray
    timestamps: list[Any]
    normalization: NormalizationStats
    split: SplitIndices
    input_resolution_deg: float
    output_resolution_deg: float
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    variable: str
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        """Number of time samples in the dataset."""
        return int(self.features.shape[0])

    @property
    def scale_factor(self) -> float:
        """Ratio of input to output resolution (e.g. 2.5 for 0.125 -> 0.05)."""
        return self.input_resolution_deg / self.output_resolution_deg

    @property
    def sizes(self) -> dict[str, int]:
        """Sample count per split block."""
        return self.split.sizes

    def normalized_features(self) -> np.ndarray:
        """Return the features standardised with the training statistics."""
        return self.normalization.apply(self.features)

    def block(self, name: str) -> np.ndarray:
        """Return the normalised features of one split block."""
        return self.normalized_features()[self.split.of(name)]

    def summary(self) -> dict[str, Any]:
        """JSON-serialisable summary (no arrays), printed by the CLI and API."""
        return {
            "variable": self.variable,
            "sample_count": self.sample_count,
            "input_resolution_deg": self.input_resolution_deg,
            "output_resolution_deg": self.output_resolution_deg,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "scale_factor": self.scale_factor,
            "feature_count": int(self.features.shape[1]),
            "split": self.split.sizes,
            "normalization_source": self.normalization.source,
            "reference_disclaimer": self.metadata.get(
                "reference_disclaimer", REFERENCE_DISCLAIMER
            ),
        }


def load_coarse_dataset(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any]]:
    """Read the coarse precipitation field of a processed NetCDF file.

    Args:
        path: Path to the processed dataset.

    Returns:
        ``(coarse, latitude, longitude, timestamps)`` where ``coarse`` is
        ``(time, lat, lon)``.

    Raises:
        FileNotFoundError: If the file does not exist.
        KeyError: If it lacks the expected variable or coordinates.
    """
    import xarray as xr

    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"No dataset at '{dataset_path}'.")
    with xr.open_dataset(dataset_path) as dataset:
        if "precipitation" not in dataset:
            raise KeyError(
                f"'{dataset_path}' has no 'precipitation' variable; available: "
                f"{list(dataset.data_vars)}."
            )
        field = dataset["precipitation"]
        coarse = np.asarray(field.values, dtype="float64")
        latitude = np.asarray(field["latitude"].values, dtype="float64")
        longitude = np.asarray(field["longitude"].values, dtype="float64")
        timestamps = list(np.atleast_1d(field["time"].values))
    return coarse, latitude, longitude, timestamps


def build_downscaling_dataset(
    coarse: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    timestamps: Sequence[Any],
    target_resolution: float,
    variable: str = DEFAULT_VARIABLE,
    feature_config: FeatureConfig | None = None,
    reference_config: ReferenceConfig | None = None,
    split_config: SplitConfig | None = None,
    time_subset: slice | None = None,
) -> DownscalingDataset:
    """Build the full downscaling dataset from a coarse field.

    For every time step the coarse field is expanded into a feature stack, a
    synthetic high-resolution reference is generated, and the **residual** target
    (reference minus bilinear baseline) is stored. The split is chronological and
    the normalisation statistics come from the training block alone.

    Args:
        coarse: ``(time, lat, lon)`` coarse field.
        latitude: Coarse latitude axis.
        longitude: Coarse longitude axis.
        timestamps: Valid time of each frame.
        target_resolution: Requested fine resolution in degrees.
        variable: Variable name.
        feature_config: Feature scaling.
        reference_config: Synthetic reference settings.
        split_config: Chronological block fractions.
        time_subset: Optional slice of frames, useful for quick tests.

    Returns:
        A :class:`DownscalingDataset`.

    Raises:
        ValueError: If the inputs are inconsistent or too short to split.
    """
    cfg_features = feature_config or FeatureConfig()
    cfg_reference = reference_config or ReferenceConfig()
    cfg_split = split_config or SplitConfig()

    field = np.asarray(coarse, dtype="float64")
    lat = np.asarray(latitude, dtype="float64")
    lon = np.asarray(longitude, dtype="float64")
    if field.ndim != 3:
        raise ValueError(f"coarse must be (time, lat, lon), got shape {field.shape}.")
    if time_subset is not None:
        field = field[time_subset]
        timestamps = list(timestamps)[time_subset]
    if field.shape[0] < 3:
        raise ValueError(
            f"At least 3 time steps are needed for a chronological split, got {field.shape[0]}."
        )
    if field.shape[1] != lat.size or field.shape[2] != lon.size:
        raise ValueError("Latitude/longitude axes do not match the coarse field.")
    if field.shape[0] != len(timestamps):
        raise ValueError(
            f"{field.shape[0]} frames but {len(timestamps)} timestamps; they must match."
        )

    input_resolution = detect_resolution(lat)
    n_samples = field.shape[0]

    features: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    references: list[np.ndarray] = []
    baselines: list[np.ndarray] = []
    conservation_errors: list[float] = []
    output_shape: tuple[int, int] | None = None
    fine_axis: np.ndarray | None = None

    for index in range(n_samples):
        coarse_frame = field[index]
        features.append(build_feature_stack(coarse_frame, lat, lon, cfg_features))
        reference = generate_reference(
            coarse_frame, lat, lon, target_resolution, variable=variable, config=cfg_reference
        )
        baseline = bilinear_baseline(coarse_frame, reference.output_shape)
        # The target is the residual the network must learn. The reference is
        # clipped at zero first, so the residual is non-negative and consistent
        # with the model's ReLU output constraint.
        clean_reference = np.maximum(reference.field, 0.0)
        targets.append(np.maximum(clean_reference - baseline, 0.0)[None, :, :])
        references.append(clean_reference)
        baselines.append(baseline)
        conservation_errors.append(reference.metadata["conservation_error_max"])
        if output_shape is None:
            output_shape = reference.output_shape
            fine_axis = reference.latitude

    if output_shape is None or fine_axis is None:
        raise ValueError("No samples were produced.")
    output_resolution = detect_resolution(fine_axis)

    feature_stack = np.stack(features, axis=0).astype("float32")
    target_stack = np.stack(targets, axis=0).astype("float32")
    reference_stack = np.stack(references, axis=0)
    baseline_stack = np.stack(baselines, axis=0)

    # --- chronological split (never shuffled) ---
    n_train = max(int(round(cfg_split.train * n_samples)), 1)
    n_validation = int(round(cfg_split.validation * n_samples))
    n_train = min(n_train, n_samples - 1)
    n_validation = min(n_validation, max(n_samples - n_train - 1, 0))
    split = SplitIndices(
        train=list(range(0, n_train)),
        validation=list(range(n_train, n_train + n_validation)),
        test=list(range(n_train + n_validation, n_samples)),
    )
    if not split.test:
        raise ValueError(
            f"The chronological split produced an empty test block from {n_samples} frames."
        )

    # --- normalisation from the TRAINING block only ---
    normalization = fit_normalization(
        feature_stack, split.train, source=f"training split only ({len(split.train)} frames)"
    )

    return DownscalingDataset(
        features=feature_stack,
        targets=target_stack,
        references=reference_stack,
        baselines=baseline_stack,
        coarse=field,
        timestamps=list(timestamps),
        normalization=normalization,
        split=split,
        input_resolution_deg=input_resolution,
        output_resolution_deg=output_resolution,
        input_shape=(field.shape[1], field.shape[2]),
        output_shape=output_shape,
        variable=variable,
        metadata={
            "reference_disclaimer": REFERENCE_DISCLAIMER,
            "conservation_error_max": float(max(conservation_errors)),
            "reference_seed": cfg_reference.seed,
            "structure_strength": cfg_reference.structure_strength,
            "target_definition": "max(reference - bilinear_baseline, 0)",
            "feature_names": list(FEATURE_NAMES),
            "coarse_latitude": [float(value) for value in np.asarray(lat).tolist()],
            "coarse_longitude": [float(value) for value in np.asarray(lon).tolist()],
            "fine_latitude": [float(value) for value in np.asarray(fine_axis).tolist()],
        },
    )





