"""Downscaling service and baseline comparison (SIH26078, Phase 4).

This module ties everything together: load the coarse field, load the checkpoint,
run any registered method (``nearest``, ``cubic``, ``ai``), score every method on
the **same** test samples, and evaluate the extreme event.

Error handling
--------------
Each failure mode raises a distinct exception so the API can map it precisely:

* :class:`DownscalingModelUnavailableError` - no usable checkpoint -> HTTP 503
* :class:`DownscalingVariableError` - unsupported variable -> HTTP 422
* :class:`DownscalingResolutionError` - invalid/unavailable target resolution
* :class:`DownscalingDataError` - missing or malformed dataset

Nothing is fabricated: without a checkpoint, ``/status`` reports
``available: false`` rather than inventing a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import xarray as xr

from extreme_weather.downscaling.checkpoint import (
    DownscalingCheckpoint,
    DownscalingCheckpointError,
    load_checkpoint,
)
from extreme_weather.downscaling.features import FEATURE_NAMES
from extreme_weather.downscaling.dataset import (
    DEFAULT_VARIABLE,
    DownscalingDataset,
    bilinear_baseline,
    build_downscaling_dataset,
    load_coarse_dataset,
)
from extreme_weather.downscaling.inference import (
    ExtremeEventMetrics,
    evaluate_extreme_event,
    predict_ai_field,
)
from extreme_weather.downscaling.metrics import (
    DEFAULT_WET_THRESHOLD,
    DownscalingMetrics,
    evaluate,
)
from extreme_weather.downscaling.model import DownscalingModel
from extreme_weather.downscaling.reference import REFERENCE_DISCLAIMER, detect_resolution

__all__ = [
    "SUPPORTED_VARIABLES",
    "ComparisonReport",
    "DownscalingDataError",
    "DownscalingModelUnavailableError",
    "DownscalingResolutionError",
    "DownscalingService",
    "DownscalingVariableError",
    "RunResult",
]

#: Variables this phase supports. Precipitation is the only one the model was
#: trained for; the others are exposed for baseline-only runs.
SUPPORTED_VARIABLES: tuple[str, ...] = ("precipitation",)


class DownscalingModelUnavailableError(RuntimeError):
    """No usable AI checkpoint exists (maps to HTTP 503)."""


class DownscalingVariableError(ValueError):
    """The requested variable is not supported (maps to HTTP 422)."""


class DownscalingResolutionError(ValueError):
    """The requested target resolution is invalid (maps to HTTP 422)."""


class DownscalingDataError(ValueError):
    """The dataset is missing or malformed (maps to HTTP 422)."""


@dataclass(frozen=True)
class RunResult:
    """Output of one downscaling run plus its metadata.

    Attributes:
        method: ``nearest``, ``cubic`` or ``ai``.
        variable: Downscaled variable name.
        input_resolution_deg / output_resolution_deg: Detected from the data.
        input_shape / output_shape: ``(lat, lon)`` grid sizes.
        statistics: Summary statistics of the produced field.
        model_version: Version of the checkpoint, or ``None`` for baselines.
        notice: The prototype disclaimer.
    """

    method: str
    variable: str
    input_resolution_deg: float
    output_resolution_deg: float
    input_shape: tuple[int, int]
    output_shape: tuple[int, int]
    statistics: dict[str, float]
    model_version: str | None
    notice: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view (no arrays)."""
        return {
            "method": self.method,
            "variable": self.variable,
            "input_resolution_deg": self.input_resolution_deg,
            "output_resolution_deg": self.output_resolution_deg,
            "input_shape": list(self.input_shape),
            "output_shape": list(self.output_shape),
            "statistics": self.statistics,
            "model_version": self.model_version,
            "notice": self.notice,
        }


@dataclass(frozen=True)
class ComparisonReport:
    """Every method scored on the same test samples.

    Attributes:
        metrics: One row per method, in the order they were run.
        extreme_event: Extreme-event metrics per method, when evaluated.
        variable: Variable compared.
        test_sample_count: Number of test frames scored.
        notice: The prototype disclaimer.
    """

    metrics: list[DownscalingMetrics]
    extreme_event: list[ExtremeEventMetrics]
    variable: str
    test_sample_count: int
    notice: str

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view used by the API and the CLI."""
        return {
            "variable": self.variable,
            "test_sample_count": self.test_sample_count,
            "methods": [row.as_dict() for row in self.metrics],
            "extreme_event": [row.as_dict() for row in self.extreme_event],
            "notice": self.notice,
        }


def _statistics(field: np.ndarray) -> dict[str, float]:
    """Return the summary statistics reported for a produced field."""
    values = np.asarray(field, dtype="float64")
    return {
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min_positive": float(values[values > 0].min()) if np.any(values > 0) else 0.0,
    }


def _backend_name() -> str:
    """Name of the tensor backend, reported in the status/model payloads."""
    import torch

    return f"pytorch {torch.__version__} (cpu)"


def _fine_axes(dataset: DownscalingDataset) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild the fine latitude/longitude axes of a dataset's output grid.

    The axes are uniform by construction, so a ``linspace`` over the same span
    and the same number of points reproduces them exactly - which is what the
    event metrics need for centroid and area calculations.
    """
    coarse_lat = np.linspace(
        float(np.min(dataset.metadata["coarse_latitude"])),
        float(np.max(dataset.metadata["coarse_latitude"])),
        dataset.input_shape[0],
    )
    coarse_lon = np.linspace(
        float(np.min(dataset.metadata["coarse_longitude"])),
        float(np.max(dataset.metadata["coarse_longitude"])),
        dataset.input_shape[1],
    )
    latitude = np.linspace(
        coarse_lat[0], coarse_lat[-1], dataset.output_shape[0]
    )
    longitude = np.linspace(
        coarse_lon[0], coarse_lon[-1], dataset.output_shape[1]
    )
    return latitude, longitude


class DownscalingService:
    """Runs baseline and AI downscaling, and compares them on the test set.

    Args:
        checkpoint_path: Checkpoint location; defaults to
            ``models/downscaling/best.pt``.
        dataset_path: Processed NetCDF to read; defaults to the configured
            processed dataset.
    """

    def __init__(
        self,
        checkpoint_path: Path | str | None = None,
        dataset_path: Path | str | None = None,
    ) -> None:
        self._checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        self._dataset_path = Path(dataset_path) if dataset_path else None
        self._checkpoint: DownscalingCheckpoint | None = None

    # --- configuration helpers -------------------------------------------

    @property
    def dataset_path(self) -> Path:
        """The coarse dataset, created on first use."""
        if self._dataset_path is None:
            from extreme_weather.config import get_settings

            settings = get_settings()
            self._dataset_path = settings.processed_dir / "sample_history_processed.nc"
        return self._dataset_path

    def checkpoint(self) -> DownscalingCheckpoint:
        """Load and cache the checkpoint.

        Raises:
            DownscalingModelUnavailableError: If no valid checkpoint exists.
        """
        if self._checkpoint is None:
            try:
                self._checkpoint = load_checkpoint(self._checkpoint_path)
            except DownscalingCheckpointError as exc:
                raise DownscalingModelUnavailableError(str(exc)) from exc
        return self._checkpoint

    def has_model(self) -> bool:
        """True when a checkpoint can be loaded and a model built."""
        try:
            self.checkpoint().build_model()
        except (DownscalingModelUnavailableError, DownscalingCheckpointError):
            return False
        return True

    @staticmethod
    def validate_variable(variable: str) -> str:
        """Check the variable is supported.

        Raises:
            DownscalingVariableError: If it is not.
        """
        if variable not in SUPPORTED_VARIABLES:
            raise DownscalingVariableError(
                f"Variable '{variable}' is not supported in this phase. "
                f"Supported: {', '.join(SUPPORTED_VARIABLES)}."
            )
        return variable

    @staticmethod
    def validate_resolution(resolution: float) -> float:
        """Check the target resolution is a usable positive number.

        Raises:
            DownscalingResolutionError: If it is not.
        """
        try:
            value = float(resolution)
        except (TypeError, ValueError) as exc:
            raise DownscalingResolutionError(
                f"target_resolution must be a number, got {resolution!r}."
            ) from exc
        if not np.isfinite(value) or value <= 0.0:
            raise DownscalingResolutionError(
                f"target_resolution must be finite and > 0, got {resolution!r}."
            )
        return value

    def load_coarse(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Any]]:
        """Read the coarse field, mapping failures to a service error.

        Raises:
            DownscalingDataError: If the dataset is missing or malformed.
        """
        try:
            return load_coarse_dataset(self.dataset_path)
        except (FileNotFoundError, KeyError) as exc:
            raise DownscalingDataError(str(exc)) from exc

    # --- single-field runs -----------------------------------------------

    def run_field(
        self,
        method: str,
        coarse_frame: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
        target_resolution: float,
        variable: str = DEFAULT_VARIABLE,
        features: np.ndarray | None = None,
    ) -> tuple[np.ndarray, RunResult]:
        """Downscale one coarse frame with the requested method.

        Args:
            method: ``nearest``, ``cubic`` or ``ai``.
            coarse_frame: ``(lat, lon)`` coarse field.
            latitude: Coarse latitude axis.
            longitude: Coarse longitude axis.
            target_resolution: Requested fine resolution in degrees.
            variable: Variable name.
            features: Pre-built feature stack, required for ``method="ai"``.

        Returns:
            ``(field, result)`` where ``field`` is the high-resolution array.

        Raises:
            DownscalingModelUnavailableError: If ``ai`` is requested without a
                usable checkpoint.
            DownscalingDataError: If ``ai`` is requested without features.
        """
        resolution = self.validate_resolution(target_resolution)
        self.validate_variable(variable)

        coarse_array = np.asarray(coarse_frame, dtype="float64")
        coarse_da = xr.DataArray(
            coarse_array,
            dims=("latitude", "longitude"),
            coords={"latitude": latitude, "longitude": longitude},
            name=variable,
        )
        input_resolution = detect_resolution(np.asarray(latitude, dtype="float64"))

        if method in ("nearest", "cubic"):
            from extreme_weather.downscaling import downscale

            field = np.asarray(
                downscale(coarse_da, resolution, method=method).values, dtype="float64"
            )
            model_version = None
            notice = (
                "Classical interpolation baseline. Agreement with the synthetic "
                "reference is not a forecast skill score."
            )
        elif method == "ai":
            checkpoint = self.checkpoint()  # raises when unavailable
            from extreme_weather.downscaling.baseline import build_target_coordinate
            from extreme_weather.downscaling.features import build_feature_stack

            output_shape = (
                int(build_target_coordinate(np.asarray(latitude), resolution).size),
                int(build_target_coordinate(np.asarray(longitude), resolution).size),
            )
            # The features are built here when the caller did not supply them, so
            # the API path works exactly like the in-process one.
            frame_features = (
                features
                if features is not None
                else build_feature_stack(coarse_array, np.asarray(latitude), np.asarray(longitude))
            )
            residual = predict_ai_field(
                checkpoint.build_model(), frame_features, checkpoint.normalization, output_shape
            )
            field = np.maximum(bilinear_baseline(coarse_array, output_shape) + residual, 0.0)
            model_version = checkpoint.model_version
            notice = (
                "AI prototype downscaling against a SYNTHETIC reference. Not an "
                "operational forecast; higher resolution does not imply higher accuracy."
            )
        else:
            raise DownscalingDataError(
                f"Unknown method '{method}'. Use 'nearest', 'cubic' or 'ai'."
            )

        output_resolution = detect_resolution(
            np.linspace(float(np.min(latitude)), float(np.max(latitude)), field.shape[0])
        )
        return field, RunResult(
            method=method,
            variable=variable,
            input_resolution_deg=input_resolution,
            output_resolution_deg=output_resolution,
            input_shape=(coarse_array.shape[0], coarse_array.shape[1]),
            output_shape=(field.shape[0], field.shape[1]),
            statistics=_statistics(field),
            model_version=model_version,
            notice=notice,
        )

    # --- status / model info ---------------------------------------------

    def status(self) -> dict[str, Any]:
        """Availability and provenance of the AI model.

        Never raises: a missing model is a *status*, not an error, so the
        dashboard can render "not trained yet" instead of failing.
        """
        try:
            checkpoint = self.checkpoint()
        except DownscalingModelUnavailableError as exc:
            return {
                "available": False,
                "backend": _backend_name(),
                "checkpoint_path": str(self._checkpoint_path) if self._checkpoint_path else None,
                "model_version": None,
                "input_resolution_deg": None,
                "output_resolution_deg": None,
                "variable": None,
                "message": str(exc),
            }

        return {
            "available": True,
            "backend": _backend_name(),
            "checkpoint_path": str(checkpoint.path) if checkpoint.path else None,
            "model_version": checkpoint.model_version,
            "input_resolution_deg": checkpoint.input_resolution_deg,
            "output_resolution_deg": checkpoint.output_resolution_deg,
            "input_shape": list(checkpoint.input_shape),
            "output_shape": list(checkpoint.output_shape),
            "scale_factor": checkpoint.scale_factor,
            "variable": checkpoint.variable,
            "training": dict(checkpoint.training),
            "best_validation_loss": checkpoint.best_validation_loss,
        }

    def model_info(self) -> dict[str, Any]:
        """Architecture, feature contract and hyper-parameters."""
        try:
            checkpoint = self.checkpoint()
            config = checkpoint.model_config.as_dict()
            normalization = checkpoint.normalization.as_dict()
            training = dict(checkpoint.training)
            available = True
        except DownscalingModelUnavailableError:
            from extreme_weather.downscaling.model import ModelConfig

            config = ModelConfig().as_dict()
            from extreme_weather.downscaling.features import feature_metadata

            normalization = feature_metadata()
            training = {}
            available = False

        return {
            "architecture": (
                "input projection -> "
                f"{config['num_blocks']} x residual block (3x3 conv) -> 1x1 output conv -> "
                "ReLU residual -> bilinear upsample -> add bilinear baseline"
            ),
            "backend": _backend_name(),
            "available": available,
            # Named `hyperparameters`, not `model_config`: the latter collides with
            # pydantic's own `model_config` attribute on the response schema.
            "hyperparameters": config,
            "input_features": list(FEATURE_NAMES),
            "output_variable": checkpoint.variable if available else None,
            "residual_learning": True,
            "output_constraint": "ReLU on the residual: the output is >= the bilinear baseline",
            "normalization": normalization,
            "training": training,
        }

    # --- baseline vs AI comparison ----------------------------------------

    def compare(
        self,
        dataset: DownscalingDataset,
        methods: Sequence[str] = ("nearest", "cubic", "ai"),
        threshold: float | None = DEFAULT_WET_THRESHOLD,
        event_frame: int | None = None,
    ) -> ComparisonReport:
        """Score every method on the **same** test samples.

        All methods are evaluated on identical frames, so the comparison is
        apples-to-apples. Each metric is whatever the implementation produces; no
        ranking is asserted and no metric is tuned.

        Args:
            dataset: A built downscaling dataset.
            methods: Methods to score, in report order.
            threshold: Wet/dry threshold; ``None`` skips classification metrics.
            event_frame: Index of the frame used for the extreme-event
                evaluation. Defaults to the **test** frame with the largest
                reference peak, so the event is scored on unseen data.

        Returns:
            A :class:`ComparisonReport`.

        Raises:
            DownscalingModelUnavailableError: If ``ai`` is requested without a
                usable checkpoint.
            DownscalingDataError: If the test block is empty.
        """
        test_indices = dataset.split.test
        if not test_indices:
            raise DownscalingDataError("The chronological split produced no test samples.")

        checkpoint = self.checkpoint() if "ai" in methods else None
        model = checkpoint.build_model() if checkpoint is not None else None

        rows: list[DownscalingMetrics] = []
        event_rows: list[ExtremeEventMetrics] = []

        # The event frame is chosen from the test block: never from training.
        if event_frame is None:
            event_frame = max(
                test_indices, key=lambda index: float(dataset.references[index].max())
            )

        latitude, longitude = _fine_axes(dataset)

        for method in methods:
            # Every method sees exactly the same frames, so the rows are
            # directly comparable.
            predictions = np.concatenate(
                [
                    self._predict_dataset_frame(
                        method, dataset, index, model, checkpoint
                    ).ravel()
                    for index in test_indices
                ]
            )
            references = dataset.references[test_indices].ravel()
            rows.append(evaluate(method, predictions, references, threshold))

            # --- extreme-event evaluation on the selected frame ---
            event_prediction = self._predict_dataset_frame(
                method, dataset, event_frame, model, checkpoint
            )
            event_rows.append(
                evaluate_extreme_event(
                    method,
                    event_prediction,
                    dataset.references[event_frame],
                    latitude,
                    longitude,
                    threshold if threshold is not None else DEFAULT_WET_THRESHOLD,
                )
            )

        return ComparisonReport(
            metrics=rows,
            extreme_event=event_rows,
            variable=dataset.variable,
            test_sample_count=len(test_indices),
            notice=(
                "Metrics measure agreement with a SYNTHETIC high-resolution reference, "
                "not forecast skill. Higher resolution does not imply higher accuracy. "
                "Research Prototype - Not an Operational Forecast."
            ),
        )

    @staticmethod
    def _predict_dataset_frame(
        method: str,
        dataset: DownscalingDataset,
        index: int,
        model: DownscalingModel | None,
        checkpoint: DownscalingCheckpoint | None,
    ) -> np.ndarray:
        """Produce one high-resolution field with the requested method.

        Raises:
            DownscalingDataError: If a method returns a grid other than the one the
                dataset was built with, which would silently corrupt the
                comparison.
        """
        coarse = np.asarray(dataset.coarse[index], dtype="float64")
        if method in ("nearest", "cubic"):
            # The axes must be attached: without real coordinates the registry
            # interpolates over integer indices and lands on a different grid.
            latitude = np.asarray(dataset.metadata["coarse_latitude"], dtype="float64")
            longitude = np.asarray(dataset.metadata["coarse_longitude"], dtype="float64")
            coarse_da = xr.DataArray(
                coarse,
                dims=("latitude", "longitude"),
                coords={"latitude": latitude, "longitude": longitude},
                name=dataset.variable,
            )
            from extreme_weather.downscaling import downscale

            field = np.asarray(
                downscale(
                    coarse_da, dataset.output_resolution_deg, method=method
                ).values,
                dtype="float64",
            )
        else:
            if model is None or checkpoint is None:
                raise DownscalingModelUnavailableError(
                    "The AI method requires a loaded checkpoint."
                )
            residual = predict_ai_field(
                model,
                dataset.features[index],
                checkpoint.normalization,
                dataset.output_shape,
            )
            baseline = bilinear_baseline(coarse, dataset.output_shape)
            field = np.maximum(baseline + residual, 0.0)

        if field.shape != dataset.output_shape:
            raise DownscalingDataError(
                f"Method '{method}' produced a {field.shape} field but the dataset target "
                f"grid is {dataset.output_shape}. The comparison would be meaningless."
            )
        return field




