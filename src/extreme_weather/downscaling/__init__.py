"""Spatial downscaling package (SIH26078).

Phase 0 provided the classical baselines (``nearest``, ``cubic``) behind a
registry, so a learned model could be dropped in without touching call sites.
**Phase 4** does exactly that: the ``ai`` method is registered alongside them and
selected with the same ``downscale(..., method=...)`` call.

The target resolution is always read from the data - the reference task of this
project is 0.125 deg (~12 km) -> 0.05 deg (~5 km), but nothing is hard-coded.

Usage::

    from extreme_weather.downscaling import available_methods, downscale

    available_methods()                                   # ('ai', 'cubic', 'nearest')
    fine = downscale(coarse_field, 0.05)                  # nearest (default)
    smooth = downscale(coarse_field, 0.05, method="cubic")

The learned path is reached through :class:`DownscalingService`, because it also
needs the checkpoint and the training-only normalisation statistics that a bare
registry call cannot carry.
"""

from __future__ import annotations

import extreme_weather.downscaling.baseline as _baseline  # noqa: F401  (registers methods)
from extreme_weather.downscaling.baseline import (
    CUBIC_MIN_POINTS,
    DEFAULT_LATITUDE_DIM,
    DEFAULT_LONGITUDE_DIM,
    DownscalingError,
    build_target_coordinate,
    cubic_downscale,
    nearest_downscale,
    target_coordinates,
)
from extreme_weather.downscaling.registry import (
    DownscalingMethod,
    MethodFunction,
    UnknownDownscalingMethodError,
    available_methods,
    downscale,
    get_method,
    register_method,
    registered_methods,
)
from extreme_weather.downscaling.checkpoint import (
    DownscalingCheckpoint,
    DownscalingCheckpointError,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
)
from extreme_weather.downscaling.dataset import (
    DownscalingDataset,
    NormalizationStats,
    SplitConfig,
    SplitIndices,
    bilinear_baseline,
    build_downscaling_dataset,
    fit_normalization,
    load_coarse_dataset,
)
from extreme_weather.downscaling.features import (
    FEATURE_COUNT,
    FEATURE_NAMES,
    FeatureConfig,
    build_feature_stack,
)
from extreme_weather.downscaling.inference import (
    ExtremeEventMetrics,
    evaluate_extreme_event,
)
from extreme_weather.downscaling.metrics import (
    DEFAULT_WET_THRESHOLD,
    ClassificationMetrics,
    DownscalingMetrics,
    ErrorMetrics,
    evaluate,
)
from extreme_weather.downscaling.model import (
    DownscalingError,
    DownscalingModel,
    ModelConfig,
)
from extreme_weather.downscaling.reference import (
    REFERENCE_DISCLAIMER,
    HighResolutionReference,
    ReferenceConfig,
    generate_reference,
)
from extreme_weather.downscaling.service import (
    ComparisonReport,
    DownscalingDataError,
    DownscalingModelUnavailableError,
    DownscalingResolutionError,
    DownscalingService,
    DownscalingVariableError,
    RunResult,
)
from extreme_weather.downscaling.training import (
    TrainingConfig,
    TrainingResult,
    train,
)

__all__ = [
    "CUBIC_MIN_POINTS",
    "DEFAULT_LATITUDE_DIM",
    "DEFAULT_LONGITUDE_DIM",
    "DEFAULT_WET_THRESHOLD",
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "REFERENCE_DISCLAIMER",
    "ClassificationMetrics",
    "ComparisonReport",
    "DownscalingCheckpoint",
    "DownscalingCheckpointError",
    "DownscalingDataError",
    "DownscalingDataset",
    "DownscalingError",
    "DownscalingMetrics",
    "DownscalingModel",
    "DownscalingModelUnavailableError",
    "DownscalingResolutionError",
    "DownscalingService",
    "DownscalingVariableError",
    "DownscalingMethod",
    "ErrorMetrics",
    "ExtremeEventMetrics",
    "FeatureConfig",
    "HighResolutionReference",
    "MethodFunction",
    "ModelConfig",
    "NormalizationStats",
    "ReferenceConfig",
    "RunResult",
    "SplitConfig",
    "SplitIndices",
    "TrainingConfig",
    "TrainingResult",
    "UnknownDownscalingMethodError",
    "available_methods",
    "bilinear_baseline",
    "build_downscaling_dataset",
    "build_feature_stack",
    "checkpoint_path",
    "cubic_downscale",
    "downscale",
    "evaluate",
    "evaluate_extreme_event",
    "fit_normalization",
    "generate_reference",
    "get_method",
    "load_checkpoint",
    "load_coarse_dataset",
    "nearest_downscale",
    "register_method",
    "registered_methods",
    "save_checkpoint",
    "target_coordinates",
    "train",
]
