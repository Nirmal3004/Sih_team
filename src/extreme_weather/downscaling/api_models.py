"""Pydantic schemas for the AI downscaling API (SIH26078, Phase 4).

These are the *external* contract of the Phase 4 API, mirroring how Phases 1-3
define their responses once and reuse them everywhere.

Every response carries the prototype ``notice``. Arrays are **never** returned
through the normal API: a downscaled field is thousands of values, so responses
carry metadata, statistics and file references instead.

Semantics
---------
* ``input_resolution_deg`` / ``output_resolution_deg`` are **detected from the
  data**, never hard-coded.
* ``method`` is ``nearest``, ``cubic`` or ``ai``.
* ``metrics`` measure agreement with a **synthetic** reference, not forecast
  verification. Undefined values (POD with no observed events) are ``None``,
  never 0.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "PROTOTYPE_NOTICE",
    "DownscalingEvaluationResponse",
    "DownscalingMetricsResponse",
    "DownscalingModelInfo",
    "DownscalingModelStatus",
    "DownscalingRequest",
    "DownscalingResponse",
    "EvaluationRequest",
    "ExtremeEventMetricsResponse",
]

PROTOTYPE_NOTICE = (
    "Research Prototype - Synthetic Data - Not an Operational Forecast. The "
    "high-resolution target is synthetic, not observational truth. Metrics measure "
    "agreement with that generated target and are NOT real-world skill. Higher "
    "resolution does not imply higher accuracy."
)


class DownscalingRequest(BaseModel):
    """Body of ``POST /api/v1/downscaling/run``."""

    model_config = ConfigDict(extra="forbid")

    variable: str = Field(
        default="precipitation",
        min_length=1,
        max_length=64,
        description="Variable to downscale. Only 'precipitation' is supported in Phase 4.",
    )
    method: str = Field(
        default="nearest", min_length=1, max_length=16, description="nearest | cubic | ai"
    )
    target_resolution: float | None = Field(
        default=None, gt=0.0, description="Target resolution in degrees (default: configured)."
    )
    time_index: int = Field(default=0, ge=0, description="Index of the frame to downscale.")


class EvaluationRequest(BaseModel):
    """Body of ``POST /api/v1/downscaling/evaluate``."""

    model_config = ConfigDict(extra="forbid")

    methods: list[str] = Field(
        default_factory=lambda: ["nearest", "cubic", "ai"],
        min_length=1,
        description="Methods to score, all on the same test samples.",
    )
    wet_threshold: float | None = Field(
        default=None, ge=0.0, description="Wet/dry threshold in mm h-1."
    )
    max_frames: int | None = Field(
        default=None, gt=0, description="Use only the first N frames of the dataset."
    )


class DownscalingResponse(BaseModel):
    """Result of one downscaling run (metadata and statistics, no array)."""

    model_config = ConfigDict(extra="forbid")

    method: str
    variable: str
    input_resolution_deg: float
    output_resolution_deg: float
    input_resolution_km: float | None = Field(
        default=None, description="Approximate input resolution in km, for readability."
    )
    output_resolution_km: float | None = Field(
        default=None, description="Approximate output resolution in km, for readability."
    )
    scale_factor: float
    input_shape: list[int]
    output_shape: list[int]
    statistics: dict[str, float]
    model_version: str | None = None
    notice: str = Field(default=PROTOTYPE_NOTICE)


class DownscalingMetricsResponse(BaseModel):
    """Metrics of one method in a comparison row."""

    model_config = ConfigDict(extra="forbid")

    method: str
    mae: float = Field(description="Mean absolute error, mm h-1.")
    rmse: float = Field(description="Root mean squared error, mm h-1.")
    bias: float = Field(description="Mean signed error; positive means over-prediction.")
    correlation: float | None = Field(
        default=None, description="Pearson correlation; None when undefined."
    )
    pod: float | None = Field(
        default=None, description="Probability of detection; None when no events observed."
    )
    far: float | None = Field(
        default=None, description="False alarm ratio; None when nothing predicted."
    )
    csi: float | None = Field(default=None, description="Critical success index.")
    frequency_bias: int | None = Field(
        default=None, description="Excess (or deficit) wet grid points, not a percentage."
    )
    n_points: int = Field(description="Number of grid points compared.")


class ExtremeEventMetricsResponse(BaseModel):
    """Extreme-event preservation metrics of one method."""

    model_config = ConfigDict(extra="forbid")

    method: str
    peak_value: float
    reference_peak: float
    peak_ratio: float | None
    peak_absolute_error: float
    peak_overestimation: float
    centroid_lat: float
    centroid_lon: float
    reference_centroid_lat: float
    reference_centroid_lon: float
    centroid_distance_km: float
    wet_area_km2: float
    reference_wet_area_km2: float
    area_ratio: float | None
    structure_correlation: float | None = Field(
        default=None,
        description="Correlation restricted to wet reference points (shape, not total).",
    )


class DownscalingEvaluationResponse(BaseModel):
    """Baseline-vs-AI comparison over the same test samples."""

    model_config = ConfigDict(extra="forbid")

    variable: str
    test_sample_count: int
    input_resolution_deg: float
    output_resolution_deg: float
    methods: list[DownscalingMetricsResponse]
    extreme_event: list[ExtremeEventMetricsResponse]
    ranking: list[str] = Field(
        default_factory=list,
        description=(
            "Methods ordered by MAE, best first. Reported exactly as measured; no "
            "claim is made that any method is operationally 'best'."
        ),
    )
    notice: str = Field(default=PROTOTYPE_NOTICE)


class DownscalingModelStatus(BaseModel):
    """Availability and provenance of the AI model (``GET /downscaling/status``)."""

    model_config = ConfigDict(extra="forbid")

    available: bool
    backend: str
    checkpoint_path: str | None = None
    model_version: str | None = None
    input_resolution_deg: float | None = None
    output_resolution_deg: float | None = None
    input_shape: list[int] | None = None
    output_shape: list[int] | None = None
    scale_factor: float | None = None
    variable: str | None = None
    training: dict = Field(default_factory=dict)
    best_validation_loss: float | None = None
    methods: list[str] = Field(
        default_factory=lambda: ["nearest", "cubic", "ai"],
        description="Downscaling methods selectable through the API.",
    )
    message: str | None = None
    notice: str = Field(default=PROTOTYPE_NOTICE)


class DownscalingModelInfo(BaseModel):
    """Architecture and feature contract (``GET /downscaling/model``)."""

    model_config = ConfigDict(extra="forbid")

    architecture: str
    backend: str
    available: bool
    hyperparameters: dict = Field(
        description=(
            "Model hyper-parameters. Named 'hyperparameters' rather than "
            "'model_config', which collides with pydantic's own field."
        )
    )
    input_features: list[str]
    output_variable: str | None = None
    residual_learning: bool = True
    output_constraint: str
    normalization: dict
    training: dict = Field(default_factory=dict)
    reference_disclaimer: str
    notice: str = Field(default=PROTOTYPE_NOTICE)

