"""Inference helpers for the AI downscaler (SIH26078, Phase 4).

The learned model is always *residual*:

```text
ai_output = bilinear(coarse) + ReLU(residual_net(normalised_features))
```

so the AI result is compared with a real baseline on exactly the same samples,
which is the only way the comparison in ``docs/AI_DOWNSCALING.md`` means anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import xarray as xr

from extreme_weather.downscaling.dataset import (
    NormalizationStats,
    bilinear_baseline,
)
from extreme_weather.downscaling.features import build_feature_stack
from extreme_weather.downscaling.model import DownscalingError, DownscalingModel

__all__ = [
    "ExtremeEventMetrics",
    "ai_downscale",
    "baseline_field",
    "evaluate_extreme_event",
    "predict_ai_field",
    "register_ai_method",
    "run_baseline_method",
]

#: The interpolation methods the registry exposes, re-exported for convenience.
BASELINE_METHODS = ("nearest", "cubic")


def baseline_field(
    coarse: xr.DataArray,
    target_resolution: float,
    method: str = "nearest",
) -> np.ndarray:
    """Run a registered baseline method and return a 2-D array.

    The Phase 0 registry is reused unchanged, so ``nearest`` and ``cubic`` behave
    exactly as they always did and the AI method can be compared against them on
    identical inputs.

    Raises:
        UnknownDownscalingMethodError: If the method is not registered.
        DownscalingError: If the resolution is invalid for this field.
    """
    from extreme_weather.downscaling import downscale

    return np.asarray(
        downscale(coarse, target_resolution, method=method).values, dtype="float64"
    )


def run_baseline_method(
    method: str,
    coarse: xr.DataArray,
    target_resolution: float,
) -> np.ndarray:
    """Alias of :func:`baseline_field` kept for readable call sites."""
    return baseline_field(coarse, target_resolution, method=method)


def ai_downscale(
    model: DownscalingModel,
    coarse_field: xr.DataArray,
    target_resolution: float,
    normalization: NormalizationStats,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Downscale one coarse field with the learned model.

    Args:
        model: A trained :class:`DownscalingModel` in ``eval()`` mode.
        coarse_field: A ``(latitude, longitude)`` labelled DataArray.
        target_resolution: Requested fine resolution in degrees.
        normalization: Statistics fitted on the **training** block only.
        output_shape: Explicit target grid; derived from ``target_resolution``
            when omitted.

    Returns:
        A ``(lat_target, lon_target)`` non-negative array in mm h-1.

    Raises:
        DownscalingError: If the field is not 2-D or the shapes are invalid.
    """
    values = np.asarray(coarse_field.values, dtype="float64")
    if values.ndim != 2:
        raise DownscalingError(
            f"ai_downscale expects a 2-D (lat, lon) field, got shape {values.shape}."
        )
    latitude = np.asarray(coarse_field["latitude"].values, dtype="float64")
    longitude = np.asarray(coarse_field["longitude"].values, dtype="float64")

    features = build_feature_stack(values, latitude, longitude)
    normalized = normalization.apply(features)[None, ...]  # [1, C, H, W]
    if output_shape is None:
        from extreme_weather.downscaling.baseline import build_target_coordinate

        output_shape = (
            int(build_target_coordinate(latitude, target_resolution).size),
            int(build_target_coordinate(longitude, target_resolution).size),
        )

    model.eval()
    with torch.no_grad():
        residual = model(torch.tensor(normalized, dtype=torch.float32), output_shape)
    residual_array = residual.numpy()[0, 0].astype("float64")

    baseline = bilinear_baseline(values, output_shape)
    # Both terms are non-negative, so the sum cannot be negative. This is a
    # documented consequence of the ReLU residual, not a silent clip.
    return np.maximum(baseline + residual_array, 0.0)


def register_ai_method() -> None:
    """Register the learned ``ai`` method in the Phase 0 registry.

    The AI method needs a checkpoint and training-only normalisation statistics,
    which a bare ``downscale(data, resolution, method=...)`` call cannot supply.
    This wrapper therefore loads the *default* checkpoint itself and raises a
    clear error when it is missing. It exists so that ``available_methods()``
    lists ``ai`` alongside the baselines and the registry stays the single
    selection point, as Phase 0 intended.

    Registration is idempotent and never replaces an existing entry.
    """
    from extreme_weather.downscaling.registry import _REGISTRY, register_method

    if "ai" in _REGISTRY:
        return

    @register_method(
        "ai",
        description=(
            "Learned CNN residual downscaling (Phase 4 prototype). Requires "
            "models/downscaling/best.pt; raises when the checkpoint is missing."
        ),
        minimum_points_per_dim=2,
    )
    def _ai_downscale(
        data: xr.DataArray,
        target_resolution: float,
        **_: Any,
    ) -> xr.DataArray:
        """Downscale with the learned model, loading the default checkpoint."""
        from extreme_weather.downscaling.checkpoint import (
            DownscalingCheckpointError,
            load_checkpoint,
        )

        try:
            checkpoint = load_checkpoint()
        except DownscalingCheckpointError as exc:
            # The loader's message already names the script that creates the
            # checkpoint, so it is not repeated here.
            raise DownscalingError(
                f"The 'ai' method needs a trained checkpoint: {exc}"
            ) from exc
        field = ai_downscale(
            checkpoint.build_model(),
            data,
            float(target_resolution),
            checkpoint.normalization,
        )
        return xr.DataArray(
            field,
            dims=data.dims,
            coords={dim: data.coords[dim] for dim in ("latitude", "longitude") if dim in data.coords},
            name=data.name,
        )



def predict_ai_field(
    model: DownscalingModel,
    features: np.ndarray,
    normalization: NormalizationStats,
    output_shape: tuple[int, int],
) -> np.ndarray:
    """Predict a residual for pre-built features (used by the evaluator)."""
    normalized = normalization.apply(features)[None, ...]
    model.eval()
    with torch.no_grad():
        residual = model(torch.tensor(normalized, dtype=torch.float32), output_shape)
    return residual.numpy()[0, 0].astype("float64")


@dataclass(frozen=True)
class ExtremeEventMetrics:
    """How well one method preserves an extreme precipitation event.

    Because this project injects a *known* event, the peak can be compared with
    the reference peak and the event centre with the reference centre. What is
    **not** claimed is exact localisation: the reported displacement is the
    distance between the two argmax points, which is a genuine measurement, but
    it is measured on a synthetic field whose "truth" is itself generated.

    Attributes:
        method: Method name.
        peak_value: Maximum of the method's field.
        reference_peak: Maximum of the reference field.
        peak_ratio: ``peak_value / reference_peak``; 1.0 is a perfect peak.
        peak_absolute_error: ``|peak_value - reference_peak|``.
        peak_overestimation: ``peak_value - reference_peak`` (signed).
        centroid_lat / centroid_lon: Intensity-weighted centre of the method.
        reference_centroid_lat / reference_centroid_lon: Centre of the reference.
        centroid_distance_km: Great-circle distance between the two centres.
        wet_area_km2: Area above the event threshold.
        reference_wet_area_km2: Reference wet area.
        area_ratio: ``wet_area_km2 / reference_wet_area_km2``.
        structure_correlation: Correlation of the fields above the threshold,
            which measures whether the *shape* is reproduced, not just the total.
    """

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
    structure_correlation: float | None

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable view used by the API and the report."""
        return dict(self.__dict__)


def _weighted_centroid(
    field: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
) -> tuple[float, float]:
    """Return the intensity-weighted centre of a 2-D field."""
    weights = np.clip(field, 0.0, None)
    total = weights.sum()
    if total <= 0.0:
        return float(np.mean(latitude)), float(np.mean(longitude))
    lat_2d, lon_2d = np.meshgrid(latitude, longitude, indexing="ij")
    return (
        float((weights * lat_2d).sum() / total),
        float((weights * lon_2d).sum() / total),
    )


def _cell_area_km2(latitude: np.ndarray, longitude: np.ndarray) -> float:
    """Approximate area of one grid cell in km^2 at the mean latitude."""
    from extreme_weather.tracking.metrics import EARTH_RADIUS_KM

    mean_latitude = float(np.mean(latitude))
    d_lat = abs(float(latitude[1] - latitude[0])) if latitude.size > 1 else 1.0
    d_lon = abs(float(longitude[1] - longitude[0])) if longitude.size > 1 else 1.0
    d_lat_km = np.deg2rad(d_lat) * EARTH_RADIUS_KM
    d_lon_km = np.deg2rad(d_lon) * EARTH_RADIUS_KM * np.cos(np.deg2rad(mean_latitude))
    return float(d_lat_km * d_lon_km)


def evaluate_extreme_event(
    method: str,
    field: np.ndarray,
    reference: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    threshold: float = 0.1,
) -> ExtremeEventMetrics:
    """Measure how one method preserves the injected extreme event.

    Args:
        method: Method name.
        field: The method's high-resolution field.
        reference: The synthetic reference field (same shape).
        latitude: Fine latitude axis.
        longitude: Fine longitude axis.
        threshold: Wet/dry threshold defining the event area, in mm h-1.

    Returns:
        An :class:`ExtremeEventMetrics`.

    Raises:
        ValueError: If the field and reference shapes differ.
    """
    from extreme_weather.downscaling.metrics import pearson_correlation
    from extreme_weather.tracking.metrics import haversine_distance_km

    values = np.asarray(field, dtype="float64")
    truth = np.asarray(reference, dtype="float64")
    if values.shape != truth.shape:
        raise ValueError(f"field and reference shapes differ: {values.shape} vs {truth.shape}.")

    peak_value = float(values.max())
    reference_peak = float(truth.max())
    centroid_lat, centroid_lon = _weighted_centroid(values, latitude, longitude)
    reference_lat, reference_lon = _weighted_centroid(truth, latitude, longitude)

    cell_area = _cell_area_km2(latitude, longitude)
    wet_area = float(np.count_nonzero(values >= threshold) * cell_area)
    reference_area = float(np.count_nonzero(truth >= threshold) * cell_area)

    wet_mask = truth >= threshold
    structure_correlation = (
        pearson_correlation(values[wet_mask], truth[wet_mask])
        if wet_mask.sum() >= 2
        else None
    )

    return ExtremeEventMetrics(
        method=method,
        peak_value=peak_value,
        reference_peak=reference_peak,
        peak_ratio=(peak_value / reference_peak) if reference_peak > 0.0 else None,
        peak_absolute_error=abs(peak_value - reference_peak),
        peak_overestimation=peak_value - reference_peak,
        centroid_lat=centroid_lat,
        centroid_lon=centroid_lon,
        reference_centroid_lat=reference_lat,
        reference_centroid_lon=reference_lon,
        centroid_distance_km=float(
            haversine_distance_km(centroid_lat, centroid_lon, reference_lat, reference_lon)
        ),
        wet_area_km2=wet_area,
        reference_wet_area_km2=reference_area,
        area_ratio=(wet_area / reference_area) if reference_area > 0.0 else None,
        structure_correlation=structure_correlation,
    )
