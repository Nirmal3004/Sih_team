"""Input features for AI downscaling (SIH26078, Phase 4).

Every feature is computed from the **coarse field at the same time step** and
from static coordinates. No feature reads the high-resolution target, and no
feature reads a future time step - both would leak the answer into the input and
inflate every metric. A test asserts that the features are unchanged when the
reference is replaced by noise.

Feature vector (channel order fixed by :data:`FEATURE_NAMES`)
-------------------------------------------------------------
========  =========================  ============================================
Channel   Name                       Meaning
========  =========================  ============================================
0        ``coarse_precipitation``    The coarse field, normalised.
1        ``latitude_scaled``         Latitude / 90, in ``[-1, 1]``.
2        ``longitude_scaled``        Longitude / 180, in ``[-1, 1]``.
3        ``north_neighbour``         Coarse precipitation one cell north.
4        ``south_neighbour``         ... one cell south.
5        ``east_neighbour``          ... one cell east.
6        ``west_neighbour``          ... one cell west.
7        ``latitude_gradient``       North minus south (meridional gradient).
8        ``longitude_gradient``      East minus west (zonal gradient).
9        ``coarse_std``              Local standard deviation of the coarse field
                                      in a 3x3 window - a proxy for how much
                                      fine structure a cell may hide.
========  =========================  ============================================

Why these features
------------------
* The **coarse value** is the primary signal.
* **Coordinates** let the network learn that precipitation is not spatially
  uniform (orography, coastal gradients); without them a convolutional model is
  translation-invariant and cannot.
* **Neighbour values and gradients** give the local trend, which distinguishes a
  cell on the edge of a storm from one in its core - the most useful cue for
  where the fine field should bulge.
* The **local standard deviation** indicates heterogeneity: a flat neighbourhood
  admits little fine structure, a rough one admits more.

Neighbour channels use edge replication, so nothing is NaN at the boundary. They
are computed from the coarse field *only*, never from the reference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

__all__ = [
    "FEATURE_COUNT",
    "FEATURE_NAMES",
    "FeatureConfig",
    "build_feature_stack",
    "feature_metadata",
]

#: Ordered channel names of the input tensor's channel dimension.
FEATURE_NAMES: tuple[str, ...] = (
    "coarse_precipitation",
    "latitude_scaled",
    "longitude_scaled",
    "north_neighbour",
    "south_neighbour",
    "east_neighbour",
    "west_neighbour",
    "latitude_gradient",
    "longitude_gradient",
    "coarse_std",
)

FEATURE_COUNT: int = len(FEATURE_NAMES)


@dataclass(frozen=True)
class FeatureConfig:
    """Scaling constants applied when building features.

    Attributes:
        precip_scale: Divisor of the coarse precipitation, so the dominant
            channel is O(1). The default 100 matches the mm h-1 range of the
            synthetic generator.
    """

    precip_scale: float = 100.0

    def __post_init__(self) -> None:
        if not (self.precip_scale > 0.0):
            raise ValueError(f"precip_scale must be > 0, got {self.precip_scale}.")


def _edge_replicate(field: np.ndarray, dlat: int, dlon: int) -> np.ndarray:
    """Shift by whole cells with edge replication (no wrap-around)."""
    result = np.empty_like(field)
    rows, cols = field.shape
    source_rows = np.clip(np.arange(rows) - dlat, 0, rows - 1)
    source_cols = np.clip(np.arange(cols) - dlon, 0, cols - 1)
    result[:] = field[np.ix_(source_rows, source_cols)]
    return result


def _local_std(field: np.ndarray) -> np.ndarray:
    """Local standard deviation over a 3x3 window, edge-replicated."""
    rows, cols = field.shape
    padded = np.pad(field, 1, mode="edge")
    total = np.zeros_like(field)
    total_sq = np.zeros_like(field)
    for dlat in (0, 1, 2):
        for dlon in (0, 1, 2):
            window = padded[dlat : dlat + rows, dlon : dlon + cols]
            total += window
            total_sq += window * window
    mean = total / 9.0
    variance = np.maximum(total_sq / 9.0 - mean * mean, 0.0)
    return np.sqrt(variance)


def build_feature_stack(
    coarse: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    config: FeatureConfig | None = None,
) -> np.ndarray:
    """Build the ``(channels, lat, lon)`` input stack of one time step.

    Args:
        coarse: ``(latitude, longitude)`` coarse field.
        latitude: Coarse latitude axis in degrees.
        longitude: Coarse longitude axis in degrees.
        config: Scaling constants; defaults to :class:`FeatureConfig`.

    Returns:
        A ``(len(FEATURE_NAMES), lat, lon)`` float32 array, finite everywhere.

    Raises:
        ValueError: If the axes do not match the field shape.
    """
    cfg = config or FeatureConfig()
    values = np.asarray(coarse, dtype="float64")
    if values.ndim != 2:
        raise ValueError(f"coarse must be 2-D (lat, lon), got shape {values.shape}.")
    lat = np.asarray(latitude, dtype="float64")
    lon = np.asarray(longitude, dtype="float64")
    if lat.size != values.shape[0] or lon.size != values.shape[1]:
        raise ValueError(
            f"Axis sizes ({lat.size}, {lon.size}) do not match the coarse field {values.shape}."
        )

    lat_2d, lon_2d = np.meshgrid(lat, lon, indexing="ij")
    north = _edge_replicate(values, 1, 0)
    south = _edge_replicate(values, -1, 0)
    east = _edge_replicate(values, 0, -1)
    west = _edge_replicate(values, 0, 1)

    stack = np.stack(
        [
            values / cfg.precip_scale,
            lat_2d / 90.0,
            lon_2d / 180.0,
            north / cfg.precip_scale,
            south / cfg.precip_scale,
            east / cfg.precip_scale,
            west / cfg.precip_scale,
            (north - south) / cfg.precip_scale,
            (east - west) / cfg.precip_scale,
            _local_std(values) / cfg.precip_scale,
        ],
        axis=0,
    )
    return np.nan_to_num(stack, nan=0.0, posinf=0.0, neginf=0.0).astype("float32")


def feature_metadata(config: FeatureConfig | None = None) -> dict[str, Any]:
    """Return a JSON-serialisable description of the feature contract."""
    cfg = config or FeatureConfig()
    return {
        "feature_names": list(FEATURE_NAMES),
        "feature_count": FEATURE_COUNT,
        "precip_scale": cfg.precip_scale,
        "layout": "(channels, latitude, longitude)",
        "note": (
            "Built from the coarse field and static coordinates only. The "
            "high-resolution reference is never used as an input."
        ),
    }

