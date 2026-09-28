"""Shared deterministic fixtures for the Phase 4 downscaling tests.

Everything is tiny and hand-built - no network, no file I/O beyond ``tmp_path``.
The coarse grid is deliberately small so the 121x121 reference of the real
project grid is not needed: a 9x9 coarse field at 0.125 deg over 1 degree gives a
21x21 fine field at 0.05 deg, which exercises every code path quickly.
"""

from __future__ import annotations

import numpy as np
import pytest

from extreme_weather.downscaling.dataset import build_downscaling_dataset
from extreme_weather.downscaling.features import FeatureConfig
from extreme_weather.downscaling.reference import ReferenceConfig

COARSE_RESOLUTION = 0.125
FINE_RESOLUTION = 0.05


def make_axis(start: float, count: int, resolution: float) -> np.ndarray:
    """Return a uniform axis, mirroring the project's ``build_axis``."""
    return start + resolution * np.arange(count, dtype="float64")


def make_coarse_field(
    time_steps: int = 12,
    size: int = 9,
    with_event: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
    """Build a small coarse precipitation field with an optional extreme event.

    Returns:
        ``(coarse, latitude, longitude, timestamps)`` where ``coarse`` is
        ``(time, lat, lon)``.
    """
    latitude = make_axis(10.0, size, COARSE_RESOLUTION)
    longitude = make_axis(74.0, size, COARSE_RESOLUTION)
    lat_2d, lon_2d = np.meshgrid(latitude, longitude, indexing="ij")

    base = 1.8 * np.clip(0.25 + 0.45 * np.cos(0.7 * lat_2d) * np.sin(0.5 * lon_2d), 0.0, None) ** 2
    field = np.stack([base * (1.0 + 0.05 * step) for step in range(time_steps)], axis=0)

    if with_event:
        # A localised burst in the middle third of the window, as in the
        # project's injected synthetic event.
        centre_lat, centre_lon = 10.5, 74.5
        sigma = 0.15
        delta_lat = lat_2d - centre_lat
        delta_lon = lon_2d - centre_lon
        spatial = np.exp(-(delta_lat**2 + delta_lon**2) / (2.0 * sigma**2))
        for step in range(time_steps):
            if 2 <= step < time_steps - 1:
                phase = (step - 2) / max(time_steps - 4, 1)
                field[step] = field[step] + 40.0 * spatial * float(np.sin(np.pi * phase))

    timestamps = [f"2024-06-01T{step:02d}:00:00" for step in range(time_steps)]
    return field, latitude, longitude, timestamps


@pytest.fixture
def ds_coarse_field():
    """A 12-frame coarse field with an injected extreme event."""
    return make_coarse_field()


@pytest.fixture
def tiny_downscaling_dataset(ds_coarse_field):
    """A built downscaling dataset on the tiny grid."""
    field, latitude, longitude, timestamps = ds_coarse_field
    return build_downscaling_dataset(
        field,
        latitude,
        longitude,
        timestamps,
        FINE_RESOLUTION,
        feature_config=FeatureConfig(),
        reference_config=ReferenceConfig(),
    )
