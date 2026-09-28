"""Deterministic *synthetic* weather data generation (SIH26078, Phase 0).

Every value produced by this module is **fully synthetic**. Nothing is
downloaded, mirrored or presented as a real observation or an operational NWP
product: the fields are analytic functions plus seeded pseudo-random structure.
The output is labelled ``synthetic`` both in the NetCDF global attributes and in
the JSON provenance manifest written next to the file.

Why synthetic data?
    Phase 0 must be reproducible everywhere (developer laptop, CI, offline SIH
    demo) and must never require API keys or network access. The synthetic fields
    therefore reproduce the *structure* of a coarse NWP product:

    * dimensions ``time`` x ``latitude`` x ``longitude``,
    * ~12 km native grid spacing (0.125 deg, the reference for the future
      12 km -> 5 km downscaling work),
    * the six core surface variables of the project,
    * one configurable, spatially localised and temporally bounded extreme
      rainfall event so that FUTURE anomaly detection has something to find.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import xarray as xr
from scipy.ndimage import gaussian_filter

from extreme_weather.config import (
    DOWNSAMPLE_TARGET_RESOLUTION_DEG,
    KM_PER_DEGREE_LATITUDE,
    NWP_NATIVE_RESOLUTION_DEG,
    PACKAGE_VERSION,
    PROJECT_ID,
    PROJECT_NAME,
    Settings,
    get_settings,
)
from extreme_weather.data.loader import save_dataset

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
GENERATOR_VERSION: str = PACKAGE_VERSION
DATASET_BASENAME: str = "sample_history"
NETCDF_SUFFIX: str = ".nc"
MANIFEST_SUFFIX: str = ".manifest.json"
SYNTHETIC_SOURCE: str = "synthetic:extreme_weather.data.generator"

#: A data variable is understood as "present" only if it is in this tuple.
VARIABLES: tuple[str, ...] = (
    "precipitation",
    "temperature",
    "pressure",
    "relative_humidity",
    "u_wind",
    "v_wind",
)

VARIABLE_UNITS: dict[str, str] = {
    "precipitation": "mm h-1",
    "temperature": "K",
    "pressure": "Pa",
    "relative_humidity": "%",
    "u_wind": "m s-1",
    "v_wind": "m s-1",
}

VARIABLE_LONG_NAMES: dict[str, str] = {
    "precipitation": "Synthetic precipitation rate",
    "temperature": "Synthetic air temperature at 2 m",
    "pressure": "Synthetic surface pressure",
    "relative_humidity": "Synthetic relative humidity at 2 m",
    "u_wind": "Synthetic eastward wind component at 10 m",
    "v_wind": "Synthetic northward wind component at 10 m",
}

VARIABLE_STANDARD_NAMES: dict[str, str] = {
    "precipitation": "precipitation_flux",
    "temperature": "air_temperature",
    "pressure": "surface_air_pressure",
    "relative_humidity": "relative_humidity",
    "u_wind": "eastward_wind",
    "v_wind": "northward_wind",
}

#: ``radius_deg`` of the injected event is interpreted as roughly 2 sigma of a
#: Gaussian footprint, i.e. ``sigma = radius_deg * SIGMA_FROM_RADIUS``.
SIGMA_FROM_RADIUS: float = 0.5


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO 8601 string (second resolution)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")



# --------------------------------------------------------------------------- #
# Configuration objects
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExtremeEventConfig:
    """Configuration of the injected synthetic extreme rainfall event.

    The event is deliberately simple and explainable: a Gaussian footprint in
    space (centred on ``center_latitude``/``center_longitude``) multiplied by a
    half-sine pulse in time over ``duration_steps`` steps starting at
    ``start_index``. Outside that time window the event contributes exactly zero,
    which makes later anomaly-detection work verifiable.
    """

    enabled: bool = True
    name: str = "synthetic_monsoon_burst"
    center_latitude: float = 13.0
    center_longitude: float = 77.0
    radius_deg: float = 1.0
    start_index: int = 24
    duration_steps: int = 12
    peak_precipitation_mm_per_hour: float = 55.0
    temperature_anomaly_k: float = -4.0
    pressure_anomaly_pa: float = -1500.0
    humidity_anomaly_percent: float = 25.0
    wind_rotation_m_per_s: float = 12.0

    def __post_init__(self) -> None:
        if self.radius_deg <= 0.0:
            raise ValueError(f"radius_deg must be > 0, got {self.radius_deg}.")
        if self.duration_steps < 1:
            raise ValueError(f"duration_steps must be >= 1, got {self.duration_steps}.")
        if self.start_index < 0:
            raise ValueError(f"start_index must be >= 0, got {self.start_index}.")
        if self.peak_precipitation_mm_per_hour < 0.0:
            raise ValueError(
                "peak_precipitation_mm_per_hour must be >= 0, "
                f"got {self.peak_precipitation_mm_per_hour}."
            )
        if not -90.0 <= self.center_latitude <= 90.0:
            raise ValueError(f"center_latitude out of range: {self.center_latitude}.")
        if not -360.0 <= self.center_longitude <= 360.0:
            raise ValueError(f"center_longitude out of range: {self.center_longitude}.")

    def as_dict(self) -> dict[str, Any]:
        """Return the event configuration as a plain dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class SyntheticConfig:
    """Full configuration of a synthetic dataset generation run."""

    seed: int = 20260601
    time_steps: int = 72
    start_time: str = "2024-06-01T00:00:00"
    time_step_hours: int = 1
    latitude_min: float = 10.0
    latitude_max: float = 16.0
    longitude_min: float = 74.0
    longitude_max: float = 80.0
    resolution_deg: float = NWP_NATIVE_RESOLUTION_DEG
    variables: tuple[str, ...] = VARIABLES
    event: ExtremeEventConfig = field(default_factory=ExtremeEventConfig)

    @property
    def latitude(self) -> np.ndarray:
        """Ascending latitude axis in degrees north (endpoints snapped)."""
        return build_axis(self.latitude_min, self.latitude_max, self.resolution_deg, name="latitude")

    @property
    def longitude(self) -> np.ndarray:
        """Ascending longitude axis in degrees east (endpoints snapped)."""
        return build_axis(
            self.longitude_min, self.longitude_max, self.resolution_deg, name="longitude"
        )

    @property
    def time(self) -> np.ndarray:
        """Datetime64[s] time axis derived from ``start_time`` and the step size."""
        start = np.datetime64(self.start_time, "s")
        offsets = np.arange(self.time_steps) * np.timedelta64(self.time_step_hours, "h")
        return start + offsets

    @property
    def shape(self) -> tuple[int, int, int]:
        """Expected ``(time, latitude, longitude)`` shape of the generated fields."""
        return (self.time_steps, self.latitude.size, self.longitude.size)

    def to_manifest_dict(self) -> dict[str, Any]:
        """Return the generation settings (without seed noise arrays)."""
        payload = asdict(self)
        payload["variables"] = list(self.variables)
        payload["dimensions"] = {
            "time": self.time_steps,
            "latitude": int(self.latitude.size),
            "longitude": int(self.longitude.size),
        }
        return payload

    def __post_init__(self) -> None:
        if self.time_steps < 2:
            raise ValueError(f"time_steps must be >= 2, got {self.time_steps}.")
        if self.time_step_hours < 1:
            raise ValueError(f"time_step_hours must be >= 1, got {self.time_step_hours}.")
        if self.resolution_deg <= 0.0:
            raise ValueError(f"resolution_deg must be > 0, got {self.resolution_deg}.")
        if self.latitude_min >= self.latitude_max:
            raise ValueError(
                "latitude_min must be smaller than latitude_max, got "
                f"{self.latitude_min} and {self.latitude_max}."
            )
        if self.longitude_min >= self.longitude_max:
            raise ValueError(
                "longitude_min must be smaller than longitude_max, got "
                f"{self.longitude_min} and {self.longitude_max}."
            )
        if not self.variables:
            raise ValueError("At least one variable must be requested.")
        unknown = [name for name in self.variables if name not in VARIABLES]
        if unknown:
            raise ValueError(f"Unknown synthetic variable(s) requested: {unknown}.")
        if self.event.enabled:
            if self.event.start_index + self.event.duration_steps > self.time_steps:
                raise ValueError(
                    "Event window exceeds the time axis: start_index "
                    f"({self.event.start_index}) + duration_steps "
                    f"({self.event.duration_steps}) > time_steps ({self.time_steps})."
                )
            inside = (
                self.latitude_min <= self.event.center_latitude <= self.latitude_max
                and self.longitude_min <= self.event.center_longitude <= self.longitude_max
            )
            if not inside:
                raise ValueError(
                    "Event centre "
                    f"({self.event.center_latitude}, {self.event.center_longitude}) is outside "
                    f"the domain lat[{self.latitude_min}, {self.latitude_max}] "
                    f"lon[{self.longitude_min}, {self.longitude_max}]. Adjust the event "
                    "configuration or disable the event."
                )



# --------------------------------------------------------------------------- #
# Geometry / noise helpers
# --------------------------------------------------------------------------- #
def build_axis(
    lower: float,
    upper: float,
    resolution: float,
    *,
    name: str = "axis",
) -> np.ndarray:
    """Build a uniform axis from ``lower`` to ``upper`` at ``resolution`` spacing.

    The upper bound is snapped onto ``upper`` exactly so that generated grids
    always cover the requested domain. When ``(upper - lower)`` is not an exact
    multiple of ``resolution`` the final interval is slightly shorter.
    """
    if resolution <= 0.0:
        raise ValueError(f"{name} resolution must be > 0, got {resolution}.")
    if lower >= upper:
        raise ValueError(f"{name} bounds must satisfy lower < upper, got {lower} and {upper}.")

    intervals = max(int(round((upper - lower) / resolution)), 1)
    axis = lower + resolution * np.arange(intervals + 1, dtype="float64")
    axis[-1] = float(upper)

    if axis.size < 2 or not np.all(np.diff(axis) > 0.0):
        raise ValueError(
            f"Could not build a strictly increasing {name} axis for bounds "
            f"[{lower}, {upper}] at resolution {resolution}."
        )
    return axis


def smooth_noise(
    rng: np.random.Generator,
    shape: tuple[int, ...],
    *,
    sigma: tuple[float, ...],
) -> np.ndarray:
    """Return spatially/temporally coherent noise (approximately unit variance).

    A white-noise field is smoothed with a Gaussian filter and standardised. This
    keeps the synthetic fields physically plausible (weather fields are
    correlated in space and time) without needing an expensive simulation.
    """
    raw = rng.standard_normal(shape)
    effective_sigma = tuple(
        float(max(min(s, (size - 1) / 2.0), 0.0)) for s, size in zip(sigma, shape)
    )
    if all(value == 0.0 for value in effective_sigma):
        smoothed = raw
    else:
        smoothed = gaussian_filter(raw, sigma=effective_sigma, mode="reflect")
    standard_deviation = float(np.std(smoothed))
    if standard_deviation < 1e-12:
        return smoothed - float(np.mean(smoothed))
    return (smoothed - float(np.mean(smoothed))) / standard_deviation


def event_field(config: SyntheticConfig) -> np.ndarray:
    """Return the dimensionless (time, latitude, longitude) event footprint.

    Values lie in ``[0, 1]``: 0 means "no event contribution", 1 means "peak of
    the event". Outside the injection window the field is exactly zero, and the
    footprint decays with distance from the event centre.
    """
    event = config.event
    latitude = config.latitude
    longitude = config.longitude
    latitude_2d, longitude_2d = np.meshgrid(latitude, longitude, indexing="ij")

    field = np.zeros((config.time_steps, latitude.size, longitude.size), dtype="float64")
    if not event.enabled:
        return field

    sigma = event.radius_deg * SIGMA_FROM_RADIUS
    delta_lat = latitude_2d - event.center_latitude
    delta_lon = (longitude_2d - event.center_longitude) * np.cos(
        np.deg2rad(event.center_latitude)
    )
    spatial = np.exp(-(delta_lat**2 + delta_lon**2) / (2.0 * sigma**2))

    span = max(event.duration_steps - 1, 1)
    for step in range(event.start_index, event.start_index + event.duration_steps):
        phase = (step - event.start_index) / span
        field[step] = spatial * float(np.sin(np.pi * phase))
    return field


def event_rotation(config: SyntheticConfig, footprint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return cyclonic rotation wind components driven by the event footprint.

    The rotation mimics the low-level circulation of a monsoon depression:
    tangentially oriented winds whose strength follows the event envelope
    (counter-clockwise in the northern hemisphere).
    """
    if not config.event.enabled:
        zeros = np.zeros_like(footprint)
        return zeros, zeros

    event = config.event
    latitude_2d, longitude_2d = np.meshgrid(config.latitude, config.longitude, indexing="ij")
    sigma = event.radius_deg * SIGMA_FROM_RADIUS
    delta_lat = latitude_2d - event.center_latitude
    delta_lon = (longitude_2d - event.center_longitude) * np.cos(
        np.deg2rad(event.center_latitude)
    )

    strength = event.wind_rotation_m_per_s * footprint
    u_component = -strength * (delta_lat / sigma)
    v_component = strength * (delta_lon / sigma)
    return u_component, v_component


# --------------------------------------------------------------------------- #
# Field construction
# --------------------------------------------------------------------------- #
def _build_fields(config: SyntheticConfig) -> dict[str, np.ndarray]:
    """Construct all synthetic fields from the seeded random generator.

    The random draws always happen in the same order and in the same shapes, so
    the background fields are byte-for-byte reproducible - and, importantly,
    identical whether or not the extreme event is enabled.
    """
    rng = np.random.default_rng(config.seed)
    shape = config.shape
    latitude = config.latitude
    longitude = config.longitude
    latitude_2d, longitude_2d = np.meshgrid(latitude, longitude, indexing="ij")

    latitude_span = float(latitude.max() - latitude.min())
    longitude_span = float(longitude.max() - longitude.min())
    normalized_latitude = (latitude_2d - latitude.min()) / latitude_span
    normalized_longitude = (longitude_2d - longitude.min()) / longitude_span
    hours = np.arange(config.time_steps, dtype="float64") * config.time_step_hours
    diurnal = np.sin(2.0 * np.pi * (hours % 24.0) / 24.0)
    diurnal_3d = diurnal[:, np.newaxis, np.newaxis]

    # --- seeded noise fields (fixed order!) ---
    temperature_noise = smooth_noise(rng, shape, sigma=(2.0, 1.5, 1.5))
    pressure_noise = smooth_noise(rng, shape, sigma=(2.0, 2.0, 2.0))
    humidity_noise = smooth_noise(rng, shape, sigma=(1.5, 1.5, 1.5))
    u_noise = smooth_noise(rng, shape, sigma=(1.5, 1.5, 1.5))
    v_noise = smooth_noise(rng, shape, sigma=(1.5, 1.5, 1.5))
    rain_noise = smooth_noise(rng, shape, sigma=(1.0, 1.0, 1.0))

    # --- injected extreme event ---
    footprint = event_field(config)
    rotation_u, rotation_v = event_rotation(config, footprint)
    event = config.event

    # --- temperature (K): warm south-to-north gradient + diurnal cycle + noise ---
    temperature = (
        303.0
        - 0.55 * (latitude_2d - latitude.min())
        + 3.5 * diurnal_3d
        - 1.2 * temperature_noise
        + event.temperature_anomaly_k * footprint
    )

    # --- surface pressure (Pa) ---
    pressure = (
        100_900.0
        - 450.0 * normalized_latitude
        + 320.0 * pressure_noise
        + event.pressure_anomaly_pa * footprint
    )

    # --- relative humidity (%) ---
    relative_humidity = np.clip(
        72.0
        + 12.0 * np.sin(2.0 * np.pi * normalized_longitude)
        - 8.0 * normalized_latitude
        + 6.0 * humidity_noise
        + 3.0 * diurnal_3d
        + event.humidity_anomaly_percent * footprint,
        15.0,
        99.0,
    )

    # --- wind components (m/s) ---
    u_wind = (
        4.5 * np.sin(2.0 * np.pi * normalized_latitude)
        + 1.2 * diurnal_3d
        + 2.6 * u_noise
        + rotation_u
    )
    v_wind = -3.0 * np.cos(2.0 * np.pi * normalized_longitude) + 2.6 * v_noise + rotation_v
    u_wind = np.clip(u_wind, -45.0, 45.0)
    v_wind = np.clip(v_wind, -45.0, 45.0)

    # --- precipitation (mm/h): positively skewed background + event core ---
    background_rain = 1.8 * np.clip(0.25 + 0.45 * rain_noise, 0.0, None) ** 2
    precipitation = background_rain + event.peak_precipitation_mm_per_hour * footprint

    fields = {
        "precipitation": precipitation,
        "temperature": temperature,
        "pressure": pressure,
        "relative_humidity": relative_humidity,
        "u_wind": u_wind,
        "v_wind": v_wind,
    }
    return {name: fields[name] for name in config.variables}


# --------------------------------------------------------------------------- #
# Dataset assembly
# --------------------------------------------------------------------------- #
def dataset_attributes(config: SyntheticConfig, *, created_utc: str | None = None) -> dict[str, Any]:
    """Return the CF-style global attributes describing a synthetic dataset."""
    timestamp = created_utc or utc_now_iso()
    return {
        "title": f"{PROJECT_NAME} synthetic weather history ({PROJECT_ID})",
        "summary": (
            "Deterministic SYNTHETIC gridded weather fields over time x latitude x longitude. "
            "These are not observations and not an operational NWP product; they exist to make "
            "the Phase 0 pipeline reproducible without network access or API keys."
        ),
        "source": SYNTHETIC_SOURCE,
        "dataset_type": "synthetic",
        "project": PROJECT_ID,
        "project_name": PROJECT_NAME,
        "generator_version": GENERATOR_VERSION,
        "random_seed": int(config.seed),
        "grid_resolution_deg": float(config.resolution_deg),
        "approximate_grid_resolution_km": round(config.resolution_deg * KM_PER_DEGREE_LATITUDE, 2),
        "nwp_native_reference_resolution_deg": float(NWP_NATIVE_RESOLUTION_DEG),
        "downscaling_target_resolution_deg": float(DOWNSAMPLE_TARGET_RESOLUTION_DEG),
        "event_injection": json.dumps(config.event.as_dict(), sort_keys=True),
        "created_utc": timestamp,
        "Conventions": "CF-1.8",
        "history": (
            f"{timestamp} - generated by extreme_weather.data.generator "
            f"{GENERATOR_VERSION} (seed={config.seed})"
        ),
    }


def generate_synthetic_dataset(
    config: SyntheticConfig | None = None, *, created_utc: str | None = None
) -> xr.Dataset:
    """Generate the deterministic synthetic weather dataset.

    Args:
        config: Generation configuration; defaults to :func:`default_config`.
        created_utc: Override the provenance timestamp (useful for reproducible
            manifests and tests).

    Returns:
        An in-memory :class:`xarray.Dataset` with dimensions
        ``(time, latitude, longitude)`` and float32 data variables.
    """
    effective_config = config or default_config()
    fields = _build_fields(effective_config)
    latitude = effective_config.latitude
    longitude = effective_config.longitude
    time = effective_config.time

    data_vars = {
        name: (
            ("time", "latitude", "longitude"),
            values.astype("float32"),
            {
                "units": VARIABLE_UNITS[name],
                "long_name": VARIABLE_LONG_NAMES[name],
                "standard_name": VARIABLE_STANDARD_NAMES[name],
                "source": "synthetic",
                "project": PROJECT_ID,
            },
        )
        for name, values in fields.items()
    }

    dataset = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": (
                "time",
                time,
                {"standard_name": "time", "long_name": "time", "axis": "T"},
            ),
            "latitude": (
                "latitude",
                latitude,
                {
                    "units": "degrees_north",
                    "standard_name": "latitude",
                    "long_name": "latitude",
                    "axis": "Y",
                },
            ),
            "longitude": (
                "longitude",
                longitude,
                {
                    "units": "degrees_east",
                    "standard_name": "longitude",
                    "long_name": "longitude",
                    "axis": "X",
                },
            ),
        },
        attrs=dataset_attributes(effective_config, created_utc=created_utc),
    )
    dataset.attrs["synthetic_variables"] = ", ".join(effective_config.variables)
    return dataset


# --------------------------------------------------------------------------- #
# Provenance manifest
# --------------------------------------------------------------------------- #
def manifest_path_for(dataset_path: Path | str) -> Path:
    """Return the provenance manifest path that belongs to ``dataset_path``."""
    path = Path(dataset_path).expanduser()
    return path.parent / f"{path.stem}{MANIFEST_SUFFIX}"


def file_sha256(path: Path | str, *, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA-256 checksum of a file (streamed, memory friendly)."""
    digest = hashlib.sha256()
    with Path(path).expanduser().open("rb") as handle:
        for block in iter(lambda: handle.read(chunk_size), b""):
            digest.update(block)
    return digest.hexdigest()


def build_manifest(
    config: SyntheticConfig,
    dataset: xr.Dataset,
    *,
    dataset_path: Path | str | None = None,
    created_utc: str | None = None,
    checksum: str | None = None,
) -> dict[str, Any]:
    """Build the JSON provenance manifest for a generated dataset.

    The manifest is the machine readable record of *how* the data was produced:
    generator version, seed, dimensions, grid resolution, injector settings and
    (when a file was written) its size and SHA-256 checksum. It explicitly states
    that the data is synthetic.
    """
    latitude = config.latitude
    longitude = config.longitude
    time = config.time
    path = Path(dataset_path).expanduser() if dataset_path is not None else None

    return {
        "manifest_version": "1.0",
        "project": {"id": PROJECT_ID, "name": PROJECT_NAME},
        "dataset": {
            "name": path.stem if path is not None else DATASET_BASENAME,
            "file": path.name if path is not None else None,
            "file_bytes": path.stat().st_size if path is not None and path.exists() else None,
            "sha256": checksum,
            "synthetic": True,
            "source": SYNTHETIC_SOURCE,
            "statement": (
                "SYNTHETIC DATA. Generated locally by a deterministic algorithm; it is not "
                "observed weather data and not an operational NWP product."
            ),
        },
        "generator": {
            "module": "extreme_weather.data.generator",
            "version": GENERATOR_VERSION,
        },
        "random_seed": int(config.seed),
        "created_utc": created_utc or utc_now_iso(),
        "variables": [
            {
                "name": name,
                "units": VARIABLE_UNITS[name],
                "long_name": VARIABLE_LONG_NAMES[name],
                "standard_name": VARIABLE_STANDARD_NAMES[name],
                "dtype": str(dataset[name].dtype) if name in dataset else "float32",
                "dimensions": list(dataset[name].dims) if name in dataset else [],
            }
            for name in config.variables
        ],
        "dimensions": {
            "time": int(time.size),
            "latitude": int(latitude.size),
            "longitude": int(longitude.size),
        },
        "coordinates": {
            "time": {
                "start": str(np.datetime_as_string(np.asarray(time[0]), unit="s")),
                "end": str(np.datetime_as_string(np.asarray(time[-1]), unit="s")),
                "count": int(time.size),
                "step_hours": int(config.time_step_hours),
            },
            "latitude": {
                "min": float(latitude.min()),
                "max": float(latitude.max()),
                "count": int(latitude.size),
                "resolution_deg": float(config.resolution_deg),
                "units": "degrees_north",
            },
            "longitude": {
                "min": float(longitude.min()),
                "max": float(longitude.max()),
                "count": int(longitude.size),
                "resolution_deg": float(config.resolution_deg),
                "units": "degrees_east",
            },
        },
        "grid_resolution": {
            "degrees": float(config.resolution_deg),
            "approximate_km": round(config.resolution_deg * KM_PER_DEGREE_LATITUDE, 3),
            "nwp_native_reference_deg": float(NWP_NATIVE_RESOLUTION_DEG),
            "downscaling_target_reference_deg": float(DOWNSAMPLE_TARGET_RESOLUTION_DEG),
        },
        "event_injection": config.event.as_dict(),
        "generation_settings": config.to_manifest_dict(),
        "dataset_attributes": {
            str(key): (value.item() if hasattr(value, "item") else value)
            for key, value in dataset.attrs.items()
        },
    }


def write_manifest(manifest: Mapping[str, Any], path: Path | str) -> Path:
    """Write ``manifest`` to ``path`` as UTF-8 JSON, creating directories first."""
    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(manifest), handle, indent=2, ensure_ascii=False, default=str)
        handle.write("\n")
    return output_path


# --------------------------------------------------------------------------- #
# Convenience entry points
# --------------------------------------------------------------------------- #
@dataclass
class GenerationResult:
    """Everything produced by :func:`generate_and_save`."""

    dataset: xr.Dataset
    config: SyntheticConfig
    dataset_path: Path
    manifest_path: Path | None
    manifest: dict[str, Any]


def default_config(
    *,
    seed: int | None = None,
    time_steps: int | None = None,
    resolution_deg: float | None = None,
    latitude_bounds: tuple[float, float] | None = None,
    longitude_bounds: tuple[float, float] | None = None,
    event_enabled: bool | None = None,
    event_overrides: Mapping[str, Any] | None = None,
    settings: Settings | None = None,
) -> SyntheticConfig:
    """Build a :class:`SyntheticConfig` from environment defaults plus overrides.

    Only explicitly provided arguments override the environment defaults, which
    keeps ``APP_ENV``/``.env`` driven tuning possible. The event window defaults
    to one third of the series with a length of one sixth of the series, and the
    event centre defaults to the centre of the domain unless overridden.
    """
    effective_settings = settings or get_settings()
    effective_time_steps = (
        int(time_steps) if time_steps is not None else effective_settings.synthetic_time_steps
    )
    bounds_lat = latitude_bounds or (
        effective_settings.synthetic_latitude_min,
        effective_settings.synthetic_latitude_max,
    )
    bounds_lon = longitude_bounds or (
        effective_settings.synthetic_longitude_min,
        effective_settings.synthetic_longitude_max,
    )

    event_overrides = dict(event_overrides or {})
    default_event = ExtremeEventConfig()
    event_kwargs: dict[str, Any] = {
        "enabled": (
            bool(event_enabled)
            if event_enabled is not None
            else effective_settings.synthetic_event_enabled
        ),
        "peak_precipitation_mm_per_hour": effective_settings.synthetic_event_peak_mm_per_hour,
        # Keep the default event centre when it is inside the domain, otherwise
        # fall back to the domain centre so the injected event stays visible.
        "center_latitude": (
            default_event.center_latitude
            if bounds_lat[0] <= default_event.center_latitude <= bounds_lat[1]
            else round((bounds_lat[0] + bounds_lat[1]) / 2.0, 6)
        ),
        "center_longitude": (
            default_event.center_longitude
            if bounds_lon[0] <= default_event.center_longitude <= bounds_lon[1]
            else round((bounds_lon[0] + bounds_lon[1]) / 2.0, 6)
        ),
        "start_index": max(int(round(effective_time_steps / 3)), 0),
        "duration_steps": max(int(round(effective_time_steps / 6)), 1),
    }
    event_kwargs.update(event_overrides)

    return SyntheticConfig(
        seed=int(seed) if seed is not None else effective_settings.synthetic_seed,
        time_steps=effective_time_steps,
        start_time=effective_settings.synthetic_start_time,
        time_step_hours=effective_settings.synthetic_time_step_hours,
        latitude_min=float(bounds_lat[0]),
        latitude_max=float(bounds_lat[1]),
        longitude_min=float(bounds_lon[0]),
        longitude_max=float(bounds_lon[1]),
        resolution_deg=(
            float(resolution_deg)
            if resolution_deg is not None
            else effective_settings.synthetic_grid_resolution_deg
        ),
        event=ExtremeEventConfig(**event_kwargs),
    )


def generate_and_save(
    output_path: Path | str | None = None,
    *,
    config: SyntheticConfig | None = None,
    write_manifest_file: bool = True,
    created_utc: str | None = None,
    compression_level: int = 4,
) -> GenerationResult:
    """Generate the synthetic dataset, write NetCDF (+ manifest) and return it.

    Args:
        output_path: Destination ``.nc`` path. Defaults to
            ``<DATA_DIR>/raw/sample_history.nc`` (``data/raw/sample_history.nc``).
        config: Generation configuration; defaults to :func:`default_config`.
        write_manifest_file: Also write ``<name>.manifest.json`` with provenance.
        created_utc: Override the provenance timestamp (reproducible manifests).
        compression_level: zlib compression level used when saving.
    """
    effective_config = config or default_config()
    dataset_file = (
        Path(output_path).expanduser()
        if output_path is not None
        else get_settings().raw_dir / f"{DATASET_BASENAME}{NETCDF_SUFFIX}"
    )

    dataset = generate_synthetic_dataset(effective_config, created_utc=created_utc)
    saved_path = save_dataset(dataset, dataset_file, compression_level=compression_level)
    manifest = build_manifest(
        effective_config,
        dataset,
        dataset_path=saved_path,
        created_utc=created_utc,
        checksum=file_sha256(saved_path),
    )
    manifest_file = (
        write_manifest(manifest, manifest_path_for(saved_path)) if write_manifest_file else None
    )

    return GenerationResult(
        dataset=dataset,
        config=effective_config,
        dataset_path=saved_path,
        manifest_path=manifest_file,
        manifest=manifest,
    )

