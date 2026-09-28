"""Central configuration for DeepWeather AI (SIH26078).

All runtime configuration is resolved from environment variables (optionally via
a ``.env`` file, see ``.env.example``). No machine-specific paths and no secrets
are hardcoded: every path is relative to the process working directory unless an
absolute value is supplied through the environment.

Typical usage::

    from extreme_weather.config import get_settings

    settings = get_settings()
    settings.ensure_directories()
    print(settings.raw_dir)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# --------------------------------------------------------------------------- #
# Static project metadata
# --------------------------------------------------------------------------- #
PROJECT_ID: str = "SIH26078"
PROJECT_NAME: str = "DeepWeather AI"
PACKAGE_VERSION: str = "0.1.0"

#: Kilometres per degree of latitude (used only for human readable reporting).
KM_PER_DEGREE_LATITUDE: float = 111.32

#: Reference resolution of the coarse (NWP-style) input grid: 0.125 deg ~= 12 km.
NWP_NATIVE_RESOLUTION_DEG: float = 0.125

#: Reference resolution of the high-resolution target grid: 0.05 deg ~= 5 km.
#: This is the target of the FUTURE 12 km -> 5 km downscaling phase.
DOWNSAMPLE_TARGET_RESOLUTION_DEG: float = 0.05


def _parse_bounds(value: str) -> tuple[float, float] | None:
    """Parse a ``"min,max"`` bounds string into a tuple, or ``None`` if empty.

    Raises:
        ValueError: If the string is not exactly two comma-separated numbers.
    """
    text = str(value).strip()
    if not text:
        return None
    parts = [part.strip() for part in text.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected bounds as 'min,max', got '{value}'.")
    low, high = float(parts[0]), float(parts[1])
    if low > high:
        raise ValueError(f"Bounds minimum {low} exceeds maximum {high}.")
    return (low, high)


class Settings(BaseSettings):
    """Environment-driven application settings.

    Field names map case-insensitively onto environment variables, therefore
    ``APP_ENV`` populates :attr:`app_env` and ``DATA_DIR`` populates
    :attr:`data_dir`.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Application ---
    app_env: str = "development"
    log_level: str = "INFO"

    # --- Paths ---
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    raw_data_subdir: str = "raw"
    processed_data_subdir: str = "processed"
    samples_data_subdir: str = "samples"

    # --- API ---
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- Synthetic data generator defaults ---
    synthetic_seed: int = 20260601
    synthetic_time_steps: int = Field(default=72, ge=2)
    synthetic_start_time: str = "2024-06-01T00:00:00"
    synthetic_time_step_hours: int = Field(default=1, ge=1)
    synthetic_grid_resolution_deg: float = Field(default=NWP_NATIVE_RESOLUTION_DEG, gt=0.0)
    synthetic_latitude_min: float = Field(default=10.0, ge=-90.0, le=90.0)
    synthetic_latitude_max: float = Field(default=16.0, ge=-90.0, le=90.0)
    synthetic_longitude_min: float = Field(default=74.0, ge=-360.0, le=360.0)
    synthetic_longitude_max: float = Field(default=80.0, ge=-360.0, le=360.0)
    synthetic_event_enabled: bool = True
    synthetic_event_peak_mm_per_hour: float = Field(default=55.0, ge=0.0)

    # --- Downscaling reference resolutions (FUTURE 12 km -> 5 km work) ---
    nwp_native_resolution_deg: float = Field(default=NWP_NATIVE_RESOLUTION_DEG, gt=0.0)
    downsample_target_resolution_deg: float = Field(
        default=DOWNSAMPLE_TARGET_RESOLUTION_DEG, gt=0.0
    )

    # --- Anomaly detection (Phase 1) ---
    events_filename: str = "events.json"
    anomaly_default_score: str = "absolute"
    anomaly_default_grouping: str = "month"

    # --- Event tracking (Phase 2) ---
    tracks_filename: str = "tracks.json"

    # --- Alerting and emergency response (Phase 5) ---
    alerts_filename: str = "alerts.json"
    notifications_filename: str = "notifications.json"
    alert_generation_enabled: bool = True
    alert_default_validity_minutes: int = Field(default=360, ge=1)
    alert_deduplication_window_minutes: int = Field(default=60, ge=1)

    # --- Real / historical data ingestion (Phase 6) ---
    # Sub-directory of DATA_DIR/raw that holds REAL ingested data. Synthetic
    # data stays in <raw_dir>/synthetic and the two are never mixed silently.
    real_data_subdir: str = "real"
    synthetic_data_subdir: str = "synthetic"
    # Manifest directory (one JSON per ingested dataset).
    manifests_subdir: str = "manifests"
    # Optional ground-truth / event-reference datasets.
    reference_subdir: str = "reference"
    # Suffix appended to a processed real dataset.
    processed_real_suffix: str = "_processed.nc"
    # Where a remote source caches what it downloads.
    cache_subdir: str = "cache"
    # Hard ceiling on a downloaded file, in megabytes. A regional ERA5 subset
    # is a few MB; a global field is tens of GB. The default keeps a
    # mistyped request from filling the disk.
    max_download_mb: int = Field(default=512, ge=1)
    # Only these variable names are kept when an ingestion request asks for a
    # subset. Empty means "keep whatever the source provides".
    ingestion_default_variables: str = "precipitation,temperature,pressure,relative_humidity,u_wind,v_wind"
    # Latitude/longitude bounds applied when a request does not state its own.
    # Empty means "no spatial subset" (the whole source grid is kept).
    ingestion_latitude_bounds: str = ""
    ingestion_longitude_bounds: str = ""

    # --- Phase 10: operations, security and observability ---------------
    log_format: Literal["text", "json"] = Field(
        default="text",
        description="Log rendering: 'text' for humans, 'json' for a log shipper.",
    )
    api_request_id_header: str = Field(
        default="X-Request-ID",
        description="Header carrying the request correlation id, in and out.",
    )
    api_max_detail_chars: int = Field(
        default=2000,
        ge=0,
        description="Longest error message returned to a client.",
    )
    enforce_data_dir_boundary: bool = Field(
        default=True,
        description=(
            "Refuse any caller-supplied path that resolves outside DATA_DIR. "
            "Leave enabled outside a deliberately unusual deployment."
        ),
    )
    verify_artifact_integrity: bool = Field(
        default=True,
        description=(
            "Hash model checkpoints and manifests before use, so a corrupt "
            "artifact is reported rather than loaded."
        ),
    )
    min_free_disk_bytes: int = Field(
        default=268435456,
        ge=0,
        description=(
            "Free space below which readiness reports the filesystem as "
            "degraded. 256 MiB."
        ),
    )
    allow_cors_wildcard: bool = Field(
        default=False,
        description=(
            "Permit a '*' CORS origin. Off by default: a wildcard origin lets "
            "any site read this API from a browser."
        ),
    )

    # ------------------------------------------------------------------ #
    # Derived helpers
    # ------------------------------------------------------------------ #
    @property
    def cors_origin_list(self) -> list[str]:
        """Return the allowed CORS origins as a list of stripped strings."""
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def raw_dir(self) -> Path:
        """Directory holding unprocessed (raw) NetCDF datasets."""
        return self.data_dir / self.raw_data_subdir

    @property
    def processed_dir(self) -> Path:
        """Directory holding preprocessed NetCDF datasets."""
        return self.data_dir / self.processed_data_subdir

    @property
    def samples_dir(self) -> Path:
        """Directory holding small dataset samples for demos/tests."""
        return self.data_dir / self.samples_data_subdir

    @property
    def real_raw_dir(self) -> Path:
        """Directory holding REAL ingested data (never mixed with synthetic)."""
        return self.raw_dir / self.real_data_subdir

    @property
    def synthetic_raw_dir(self) -> Path:
        """Directory holding the project's own generated data."""
        return self.raw_dir / self.synthetic_data_subdir

    @property
    def manifests_dir(self) -> Path:
        """Directory holding one JSON manifest per ingested dataset."""
        return self.data_dir / self.manifests_subdir

    @property
    def reference_dir(self) -> Path:
        """Directory holding optional ground-truth / event reference datasets."""
        return self.data_dir / self.reference_subdir

    @property
    def cache_dir(self) -> Path:
        """Directory where remote sources cache what they download."""
        return self.data_dir / self.cache_subdir

    @property
    def ingestion_variable_list(self) -> list[str]:
        """The variable whitelist applied when a request states none."""
        return [name.strip() for name in self.ingestion_default_variables.split(",") if name.strip()]

    @property
    def ingestion_latitude_range(self) -> tuple[float, float] | None:
        """Default latitude bounds, or ``None`` for no spatial subset."""
        return _parse_bounds(self.ingestion_latitude_bounds)

    @property
    def ingestion_longitude_range(self) -> tuple[float, float] | None:
        """Default longitude bounds, or ``None`` for no spatial subset."""
        return _parse_bounds(self.ingestion_longitude_bounds)

    @property
    def is_production(self) -> bool:
        """True when ``APP_ENV`` is set to a production-like value."""
        return self.app_env.strip().lower() in {"prod", "production"}

    def ensure_directories(self) -> None:
        """Create the data/model directories if they do not exist yet.

        Safe to call repeatedly (``exist_ok=True``). Sub-directories are created
        so that scripts and the API never have to guess about the layout.
        """
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.samples_dir,
            self.model_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide cached :class:`Settings` instance.

    The cache keeps configuration reads cheap inside request handlers and
    scripts. Tests (or tools that change the environment at runtime) can reset
    it with ``get_settings.cache_clear()``.
    """
    return Settings()
