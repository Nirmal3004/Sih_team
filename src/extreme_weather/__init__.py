"""DeepWeather AI (SIH26078) - extreme weather intelligence toolkit.

Phase 0 scope
-------------
This package currently implements a reproducible foundation only:

* deterministic synthetic weather data generation (no API keys, no network),
* Xarray based loading/validation,
* missing-value handling with ``max_gap`` semantics,
* derived weather variables (wind speed / direction),
* a preprocessing pipeline (load -> validate -> missing -> derived -> save),
* baseline spatial downscaling with a method registry,
* a minimal FastAPI service and JSON schemas.

Anomaly detection (``extreme_weather.anomaly``) and event tracking
(``extreme_weather.tracking``) are intentionally placeholder packages in
Phase 0 and are marked FUTURE throughout the documentation.
"""

from __future__ import annotations

from extreme_weather.config import (
    DOWNSAMPLE_TARGET_RESOLUTION_DEG,
    NWP_NATIVE_RESOLUTION_DEG,
    PROJECT_ID,
    PROJECT_NAME,
    Settings,
    get_settings,
)

__version__: str = "0.1.0"

__all__ = [
    "DOWNSAMPLE_TARGET_RESOLUTION_DEG",
    "NWP_NATIVE_RESOLUTION_DEG",
    "PROJECT_ID",
    "PROJECT_NAME",
    "Settings",
    "__version__",
    "get_settings",
]
