"""FastAPI application package for DeepWeather AI (SIH26078).

The application itself lives in :mod:`extreme_weather.api.main` and is started
with::

    uvicorn extreme_weather.api.main:app --reload

This package initialiser intentionally imports nothing, so that importing
``extreme_weather.api`` never has the side effect of building the application.
"""

from __future__ import annotations

__all__: list[str] = []
