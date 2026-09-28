"""Pydantic response/request schemas exposed by the FastAPI layer (SIH26078).

These models are the *external* contract of the service. They are intentionally
decoupled from the internal dataclasses (``extreme_weather.data.validator``) so
that internal refactoring does not silently change the public JSON API.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class VariableInfoResponse(BaseModel):
    """Metadata of one data variable."""

    model_config = ConfigDict(extra="forbid")

    name: str
    dimensions: list[str]
    shape: list[int]
    dtype: str
    units: str | None = None
    long_name: str | None = None


class DatasetSummaryResponse(BaseModel):
    """Metadata-only summary of a NetCDF dataset (no array payloads)."""

    model_config = ConfigDict(extra="forbid")

    name: str
    path: str | None = None
    dimensions: dict[str, int]
    data_variables: list[str]
    variables: list[VariableInfoResponse]
    time_steps: int
    time_start: str | None = None
    time_end: str | None = None
    time_step_seconds: float | None = None
    latitude_min: float | None = None
    latitude_max: float | None = None
    latitude_points: int
    longitude_min: float | None = None
    longitude_max: float | None = None
    longitude_points: int
    grid_resolution_deg: float | None = None
    approximate_grid_resolution_km: float | None = None
    is_synthetic: bool
    attributes: dict[str, Any] = Field(default_factory=dict)


class DatasetFileInfo(BaseModel):
    """A NetCDF file discovered in the configured data directory."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="Path of the file relative to the data directory.")
    subdirectory: str = Field(description="raw / processed / samples / other")
    size_bytes: int
    modified_utc: str


class DatasetListResponse(BaseModel):
    """Result of ``GET /api/v1/datasets``."""

    model_config = ConfigDict(extra="forbid")

    data_dir: str
    count: int
    datasets: list[DatasetFileInfo]


class DatasetValidationRequest(BaseModel):
    """Body of ``POST /api/v1/datasets/validate``."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(description="Dataset path relative to the data directory, e.g. raw/sample_history.nc")


class DatasetValidationResponse(BaseModel):
    """Result of a dataset validation request."""

    model_config = ConfigDict(extra="forbid")

    valid: bool
    path: str
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
