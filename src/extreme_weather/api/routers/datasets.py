"""Dataset metadata endpoints (SIH26078, Phase 0).

Phase 0 deliberately exposes *structure*, not processing:

``GET  /api/v1/datasets``          - list NetCDF files in the data directory
``GET  /api/v1/datasets/{path}/summary`` - validated metadata of one dataset
``POST /api/v1/datasets/validate`` - dry-run the data contract against a file

Heavy operations (anomaly detection, tracking, on-demand downscaling, tile
rendering) belong to FUTURE phases and are not exposed yet.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import xarray as xr
from fastapi import APIRouter, Depends, HTTPException, status

from extreme_weather.config import Settings, get_settings
from extreme_weather.data.loader import DatasetLoadError, resolve_dataset_path
from extreme_weather.data.validator import collect_validation_issues, summarise_dataset
from extreme_weather.schemas.dataset import (
    DatasetFileInfo,
    DatasetListResponse,
    DatasetSummaryResponse,
    DatasetValidationRequest,
    DatasetValidationResponse,
)

router = APIRouter(prefix="/api/v1/datasets", tags=["datasets"])

NETCDF_SUFFIX = ".nc"


def _list_netcdf_files(data_dir: Path) -> list[DatasetFileInfo]:
    """Return every ``.nc`` file below ``data_dir`` with size and mtime."""
    if not data_dir.exists():
        return []

    entries: list[DatasetFileInfo] = []
    for path in sorted(data_dir.rglob(f"*{NETCDF_SUFFIX}")):
        if not path.is_file():
            continue
        relative = path.relative_to(data_dir)
        subdirectory = relative.parts[0] if len(relative.parts) > 1 else ""
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
        entries.append(
            DatasetFileInfo(
                name=relative.as_posix(),
                subdirectory=subdirectory,
                size_bytes=path.stat().st_size,
                modified_utc=modified.replace(microsecond=0).isoformat(),
            )
        )
    return entries


@router.get("", response_model=DatasetListResponse, summary="List available datasets")
def list_datasets(settings: Settings = Depends(get_settings)) -> DatasetListResponse:
    """List the NetCDF datasets found under the configured data directory."""
    data_dir = settings.data_dir
    datasets = _list_netcdf_files(data_dir)
    return DatasetListResponse(
        data_dir=str(data_dir),
        count=len(datasets),
        datasets=datasets,
    )


@router.post(
    "/validate",
    response_model=DatasetValidationResponse,
    summary="Validate a dataset against the Phase 0 data contract",
)
def validate_dataset_endpoint(
    payload: DatasetValidationRequest,
    settings: Settings = Depends(get_settings),
) -> DatasetValidationResponse:
    """Open a dataset and report the data contract findings (no processing)."""
    try:
        dataset_path = resolve_dataset_path(settings.data_dir, payload.path)
    except DatasetLoadError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not dataset_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{payload.path}' was not found in '{settings.data_dir}'.",
        )

    try:
        with xr.open_dataset(dataset_path) as opened:
            issues = collect_validation_issues(opened.load())
    except Exception as exc:  # noqa: BLE001 - reported as an API error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not read dataset '{payload.path}': {type(exc).__name__}: {exc}",
        ) from exc

    return DatasetValidationResponse(
        valid=issues.ok,
        path=payload.path,
        errors=list(issues.errors),
        warnings=list(issues.warnings),
    )


@router.get(
    "/{dataset_path:path}/summary",
    response_model=DatasetSummaryResponse,
    summary="Metadata summary of one dataset",
)
def dataset_summary(
    dataset_path: str,
    settings: Settings = Depends(get_settings),
) -> DatasetSummaryResponse:
    """Return validated metadata (dims, variables, coordinates, provenance)."""
    try:
        resolved = resolve_dataset_path(settings.data_dir, dataset_path)
    except DatasetLoadError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    if not resolved.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{dataset_path}' was not found in '{settings.data_dir}'.",
        )

    try:
        with xr.open_dataset(resolved) as opened:
            summary = summarise_dataset(opened, name=Path(dataset_path).stem, path=dataset_path)
    except Exception as exc:  # noqa: BLE001 - reported as an API error
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not read dataset '{dataset_path}': {type(exc).__name__}: {exc}",
        ) from exc

    return DatasetSummaryResponse(**summary.as_dict())
