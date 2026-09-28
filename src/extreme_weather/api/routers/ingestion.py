"""Data ingestion and dataset-registry endpoints (SIH26078, Phase 6).

These endpoints expose the ingestion layer: the available sources, the datasets
that have been ingested, their quality reports, and the ability to validate,
ingest or process a file.

```text
GET  /api/v1/data/sources                        # registered sources + availability
GET  /api/v1/data/datasets                       # ingested datasets (filters)
GET  /api/v1/data/datasets/{dataset_id}          # one manifest
GET  /api/v1/data/datasets/{dataset_id}/quality  # its quality report
POST /api/v1/data/validate                       # quality-check a path, write nothing
POST /api/v1/data/ingest                         # fetch -> normalise -> manifest
POST /api/v1/data/datasets/{dataset_id}/process  # Phase 0 preprocessing
```

Status codes
------------
=====  ==========================================================
200     Success.
404     Unknown dataset id, or the named file does not exist.
422     Invalid request body, or a dataset fails a blocking quality
        check while ``strict`` is set.
503     A remote source is unreachable, or needs credentials that
        are not configured.
=====  ==========================================================

Two things this router deliberately does not do:

* **It never exposes filesystem internals.** Paths are resolved inside the
  configured data directory and a path that escapes it is rejected, mirroring
  the Phase 0 dataset API's traversal guard.
* **It never invents provenance.** A local file is reported as
  ``source="local_file"`` with no URL, and the response always carries
  ``is_synthetic`` and ``origin`` so a client cannot present real data as
  synthetic or vice versa.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, status as http_status
from pydantic import BaseModel, ConfigDict, Field

from extreme_weather.config import Settings, get_settings
from extreme_weather.ingestion import (
    DataOrigin,
    DataQualityReport,
    DatasetManifest,
    DatasetRegistry,
    IngestionError,
    IngestionRequest,
    IngestionService,
    SourceInfo,
    SourceNotFoundError,
    check_quality,
    default_sources,
)
from extreme_weather.ingestion.netcdf import NetCDFSource
from extreme_weather.ingestion.quality import DataQualityReport as QualityReport
from extreme_weather.schemas.dataset import DatasetValidationRequest

router = APIRouter(prefix="/api/v1/data", tags=["data"])

#: Sentinel returned when no ingest result has been produced in this process.
_NO_INGEST = "No ingestion has been run in this process yet. POST /api/v1/data/ingest first."


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #
class ProvenanceStepOut(BaseModel):
    """One provenance step, as returned by the API."""

    model_config = ConfigDict(extra="forbid")

    stage: str
    tool: str
    version: str = ""
    input_hash: str | None = None
    output_hash: str | None = None
    configuration: dict[str, Any] = Field(default_factory=dict)
    performed_at: str | None = None
    details: str = ""


class DatasetManifestOut(BaseModel):
    """A manifest, flattened for the API.

    A dedicated response model rather than reusing the dataclass, because the
    dataclass exposes ``is_synthetic``/``is_live`` as *properties* and a
    Pydantic model built from it would silently drop them - and those two flags
    are exactly what a client must not lose.
    """

    model_config = ConfigDict(extra="allow")

    dataset_id: str
    source: str
    source_url: str | None = None
    retrieved_at: str
    file_name: str = ""
    file_hash: str = ""
    format: str = ""
    origin: str
    is_synthetic: bool
    is_live: bool = False
    status_label: str = ""
    variables: list[str] = Field(default_factory=list)
    variables_missing: list[str] = Field(default_factory=list)
    dimensions: dict[str, int] = Field(default_factory=dict)
    resolution_deg: float | None = None
    resolution_km: float | None = None
    time_start: str | None = None
    time_end: str | None = None
    time_step_seconds: float | None = None
    coordinate_system: str = "latitude_longitude"
    units: dict[str, str] = Field(default_factory=dict)
    unit_conversions: dict[str, Any] = Field(default_factory=dict)
    coordinate_actions: list[str] = Field(default_factory=list)
    quality_summary: dict[str, Any] = Field(default_factory=dict)
    processing_steps: list[ProvenanceStepOut] = Field(default_factory=list)
    license: str | None = None
    notes: str = ""


class SourceListResponse(BaseModel):
    """Result of ``GET /data/sources``."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    sources: list[SourceInfo]


class QualityResponse(BaseModel):
    """A quality report, as returned by validate/quality/ingest endpoints.

    ``extra="allow"`` so the full :meth:`DataQualityReport.as_dict` payload
    survives: the detailed counters (per-variable NaN fractions, duplicate
    counts) are exactly what a quality panel needs, and dropping them would
    force the client to re-request.
    """

    model_config = ConfigDict(extra="allow")

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_fraction: float = 0.0
    time_steps: int = 0
    spatial_points: int = 0
    variables: list[str] = Field(default_factory=list)


class DatasetListResponse(BaseModel):
    """Result of ``GET /data/datasets``."""

    model_config = ConfigDict(extra="forbid")

    count: int = Field(ge=0)
    datasets: list[DatasetManifestOut]
    message: str | None = None


class ValidateRequest(BaseModel):
    """Body of ``POST /data/validate``."""

    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, description="Dataset path inside the data directory.")
    strict_ranges: bool = Field(
        default=False, description="Treat implausible values as errors, not warnings."
    )


class IngestRequest(BaseModel):
    """Body of ``POST /data/ingest``."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(min_length=1, max_length=200)
    path: str = Field(min_length=1, description="Local NetCDF/GRIB file inside the data dir.")
    origin: str = Field(
        default=DataOrigin.HISTORICAL,
        description="SYNTHETIC | HISTORICAL | EXTERNAL.",
    )
    source: str = Field(default="netcdf", description="Registered source name.")
    variables: list[str] = Field(default_factory=list)
    latitude: tuple[float, float] | None = None
    longitude: tuple[float, float] | None = None
    strict: bool = Field(default=True)
    dry_run: bool = Field(default=False)
    source_url: str | None = None
    license: str | None = None
    notes: str = ""


class IngestResponse(BaseModel):
    """Result of ``POST /data/ingest``."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    manifest: DatasetManifestOut
    quality: QualityReport
    dry_run: bool
    steps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    degraded: list[str] = Field(default_factory=list)


class ProcessRequest(BaseModel):
    """Body of ``POST /data/datasets/{id}/process``."""

    model_config = ConfigDict(extra="forbid")

    max_gap: int = Field(default=3, ge=0, le=100)


class ProcessResponse(BaseModel):
    """Result of running Phase 0 preprocessing over an ingested dataset."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    processed_path: str
    time_steps: int
    message: str


# --------------------------------------------------------------------------- #
# Dependencies
# --------------------------------------------------------------------------- #
def get_sources(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    """Dependency returning the registered data sources."""
    return default_sources(settings)


def get_registry(settings: Settings = Depends(get_settings)) -> DatasetRegistry:
    """Dependency returning the dataset registry.

    ``settings`` is injected so the registry resolves data files inside the
    configured data directory rather than the process-wide one.
    """
    from extreme_weather.ingestion import ManifestStore

    return DatasetRegistry(ManifestStore(settings.manifests_dir), settings=settings)


def get_ingestion_service(
    sources: Annotated[dict[str, Any], Depends(get_sources)],
    registry: Annotated[DatasetRegistry, Depends(get_registry)],
    settings: Settings = Depends(get_settings),
) -> IngestionService:
    """Dependency returning a configured ingestion service."""
    return IngestionService(sources=sources, registry=registry, settings=settings)


SourceDependency = Annotated[dict[str, Any], Depends(get_sources)]
RegistryDependency = Annotated[DatasetRegistry, Depends(get_registry)]
ServiceDependency = Annotated[IngestionService, Depends(get_ingestion_service)]



def _http_error(exc: Exception) -> HTTPException:
    """Translate an ingestion exception into the right HTTP status."""
    from extreme_weather.ingestion.exceptions import (
        DataSourceUnavailableError,
        FormatNotSupportedError,
        MissingDependencyError,
        ProviderCredentialsMissingError,
    )

    if isinstance(exc, SourceNotFoundError):
        return HTTPException(status_code=http_status.HTTP_404_NOT_FOUND, detail=str(exc))
    if isinstance(
        exc,
        (
            DataSourceUnavailableError,
            ProviderCredentialsMissingError,
            MissingDependencyError,
        ),
    ):
        return HTTPException(
            status_code=http_status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        )
    if isinstance(exc, FormatNotSupportedError):
        return HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    return HTTPException(
        status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
    )


# --------------------------------------------------------------------------- #
# GET /api/v1/data/sources
# --------------------------------------------------------------------------- #
@router.get(
    "/sources", response_model=SourceListResponse, summary="List available data sources"
)
def list_sources(sources: SourceDependency) -> SourceListResponse:
    """Return every registered source with its availability and requirements.

    A source that is registered but unusable (no GRIB engine, no NWP endpoint)
    is still listed with ``available: false`` - a client can then show *why*
    rather than being told the source does not exist.
    """
    infos = []
    for name in sorted(sources):
        source = sources[name]
        info: SourceInfo = source.metadata()
        available = source.available()
        infos.append(
            SourceInfo(
                name=info.name,
                kind=info.kind,
                description=info.description,
                formats=info.formats,
                requires_credentials=info.requires_credentials,
                required_environment=info.required_environment,
                optional_dependency=info.optional_dependency,
                default_origin=info.default_origin,
                notes=f"{info.notes} available={available}",
            )
        )
    return SourceListResponse(count=len(infos), sources=infos)


# --------------------------------------------------------------------------- #
# GET /api/v1/data/datasets
# --------------------------------------------------------------------------- #
@router.get(
    "/datasets",
    response_model=DatasetListResponse,
    summary="List ingested datasets (registry)",
)
def list_datasets(
    registry: RegistryDependency,
    origin: Annotated[
        str | None, Query(description="Filter by origin: SYNTHETIC, HISTORICAL or EXTERNAL.")
    ] = None,
    source: Annotated[str | None, Query(description="Filter by source name.")] = None,
    start_time: Annotated[
        str | None, Query(description="Only datasets whose period ends at or after this time.")
    ] = None,
    end_time: Annotated[
        str | None, Query(description="Only datasets whose period starts at or before this time.")
    ] = None,
) -> DatasetListResponse:
    """Return the ingested datasets, optionally filtered.

    Filters the registry rather than the filesystem, so a dataset that was never
    deliberately ingested is not silently listed.
    """
    from extreme_weather.ingestion.exceptions import IngestionError as _IE

    try:
        if start_time or end_time:
            entries = registry.find_by_time(start=start_time, end=end_time)
        else:
            entries = registry.list(origin=origin, source=source)
        if start_time or end_time:
            if origin:
                entries = [e for e in entries if e.origin == DataOrigin.validate(origin)]
            if source:
                entries = [e for e in entries if e.manifest.source == source]
    except (ValueError, _IE) as exc:
        raise _http_error(exc) from exc

    return DatasetListResponse(
        count=len(entries),
        datasets=[DatasetManifestOut(**entry.manifest.as_dict()) for entry in entries],
        message=None if entries else "No datasets have been ingested yet.",
    )


# --------------------------------------------------------------------------- #
# GET /api/v1/data/datasets/{dataset_id}
# --------------------------------------------------------------------------- #
@router.get(
    "/datasets/{dataset_id}",
    response_model=DatasetManifestOut,
    summary="Get one ingested dataset's manifest",
)
def get_dataset(dataset_id: str, registry: RegistryDependency) -> DatasetManifestOut:
    """Return the stored manifest for one dataset id (HTTP 404 when unknown)."""
    from extreme_weather.ingestion.exceptions import IngestionError as _IE

    try:
        return DatasetManifestOut(**registry.get(dataset_id).manifest.as_dict())
    except (SourceNotFoundError, _IE) as exc:
        raise _http_error(exc) from exc



# --------------------------------------------------------------------------- #
# GET /api/v1/data/datasets/{dataset_id}/quality
# --------------------------------------------------------------------------- #
@router.get(
    "/datasets/{dataset_id}/quality",
    response_model=QualityResponse,
    summary="Quality report of one ingested dataset",
)
def dataset_quality(dataset_id: str, registry: RegistryDependency) -> QualityResponse:
    """Re-run the quality check for a stored dataset and return the report."""
    from extreme_weather.ingestion.exceptions import IngestionError as _IE

    try:
        report = registry.validate(dataset_id)
    except (SourceNotFoundError, _IE) as exc:
        raise _http_error(exc) from exc
    return QualityResponse(**report.as_dict())


# --------------------------------------------------------------------------- #
# POST /api/v1/data/validate
# --------------------------------------------------------------------------- #
@router.post(
    "/validate",
    response_model=QualityResponse,
    summary="Quality-check a file without writing anything",
)
def validate_file(
    request: ValidateRequest, settings: Settings = Depends(get_settings)
) -> QualityResponse:
    """Open a file and report its quality. **Never writes.**

    The path is resolved inside the configured data directory, so a traversal
    attempt is rejected exactly as the Phase 0 dataset API does.
    """
    source = NetCDFSource(base_dir=settings.data_dir)
    try:
        resolved = source.resolve(request.path)
    except IngestionError as exc:
        raise _http_error(exc) from exc
    if not resolved.is_file():
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Dataset '{request.path}' was not found in '{settings.data_dir}'.",
        )
    try:
        with source.open_context(resolved) as dataset:
            report = check_quality(dataset, strict_ranges=request.strict_ranges)
    except IngestionError as exc:
        raise _http_error(exc) from exc
    return QualityResponse(**report.as_dict())


# --------------------------------------------------------------------------- #
# POST /api/v1/data/ingest
# --------------------------------------------------------------------------- #
@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingest a file: validate, normalise, manifest and register",
)
def ingest_file(
    request: IngestRequest, service: ServiceDependency, settings: Settings = Depends(get_settings)
) -> IngestResponse:
    """Run the full ingestion chain for a local file.

    The file is resolved inside the configured data directory. Real data is
    written under ``raw/real`` and synthetic under ``raw/synthetic``; the two are
    never mixed. Use ``dry_run`` to preview without writing.
    """
    try:
        path = _resolve_ingest_path(settings, request.path)
    except IngestionError as exc:
        raise _http_error(exc) from exc

    try:
        result = service.ingest(
            IngestionRequest(
                dataset_id=request.dataset_id,
                origin=request.origin,
                source_name=request.source,
                path=path,
                variables=tuple(request.variables),
                latitude=request.latitude,
                longitude=request.longitude,
                strict=request.strict,
                dry_run=request.dry_run,
                source_url=request.source_url,
                license_name=request.license,
                notes=request.notes,
            )
        )
    except (IngestionError, ValueError) as exc:
        raise _http_error(exc) from exc

    return IngestResponse(
        dataset_id=result.dataset_id,
        manifest=DatasetManifestOut(**result.manifest.as_dict()),
        quality=result.quality,
        dry_run=result.dry_run,
        steps=list(result.steps),
        warnings=list(result.warnings),
        degraded=list(result.degraded),
    )


def _resolve_ingest_path(settings: Settings, path: str) -> Any:
    """Resolve an ingest path inside the data directory.

    Accepts NetCDF and GRIB extensions; anything else is rejected so a caller
    cannot ask the service to read an arbitrary file type.
    """
    from pathlib import Path as _Path

    from extreme_weather.ingestion.grib import GRIB_SUFFIXES
    from extreme_weather.ingestion.netcdf import NETCDF_SUFFIXES

    candidate = _Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = settings.data_dir / candidate
    resolved = candidate.resolve()
    if resolved != settings.data_dir.resolve() and settings.data_dir.resolve() not in resolved.parents:
        from extreme_weather.ingestion.exceptions import SourceNotFoundError as _SNF

        raise _SNF(
            f"Path '{path}' resolves outside the configured data directory. "
            "Dataset paths must stay inside it."
        )
    if resolved.suffix.lower() not in NETCDF_SUFFIXES + GRIB_SUFFIXES:
        from extreme_weather.ingestion.exceptions import FormatNotSupportedError as _FNS

        raise _FNS(
            f"Unsupported extension '{resolved.suffix}'. Supported: "
            f"{', '.join(NETCDF_SUFFIXES + GRIB_SUFFIXES)}."
        )
    return resolved



# --------------------------------------------------------------------------- #
# POST /api/v1/data/datasets/{dataset_id}/process
# --------------------------------------------------------------------------- #
@router.post(
    "/datasets/{dataset_id}/process",
    response_model=ProcessResponse,
    summary="Run Phase 0 preprocessing over an ingested dataset",
)
def process_dataset(
    dataset_id: str,
    request: ProcessRequest,
    registry: RegistryDependency,
    settings: Settings = Depends(get_settings),
) -> ProcessResponse:
    """Preprocess an ingested dataset into the processed directory.

    This is the **existing Phase 0 pipeline** (``preprocess_file``) run over an
    ingested file, so the output is byte-for-byte the same kind of artefact the
    synthetic pipeline produces and Phases 1-5 read it unchanged.
    """
    from extreme_weather.data.preprocessing import preprocess_file
    from extreme_weather.data.validator import DatasetValidationError as _DV
    from extreme_weather.ingestion.exceptions import IngestionError as _IE

    try:
        entry = registry.get(dataset_id)
    except (SourceNotFoundError, _IE) as exc:
        raise _http_error(exc) from exc

    base = (
        settings.synthetic_raw_dir if entry.is_synthetic else settings.real_raw_dir
    )
    source_path = base / entry.manifest.file_name
    if not source_path.is_file():
        source_path = Path(entry.manifest.file_name)
    if not source_path.is_file():
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=(
                f"The data file '{entry.manifest.file_name}' for dataset "
                f"'{dataset_id}' was not found."
            ),
        )

    target = settings.processed_dir / f"{dataset_id}{settings.processed_real_suffix}"
    try:
        result = preprocess_file(source_path, target, max_gap=request.max_gap)
    except _DV as exc:
        raise HTTPException(
            status_code=http_status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        ) from exc
    except (IngestionError, OSError) as exc:
        raise _http_error(exc) from exc

    return ProcessResponse(
        dataset_id=dataset_id,
        processed_path=str(result.output_path),
        time_steps=int(result.dataset.sizes.get("time", 0)),
        message=(
            "Preprocessed with the Phase 0 pipeline. Phases 1-5 can now read this "
            "file unchanged. This is a research prototype, not an operational forecast."
        ),
    )

