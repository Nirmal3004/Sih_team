"""Real / historical weather data ingestion (SIH26078, Phase 6).

This package turns a **real** weather file into a dataset the Phase 1-5 pipeline
can consume, without changing that pipeline at all.

```text
WeatherDataSource (netcdf | grib | era5 | nwp)
        |
        v
   fetch()      a real path, or a clear error - never fabricated data
        |
        v
   load()       opened in the SOURCE's own units
        |
        v
   check_quality()      DataQualityReport: valid / errors / warnings
        |
        v
   normalize_dataset()  standard units + coordinates, every step recorded
        |
        v
   build_manifest()     provenance, hashes, origin, synthetic/real flag
        |
        v
   DatasetRegistry      a queryable entry
        |
        v
   Phase 1 anomaly detection  (unchanged)
```

Three rules the whole package is built around:

1. **Synthetic and real never mix.** Every dataset declares an ``origin`` and is
   written to ``raw/synthetic`` or ``raw/real``. There is no default origin,
   because guessing is how real data ends up presented as synthetic.
2. **Never fabricate.** A missing file, an unreachable provider, a missing
   optional dependency and a missing credential each raise a *specific*,
   actionable error. None of them produces a plausible-looking empty dataset.
3. **Never invent provenance.** A local file has ``source="local_file"`` and no
   URL; the manifest constructor refuses the combination outright.

Quick start::

    from extreme_weather.ingestion import IngestionRequest, IngestionService

    service = IngestionService()
    result = service.ingest(
        IngestionRequest(
            dataset_id="era5_demo",
            path="data/raw/real/era5_demo.nc",
            origin="HISTORICAL",
        )
    )
    print(result.quality.as_dict())
"""

from __future__ import annotations

from extreme_weather.ingestion.base import (
    DataOrigin,
    SourceInfo,
    WeatherDataSource,
    utc_now,
)
from extreme_weather.ingestion.era5 import (
    ERA5_API_URL,
    ERA5_ATTRIBUTION,
    ERA5_LICENSE,
    ERA5Source,
    OpenMeteoRequest,
    era5_payload_to_dataset,
)
from extreme_weather.ingestion.exceptions import (
    DataSourceUnavailableError,
    FormatNotSupportedError,
    IngestionError,
    MissingDependencyError,
    NormalizationError,
    ProviderCredentialsMissingError,
    QualityCheckError,
    SourceNotFoundError,
    UnsupportedFormatError,
)
from extreme_weather.ingestion.grib import (
    GRIB_SUFFIXES,
    GRIBSource,
    grib_backend_available,
    grib_backend_name,
)
from extreme_weather.ingestion.manifest import (
    MANIFEST_SCHEMA_VERSION,
    SOURCE_LOCAL_FILE,
    DatasetManifest,
    ManifestStore,
    ProvenanceStep,
    build_manifest,
    content_hash,
    file_hash,
)
from extreme_weather.ingestion.netcdf import (
    NETCDF_SUFFIXES,
    NetCDFSource,
    default_chunks,
)
from extreme_weather.ingestion.normalizer import (
    ALIAS_MAP,
    STANDARD_UNITS,
    ConversionRecord,
    NormalizationResult,
    normalize_coordinates,
    normalize_dataset,
    normalize_variable_name,
)
from extreme_weather.ingestion.nwp import (
    NWP_API_TOKEN_ENV,
    NWP_API_URL_ENV,
    NWPFormRequest,
    NWPSource,
)
from extreme_weather.ingestion.quality import (
    PLAUSIBLE_RANGES,
    DataQualityReport,
    check_quality,
    require_valid,
)
from extreme_weather.ingestion.registry import (
    DatasetEntry,
    DatasetRegistry,
    default_sources,
)
from extreme_weather.ingestion.service import (
    IngestionRequest,
    IngestionResult,
    IngestionService,
    derive_uv_from_polar_wind,
    subset_dataset,
)

__all__ = [
    "ALIAS_MAP",
    "ERA5_API_URL",
    "ERA5_ATTRIBUTION",
    "ERA5_LICENSE",
    "GRIB_SUFFIXES",
    "MANIFEST_SCHEMA_VERSION",
    "NETCDF_SUFFIXES",
    "NWP_API_TOKEN_ENV",
    "NWP_API_URL_ENV",
    "PLAUSIBLE_RANGES",
    "SOURCE_LOCAL_FILE",
    "STANDARD_UNITS",
    "ConversionRecord",
    "DataOrigin",
    "DataQualityReport",
    "DataSourceUnavailableError",
    "DatasetEntry",
    "DatasetManifest",
    "DatasetRegistry",
    "ERA5Source",
    "FormatNotSupportedError",
    "GRIBSource",
    "IngestionError",
    "IngestionRequest",
    "IngestionResult",
    "IngestionService",
    "ManifestStore",
    "MissingDependencyError",
    "NWPFormRequest",
    "NWPSource",
    "NetCDFSource",
    "NormalizationError",
    "NormalizationResult",
    "OpenMeteoRequest",
    "ProvenanceStep",
    "ProviderCredentialsMissingError",
    "QualityCheckError",
    "SourceInfo",
    "SourceNotFoundError",
    "UnsupportedFormatError",
    "WeatherDataSource",
    "build_manifest",
    "check_quality",
    "content_hash",
    "default_chunks",
    "default_sources",
    "era5_payload_to_dataset",
    "file_hash",
    "grib_backend_available",
    "grib_backend_name",
    "normalize_coordinates",
    "normalize_dataset",
    "normalize_variable_name",
    "require_valid",
    "subset_dataset",
    "utc_now",
]
