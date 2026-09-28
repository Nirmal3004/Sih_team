"""Data layer: synthetic generation, NetCDF I/O, validation, preprocessing.

Typical Phase 0 usage::

    from extreme_weather.data import generate_and_save, preprocess_file

    generate_and_save()                       # data/raw/sample_history.nc (+ manifest)
    result = preprocess_file("data/raw/sample_history.nc")
    print(result.report.as_dict())
"""

from __future__ import annotations

from extreme_weather.data.derived import (
    WIND_DIRECTION_NAME,
    WIND_SPEED_NAME,
    add_derived_variables,
    wind_direction,
    wind_speed,
)
from extreme_weather.data.generator import (
    DATASET_BASENAME,
    GENERATOR_VERSION,
    MANIFEST_SUFFIX,
    NETCDF_SUFFIX,
    SYNTHETIC_SOURCE,
    VARIABLES,
    ExtremeEventConfig,
    GenerationResult,
    SyntheticConfig,
    build_manifest,
    default_config,
    event_field,
    generate_and_save,
    generate_synthetic_dataset,
    manifest_path_for,
    write_manifest,
)
from extreme_weather.data.loader import (
    default_encoding,
    load_dataset,
    resolve_dataset_path,
    save_dataset,
)
from extreme_weather.data.missing import (
    DEFAULT_MAX_GAP,
    MissingGap,
    MissingValueError,
    count_missing,
    fill_missing,
    fill_missing_dataset,
    find_missing_gaps,
    missing_fraction,
)
from extreme_weather.data.preprocessing import (
    PIPELINE_VERSION,
    PreprocessingReport,
    PreprocessingResult,
    default_processed_path,
    preprocess_dataset,
    preprocess_file,
)
from extreme_weather.data.validator import (
    LATITUDE_DIM,
    LONGITUDE_DIM,
    PLAUSIBLE_RANGES,
    REQUIRED_DIMENSIONS,
    REQUIRED_VARIABLES,
    TIME_DIM,
    DatasetError,
    DatasetLoadError,
    DatasetSummary,
    DatasetValidationError,
    ValidationIssues,
    VariableInfo,
    collect_validation_issues,
    summarise_dataset,
    validate_dataset,
)

__all__ = [
    "DATASET_BASENAME",
    "DEFAULT_MAX_GAP",
    "DatasetError",
    "DatasetLoadError",
    "DatasetSummary",
    "DatasetValidationError",
    "ExtremeEventConfig",
    "GENERATOR_VERSION",
    "GenerationResult",
    "LATITUDE_DIM",
    "LONGITUDE_DIM",
    "MANIFEST_SUFFIX",
    "MissingGap",
    "MissingValueError",
    "NETCDF_SUFFIX",
    "PIPELINE_VERSION",
    "PLAUSIBLE_RANGES",
    "PreprocessingReport",
    "PreprocessingResult",
    "REQUIRED_DIMENSIONS",
    "REQUIRED_VARIABLES",
    "SYNTHETIC_SOURCE",
    "SyntheticConfig",
    "TIME_DIM",
    "VARIABLES",
    "ValidationIssues",
    "VariableInfo",
    "WIND_DIRECTION_NAME",
    "WIND_SPEED_NAME",
    "add_derived_variables",
    "build_manifest",
    "collect_validation_issues",
    "count_missing",
    "default_config",
    "default_encoding",
    "default_processed_path",
    "event_field",
    "fill_missing",
    "fill_missing_dataset",
    "find_missing_gaps",
    "generate_and_save",
    "generate_synthetic_dataset",
    "load_dataset",
    "manifest_path_for",
    "missing_fraction",
    "preprocess_dataset",
    "preprocess_file",
    "resolve_dataset_path",
    "save_dataset",
    "summarise_dataset",
    "validate_dataset",
    "wind_direction",
    "wind_speed",
    "write_manifest",
]
