"""Preprocessing pipeline: load -> validate -> missing values -> derived -> save.

The pipeline is intentionally explicit and side-effect free with respect to its
inputs:

1. :func:`preprocess_dataset` never mutates the dataset it receives - it works on
   a deep copy, so callers can keep the pristine raw data for comparison.
2. Output directories are created automatically using :mod:`pathlib`.
3. The returned :class:`PreprocessingReport` records how many values were filled
   per variable, which variables were derived and which options were used, and
   that report is also persisted in the NetCDF global attributes.

Deterministic by construction: the same input plus the same options produces the
same output, because gap filling is a pure computation (no randomness, no
timestamps inside the arrays).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import xarray as xr

from extreme_weather.config import PROJECT_ID, PACKAGE_VERSION, get_settings
from extreme_weather.data.derived import WIND_DIRECTION_NAME, WIND_SPEED_NAME, add_derived_variables
from extreme_weather.data.loader import load_dataset, save_dataset
from extreme_weather.data.missing import (
    DEFAULT_MAX_GAP,
    count_missing,
    fill_missing_dataset,
)
from extreme_weather.data.validator import TIME_DIM, validate_dataset

PIPELINE_VERSION: str = PACKAGE_VERSION
DEFAULT_PROCESSED_SUFFIX: str = "_processed"
PROCESSED_DATASET_BASENAME: str = "sample_history"

__all__ = [
    "DEFAULT_MAX_GAP",
    "DEFAULT_PROCESSED_SUFFIX",
    "PIPELINE_VERSION",
    "PROCESSED_DATASET_BASENAME",
    "PreprocessingReport",
    "PreprocessingResult",
    "default_processed_path",
    "preprocess_dataset",
    "preprocess_file",
]


@dataclass(frozen=True)
class PreprocessingReport:
    """Structured summary of a preprocessing run."""

    input_path: str | None
    output_path: str | None
    pipeline_version: str
    max_gap: int | None
    fill_dim: str
    variables: tuple[str, ...]
    derived_variables: tuple[str, ...]
    missing_before: dict[str, int]
    missing_after: dict[str, int]
    total_missing_before: int
    total_missing_after: int
    values_filled: int
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, object]:
        """Return the report as a JSON friendly dictionary."""
        payload = asdict(self)
        payload["variables"] = list(self.variables)
        payload["derived_variables"] = list(self.derived_variables)
        payload["warnings"] = list(self.warnings)
        return payload


@dataclass(frozen=True, eq=False)
class PreprocessingResult:
    """Processed dataset plus the report describing how it was produced."""

    dataset: xr.Dataset
    report: PreprocessingReport

    @property
    def output_path(self) -> Path | None:
        """Path of the processed NetCDF file, when it was written to disk."""
        return Path(self.report.output_path) if self.report.output_path else None


# --------------------------------------------------------------------------- #
# Pipeline stages
# --------------------------------------------------------------------------- #
def preprocess_dataset(
    dataset: xr.Dataset,
    *,
    max_gap: int | None = DEFAULT_MAX_GAP,
    fill_dim: str = TIME_DIM,
    add_wind_direction: bool = True,
    validate_input: bool = True,
    input_path: Path | str | None = None,
) -> PreprocessingResult:
    """Run the in-memory preprocessing pipeline.

    Args:
        dataset: Raw dataset (never mutated).
        max_gap: Maximum interior gap length that may be filled.
        fill_dim: Dimension along which gaps are filled.
        add_wind_direction: Also derive the meteorological wind direction.
        validate_input: Validate the raw dataset against the data contract first.
        input_path: Optional provenance path recorded in the report.

    Returns:
        :class:`PreprocessingResult` holding the processed dataset and report.

    Raises:
        DatasetValidationError: If the input violates the data contract.
    """
    if not isinstance(dataset, xr.Dataset):
        raise TypeError(
            f"preprocess_dataset expects an xarray.Dataset, got {type(dataset).__name__}."
        )

    warnings: list[str] = []
    if validate_input:
        issues = validate_dataset(dataset)
        warnings.extend(issues.warnings)

    # Deep copy so the caller's raw dataset stays untouched.
    working = dataset.copy(deep=True)

    missing_before = count_missing(working)
    filled = fill_missing_dataset(working, max_gap, dim=fill_dim)
    with_derived = add_derived_variables(filled, include_wind_direction=add_wind_direction)
    missing_after = count_missing(with_derived)

    derived_variables = [WIND_SPEED_NAME]
    if add_wind_direction:
        derived_variables.append(WIND_DIRECTION_NAME)

    total_before = sum(missing_before.values())
    total_after = sum(missing_after.values())

    report = PreprocessingReport(
        input_path=str(input_path) if input_path is not None else None,
        output_path=None,
        pipeline_version=PIPELINE_VERSION,
        max_gap=max_gap,
        fill_dim=fill_dim,
        variables=tuple(dataset.data_vars),
        derived_variables=tuple(derived_variables),
        missing_before=missing_before,
        missing_after=missing_after,
        total_missing_before=total_before,
        total_missing_after=total_after,
        values_filled=total_before - total_after,
        warnings=tuple(warnings),
    )

    with_derived.attrs = dict(dataset.attrs)
    with_derived.attrs.update(
        {
            "project": PROJECT_ID,
            "processed": "true",
            "preprocessing_version": PIPELINE_VERSION,
            "preprocessing": json.dumps(report.as_dict(), sort_keys=True),
            "derived_variables": ", ".join(derived_variables),
            "missing_value_policy": (
                "linear interpolation of interior gaps with max_gap="
                + ("unlimited" if max_gap is None else str(max_gap))
                + f" along '{fill_dim}'; leading/trailing gaps preserved"
            ),
            "history": (
                f"{dataset.attrs.get('history', 'n/a')} | preprocessed by "
                f"extreme_weather.data.preprocessing {PIPELINE_VERSION}"
            ),
        }
    )
    return PreprocessingResult(dataset=with_derived, report=report)



def default_processed_path(
    input_path: Path | str,
    *,
    suffix: str = DEFAULT_PROCESSED_SUFFIX,
    output_dir: Path | str | None = None,
) -> Path:
    """Return ``<processed_dir>/<input stem><suffix>.nc`` for ``input_path``."""
    source = Path(input_path).expanduser()
    directory = (
        Path(output_dir).expanduser() if output_dir is not None else get_settings().processed_dir
    )
    return directory / f"{source.stem}{suffix}{source.suffix}"


def preprocess_file(
    input_path: Path | str,
    output_path: Path | str | None = None,
    *,
    max_gap: int | None = DEFAULT_MAX_GAP,
    fill_dim: str = TIME_DIM,
    add_wind_direction: bool = True,
    compression_level: int = 4,
) -> PreprocessingResult:
    """Preprocess a NetCDF file end-to-end and write the processed NetCDF.

    Pipeline: ``load -> validate -> missing values -> derived variables -> save``.
    The output directory is created automatically when it does not exist, and the
    input file is never modified.

    Args:
        input_path: Raw NetCDF file (e.g. ``data/raw/sample_history.nc``).
        output_path: Destination file; defaults to
            ``<DATA_DIR>/processed/<name>_processed.nc``.
        max_gap: Maximum interior gap length that may be filled.
        fill_dim: Dimension along which gaps are filled.
        add_wind_direction: Also derive the meteorological wind direction.
        compression_level: zlib compression level used when writing.

    Returns:
        :class:`PreprocessingResult` for the written dataset.
    """
    source = Path(input_path).expanduser()
    raw = load_dataset(source, validate=False)
    in_memory = preprocess_dataset(
        raw,
        max_gap=max_gap,
        fill_dim=fill_dim,
        add_wind_direction=add_wind_direction,
        validate_input=True,
        input_path=source,
    )

    destination = (
        Path(output_path).expanduser()
        if output_path is not None
        else default_processed_path(source)
    )
    saved_path = save_dataset(
        in_memory.dataset, destination, compression_level=compression_level
    )
    report = replace(in_memory.report, output_path=str(saved_path))
    return PreprocessingResult(dataset=in_memory.dataset, report=report)
