"""Configurable research-dataset builder (SIH26078, Phase 7).

Phase 6 proved real data can be ingested. Phase 7 needs a **specific, bounded**
dataset - a date range, a box, a set of variables - laid out under
``data/validation/`` with a manifest, so the whole validation is reproducible
from one command.

The builder deliberately does **not** download anything on import, and it
refuses a request it cannot bound. A year of global ERA5 is tens of gigabytes;
a forgotten bounding box is how a research script fills a disk, so the
configuration requires an explicit box and a maximum day count.

Layout::

    data/validation/
        raw/          the ingested source, in the dataset's own units
        processed/    the Phase 0 output the pipeline consumes
        manifests/    the Phase 6 provenance manifest
        splits/       the train/validation/test boundaries
        reports/      every JSON report Phase 7 produces
        provenance/   one record per validation run
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from extreme_weather.config import Settings, get_settings
from extreme_weather.ingestion import (
    DataOrigin,
    IngestionRequest,
    IngestionService,
    OpenMeteoRequest,
    file_hash,
)
from extreme_weather.ingestion.base import utc_now
from extreme_weather.ingestion.manifest import DatasetManifest, ManifestStore
from extreme_weather.ingestion.registry import DatasetRegistry
from extreme_weather.validation.splits import SplitSummary, split_dataset, split_timestamps

__all__ = [
    "MAX_REQUEST_DAYS",
    "ValidationDatasetConfig",
    "ValidationDatasetError",
    "ValidationDatasetResult",
    "ValidationDatasetBuilder",
    "validation_dir",
]


#: Refuse a request longer than this. A year of hourly global data is tens of
#: gigabytes, and a research validation run should be minutes, not days.
MAX_REQUEST_DAYS: int = 120

#: Sub-directory of DATA_DIR holding the Phase 7 workspace.
VALIDATION_SUBDIR = "validation"


class ValidationDatasetError(ValueError):
    """Raised when a validation dataset cannot be built as requested."""


def validation_dir(settings: Settings | None = None) -> Path:
    """Return the ``data/validation`` workspace directory."""
    resolved = settings or get_settings()
    return resolved.data_dir / VALIDATION_SUBDIR


@dataclass(frozen=True)
class ValidationDatasetConfig:
    """What to build, bounded.

    Attributes:
        dataset_id: Stable identifier for the dataset.
        start_date: First day, ``YYYY-MM-DD``.
        end_date: Last day, ``YYYY-MM-DD`` (inclusive).
        latitudes: Latitude points.
        longitudes: Longitude points.
        variables: Canonical variables to request.
        train_ratio: Fraction of the period for training.
        validation_ratio: Fraction for validation; the test takes the rest.
        origin: Provenance class; ``HISTORICAL`` for real reanalysis.
        max_days: Hard ceiling on the requested period.
    """

    dataset_id: str
    start_date: str
    end_date: str
    latitudes: tuple[float, ...]
    longitudes: tuple[float, ...]
    variables: tuple[str, ...] = (
        "precipitation",
        "temperature",
        "surface_pressure",
        "relative_humidity_2m",
        "wind_speed_10m",
        "wind_direction_10m",
    )
    train_ratio: float = 0.6
    validation_ratio: float = 0.2
    origin: str = DataOrigin.HISTORICAL
    max_days: int = MAX_REQUEST_DAYS
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.dataset_id).strip():
            raise ValidationDatasetError("dataset_id must not be empty.")
        DataOrigin.validate(self.origin)

        if not self.latitudes or not self.longitudes:
            raise ValidationDatasetError(
                "A validation dataset needs an explicit bounding box. An unbounded "
                "request would be a multi-gigabyte download."
            )
        for value in self.latitudes:
            if not -90.0 <= value <= 90.0:
                raise ValidationDatasetError(f"Latitude {value} is outside [-90, 90].")
        for value in self.longitudes:
            if not -180.0 <= value <= 180.0:
                raise ValidationDatasetError(
                    f"Longitude {value} is outside [-180, 180]."
                )
        if self.start_date > self.end_date:
            raise ValidationDatasetError(
                f"start_date '{self.start_date}' is after end_date '{self.end_date}'."
            )
        days = self.day_count()
        if days > int(self.max_days):
            raise ValidationDatasetError(
                f"The request covers {days} day(s), over the {self.max_days}-day "
                "ceiling. Narrow the date range or the bounding box."
            )
        if not self.variables:
            raise ValidationDatasetError("At least one variable must be requested.")

    def day_count(self) -> int:
        """Number of days the request covers (both endpoints included)."""
        start = datetime.fromisoformat(self.start_date).date()
        end = datetime.fromisoformat(self.end_date).date()
        return (end - start).days + 1

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "start_date": self.start_date,
            "end_date": self.end_date,
            "days": self.day_count(),
            "latitudes": list(self.latitudes),
            "longitudes": list(self.longitudes),
            "variables": list(self.variables),
            "train_ratio": self.train_ratio,
            "validation_ratio": self.validation_ratio,
            "origin": self.origin,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class ValidationDatasetResult:
    """What the builder produced.

    Attributes:
        config: The request that was honoured.
        manifest: The Phase 6 provenance manifest of the ingested dataset.
        raw_path: Where the normalised raw data was written.
        processed_path: Where the Phase 0 output was written, when processed.
        split: The chronological split of the dataset.
        datasets: The three split datasets, keyed by split name.
        warnings: Anything a reader must know.
    """

    config: ValidationDatasetConfig
    manifest: DatasetManifest
    raw_path: Path
    split: SplitSummary
    processed_path: Path | None = None
    datasets: dict[str, xr.Dataset] = field(default_factory=dict)
    warnings: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (no array data)."""
        return {
            "config": self.config.as_dict(),
            "manifest": self.manifest.as_dict(),
            "raw_path": str(self.raw_path),
            "processed_path": (
                str(self.processed_path) if self.processed_path else None
            ),
            "split": self.split.as_dict(),
            "split_sizes": {
                name: int(dataset.sizes.get("time", 0))
                for name, dataset in self.datasets.items()
            },
            "warnings": list(self.warnings),
        }


class ValidationDatasetBuilder:
    """Build the Phase 7 research dataset from real historical data.

    Args:
        settings: Application settings. Defaults to the process-wide ones.
        service: An ingestion service to reuse. Defaults to a fresh one wired
            to the validation manifests directory.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        service: IngestionService | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._root = validation_dir(self._settings)
        self._service = service or IngestionService(
            registry=DatasetRegistry(
                ManifestStore(self.manifests_dir), settings=self._settings
            ),
            settings=self._settings,
        )

    # -- directories ---------------------------------------------------- #
    @property
    def root(self) -> Path:
        """The ``data/validation`` directory."""
        return self._root

    @property
    def settings(self) -> Settings:
        """The settings this builder was constructed with."""
        return self._settings

    @property
    def raw_dir(self) -> Path:
        """Where the ingested raw dataset is written."""
        return self._root / "raw"

    @property
    def processed_dir(self) -> Path:
        """Where the Phase 0 output is written."""
        return self._root / "processed"

    @property
    def manifests_dir(self) -> Path:
        """Where the dataset manifest is written."""
        return self._root / "manifests"

    @property
    def splits_dir(self) -> Path:
        """Where the split definition is written."""
        return self._root / "splits"

    @property
    def reports_dir(self) -> Path:
        """Where every report is written."""
        return self._root / "reports"

    @property
    def provenance_dir(self) -> Path:
        """Where run provenance records are written."""
        return self._root / "provenance"

    def ensure_directories(self) -> None:
        """Create every sub-directory of the workspace."""
        for directory in (
            self.raw_dir,
            self.processed_dir,
            self.manifests_dir,
            self.splits_dir,
            self.reports_dir,
            self.provenance_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


    # -- building -------------------------------------------------------- #
    def fetch(self, config: ValidationDatasetConfig) -> Path:
        """Download the configured ERA5 subset into the validation raw directory.

        Returns:
            The path of the downloaded (source-unit) NetCDF file.

        Raises:
            ValidationDatasetError: If the request is unbounded or the download
                fails. Nothing unbounded is ever attempted.
        """
        self.ensure_directories()
        request = OpenMeteoRequest(
            latitudes=config.latitudes,
            longitudes=config.longitudes,
            start_date=config.start_date,
            end_date=config.end_date,
            variables=config.variables,
        )
        source = self._service.get_source("era5")
        dataset = source.load_request(request)  # type: ignore[attr-defined]
        target = self.raw_dir / f"{config.dataset_id}_source.nc"
        dataset.to_netcdf(target, mode="w")
        return target

    def build(
        self,
        config: ValidationDatasetConfig,
        *,
        fetch: bool = True,
        source_path: Path | str | None = None,
        process: bool = True,
    ) -> ValidationDatasetResult:
        """Build the dataset, ingest it, split it, and optionally preprocess it.

        Args:
            config: What to build.
            fetch: Download the source first. Set False when ``source_path``
                points at a file that already exists.
            source_path: An existing source file to use instead of downloading.
            process: Also run the Phase 0 preprocessing pipeline.

        Returns:
            A :class:`ValidationDatasetResult`.

        Raises:
            ValidationDatasetError: If the dataset cannot be built or split.
        """
        self.ensure_directories()
        warnings: list[str] = []

        if source_path is None:
            if not fetch:
                raise ValidationDatasetError(
                    "build() needs either fetch=True or an explicit source_path."
                )
            source_path = self.fetch(config)
        path = Path(source_path)
        if not path.is_file():
            raise ValidationDatasetError(f"Source file '{path}' does not exist.")

        # Ingest through Phase 6 so the dataset gets units, quality control, a
        # manifest and provenance exactly as any other real dataset would.
        result = self._service.ingest(
            IngestionRequest(
                dataset_id=config.dataset_id,
                origin=config.origin,
                path=path,
                notes=(
                    f"Phase 7 validation dataset. Requested {config.day_count()} "
                    f"day(s) over {len(config.latitudes)}x{len(config.longitudes)} "
                    "grid point(s). Historical reanalysis: not a forecast and not a "
                    "live feed."
                ),
            )
        )
        raw_path = result.output_path
        if raw_path is None:
            raise ValidationDatasetError(
                "Ingestion produced no output file; the dataset is unusable."
            )

        processed_path: Path | None = None
        if process:
            from extreme_weather.data.preprocessing import preprocess_file

            target = self.processed_dir / f"{config.dataset_id}_processed.nc"
            try:
                processed = preprocess_file(raw_path, target, max_gap=3)
                processed_path = processed.output_path
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                warnings.append(
                    f"Phase 0 preprocessing failed: {type(exc).__name__}: {exc}. "
                    "The raw normalised dataset is still available."
                )

        dataset = xr.open_dataset(processed_path or raw_path).load()
        try:
            summary = split_timestamps(
                dataset.coords["time"].values,
                train_ratio=config.train_ratio,
                validation_ratio=config.validation_ratio,
            )
            splits = split_dataset(
                dataset,
                train_ratio=config.train_ratio,
                validation_ratio=config.validation_ratio,
            )
        except Exception as exc:  # noqa: BLE001 - normalised into a domain error
            raise ValidationDatasetError(
                f"The dataset could not be split chronologically: {exc}"
            ) from exc
        finally:
            dataset.close()

        self._write_split(config, summary)

        if config.day_count() < 30:
            warnings.append(
                f"The dataset covers only {config.day_count()} day(s). That is far "
                "too short for a day-of-year climatology or a meaningful threshold "
                "calibration; the diagnostics will say so."
            )

        return ValidationDatasetResult(
            config=config,
            manifest=result.manifest,
            raw_path=raw_path,
            processed_path=processed_path,
            split=summary,
            datasets=splits,
            warnings=tuple(warnings),
        )

    def _write_split(
        self, config: ValidationDatasetConfig, summary: SplitSummary
    ) -> Path:
        """Persist the split boundaries so a run can be reproduced exactly."""
        target = self.splits_dir / f"{config.dataset_id}_split.json"
        payload = {
            "dataset_id": config.dataset_id,
            "train_ratio": summary.train_ratio,
            "validation_ratio": summary.validation_ratio,
            "chronological": True,
            "leakage_free": True,
            **summary.as_dict(),
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return target

