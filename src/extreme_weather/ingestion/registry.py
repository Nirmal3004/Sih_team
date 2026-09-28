"""Ingested-dataset registry (SIH26078, Phase 6).

The registry is the index of what has been ingested: it maps a stable
``dataset_id`` to a manifest, and it is what the ``/api/v1/data/datasets``
endpoints read.

Design notes
------------
* It **extends** rather than replaces the Phase 0 dataset API. That API lists
  ``.nc`` files on disk; the registry knows which of them were deliberately
  ingested, where they came from, whether they are synthetic, and what was done
  to them. Both views are useful and neither replaces the other.
* It is backed by :class:`~extreme_weather.ingestion.manifest.ManifestStore`, so
  a registry entry is a file a human can read.
* Synthetic and real datasets are queryable **separately**. That separation is
  the point: an analysis over mixed data is uninterpretable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

from extreme_weather.ingestion.base import DataOrigin
from extreme_weather.ingestion.exceptions import SourceNotFoundError
from extreme_weather.ingestion.manifest import DatasetManifest, ManifestStore
from extreme_weather.ingestion.quality import DataQualityReport, check_quality

__all__ = [
    "REGISTRY_SCHEMA_VERSION",
    "DatasetEntry",
    "DatasetRegistry",
    "default_sources",
    "parse_timestamp",
]

#: Bumped when the registry's on-disk representation changes.
REGISTRY_SCHEMA_VERSION: str = "1.0"


def parse_timestamp(value: str | datetime | None) -> datetime | None:
    """Parse an ISO 8601 string, a ``datetime``, or ``None``.

    Raises:
        ValueError: If the string is not a valid ISO 8601 timestamp.
    """
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"'{value}' is not a valid ISO 8601 timestamp.") from exc
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


@dataclass(frozen=True)
class DatasetEntry:
    """A manifest plus the derived flags the API needs.

    Attributes:
        manifest: The stored manifest.
        quality: The most recent quality report, when one has been run.
    """

    manifest: DatasetManifest
    quality: DataQualityReport | None = None

    @property
    def dataset_id(self) -> str:
        """Stable identifier of the dataset."""
        return self.manifest.dataset_id

    @property
    def is_synthetic(self) -> bool:
        """True only for data this project generated itself."""
        return self.manifest.origin == DataOrigin.SYNTHETIC

    @property
    def origin(self) -> str:
        """One of ``SYNTHETIC``, ``HISTORICAL``, ``EXTERNAL``."""
        return self.manifest.origin

    @property
    def quality_valid(self) -> bool | None:
        """Whether the last quality check passed, or ``None`` if never checked."""
        return None if self.quality is None else self.quality.valid

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary for the API."""
        payload = self.manifest.as_dict()
        payload["quality_valid"] = self.quality_valid
        payload["quality"] = self.quality.as_dict() if self.quality is not None else None
        return payload


class DatasetRegistry:
    """Index of ingested datasets, backed by a :class:`ManifestStore`.

    Args:
        store: Where manifests live. Defaults to ``<DATA_DIR>/manifests``.
        quality_reports: Optional per-dataset quality reports, so a registry
            rebuilt in a fresh process can still report quality without
            re-reading every dataset.
    """

    def __init__(
        self,
        store: ManifestStore | None = None,
        *,
        quality_reports: dict[str, DataQualityReport] | None = None,
        settings: Any | None = None,
    ) -> None:
        from extreme_weather.config import get_settings

        self._settings = settings or get_settings()
        self._store = store or ManifestStore()
        self._quality = dict(quality_reports or {})

    def _data_file(self, entry: DatasetEntry) -> Path:
        """Locate an entry's data file.

        The manifest records only the *name*, so the file is looked for in the
        raw directory that matches the entry's origin. The directories come from
        the **injected** settings, not the process-wide ones, so a registry built
        with a temporary data directory reads that directory rather than the
        repository's real ``data/``.
        """
        base = (
            self._settings.synthetic_raw_dir
            if entry.is_synthetic
            else self._settings.real_raw_dir
        )
        candidate = base / entry.manifest.file_name
        if candidate.is_file():
            return candidate
        fallback = Path(entry.manifest.file_name)
        return fallback if fallback.is_file() else candidate

    @property
    def store(self) -> ManifestStore:
        """The underlying manifest store."""
        return self._store

    def register(
        self,
        manifest: DatasetManifest,
        quality: DataQualityReport | None = None,
    ) -> DatasetEntry:
        """Add (or replace) a dataset and persist its manifest.

        Replacing rather than erroring is deliberate: re-ingesting the same
        ``dataset_id`` after a re-download is a normal operation, and the
        manifest describes the *current* file, not an append-only log.
        """
        self._store.save(manifest)
        if quality is not None:
            self._quality[manifest.dataset_id] = quality
        return DatasetEntry(manifest=manifest, quality=quality)

    def list(
        self,
        *,
        origin: str | None = None,
        source: str | None = None,
        valid_only: bool = False,
    ) -> list[DatasetEntry]:
        """Return every registered dataset, optionally filtered.

        Args:
            origin: Keep only this origin (``SYNTHETIC``/``HISTORICAL``/
                ``EXTERNAL``) - the filter that keeps synthetic and real results
                apart.
            source: Keep only datasets from this source name.
            valid_only: Keep only datasets whose last quality check passed.

        Raises:
            ValueError: If ``origin`` is not a known origin.
        """
        wanted = DataOrigin.validate(origin) if origin else None
        entries: list[DatasetEntry] = []
        for manifest in self._store.list():
            entry = DatasetEntry(
                manifest=manifest, quality=self._quality.get(manifest.dataset_id)
            )
            if wanted and entry.origin != wanted:
                continue
            if source and manifest.source != source:
                continue
            if valid_only and entry.quality_valid is False:
                continue
            entries.append(entry)
        return sorted(entries, key=lambda item: item.dataset_id)

    def get(self, dataset_id: str) -> DatasetEntry:
        """Return one entry.

        Raises:
            SourceNotFoundError: If the dataset id is not registered.
        """
        manifest = self._store.load(dataset_id)
        if manifest is None:
            known = ", ".join(self._store.list_ids()) or "none"
            raise SourceNotFoundError(
                f"Dataset '{dataset_id}' is not registered. Known datasets: {known}."
            )
        return DatasetEntry(manifest=manifest, quality=self._quality.get(dataset_id))

    def find(self, dataset_id: str) -> DatasetEntry | None:
        """Return one entry, or ``None`` when it is not registered."""
        try:
            return self.get(dataset_id)
        except SourceNotFoundError:
            return None

    def remove(self, dataset_id: str) -> bool:
        """Remove a dataset from the registry.

        Returns True when a manifest was actually deleted. The **data file is
        left on disk** - deleting ingested data is a separate, explicit act.
        """
        self._quality.pop(dataset_id, None)
        return self._store.remove(dataset_id)

    def ids(self) -> list[str]:
        """Return every registered dataset id, sorted."""
        return self._store.list_ids()


    def find_by_time(
        self,
        start: str | datetime | None = None,
        end: str | datetime | None = None,
    ) -> list[DatasetEntry]:
        """Return datasets whose time range overlaps ``[start, end]``."""
        from extreme_weather.ingestion.manifest import _overlaps

        start_iso = parse_timestamp(start)
        end_iso = parse_timestamp(end)
        start_text = start_iso.isoformat() if start_iso else None
        end_text = end_iso.isoformat() if end_iso else None

        matches: list[DatasetEntry] = []
        for entry in self.list():
            if _overlaps(
                entry.manifest.time_start, entry.manifest.time_end, start_text, end_text
            ):
                matches.append(entry)
        return matches

    def find_by_variable(self, variable: str) -> list[DatasetEntry]:
        """Return datasets that carry ``variable`` (case-insensitive)."""
        wanted = str(variable).strip().lower()
        return [
            entry
            for entry in self.list()
            if any(name.lower() == wanted for name in entry.manifest.variables)
        ]

    def validate(self, dataset_id: str, *, attach: bool = True) -> DataQualityReport:
        """Re-run the quality check for one dataset and remember the result.

        The data file is located by the manifest's ``file_name`` inside
        ``raw_dir`` (real or synthetic, per the manifest's origin), so a registry
        entry can be re-checked without the caller supplying a path.
        """
        from extreme_weather.ingestion.netcdf import NetCDFSource

        entry = self.get(dataset_id)
        candidate = self._data_file(entry)
        if not candidate.is_file():
            raise SourceNotFoundError(
                f"The data file '{entry.manifest.file_name}' for dataset "
                f"'{dataset_id}' was not found under '{candidate.parent}'."
            )

        source = NetCDFSource()
        with source.open_context(candidate) as dataset:
            report = check_quality(dataset)
        if attach:
            self._quality[dataset_id] = report
        return report

    def validate_all(self) -> dict[str, DataQualityReport]:
        """Re-run the quality check for every registered dataset.

        A dataset whose file is missing becomes an invalid report rather than
        aborting the whole sweep.
        """
        from extreme_weather.ingestion.base import utc_now

        reports: dict[str, DataQualityReport] = {}
        for dataset_id in self._store.list_ids():
            try:
                reports[dataset_id] = self.validate(dataset_id)
            except SourceNotFoundError as exc:
                reports[dataset_id] = DataQualityReport(
                    valid=False, errors=(str(exc),), generated_at=utc_now()
                )
        self._quality.update(reports)
        return reports



def default_sources(settings: Any | None = None) -> dict[str, Any]:
    """Return the built-in data sources, keyed by name.

    Four sources, all satisfying
    :class:`~extreme_weather.ingestion.base.WeatherDataSource`:

    * ``netcdf`` - local NetCDF, always available;
    * ``grib``   - local GRIB, available only with the optional backend;
    * ``era5``   - the Open-Meteo archive, no credentials required;
    * ``nwp``    - NWP model output, local or from a configured endpoint.

    A source that is not usable is still returned, so the API can list it with
    ``available: false`` rather than pretending it does not exist.
    """
    from extreme_weather.config import get_settings
    from extreme_weather.ingestion.base import WeatherDataSource
    from extreme_weather.ingestion.era5 import ERA5Source
    from extreme_weather.ingestion.grib import GRIBSource
    from extreme_weather.ingestion.netcdf import NetCDFSource
    from extreme_weather.ingestion.nwp import NWPSource

    resolved = settings or get_settings()
    sources: dict[str, WeatherDataSource] = {
        "netcdf": NetCDFSource(),
        "grib": GRIBSource(),
        "era5": ERA5Source(),
        "nwp": NWPSource(),
    }
    return sources

