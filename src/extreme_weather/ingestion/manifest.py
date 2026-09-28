"""Dataset manifests and provenance (SIH26078, Phase 6).

A manifest is the answer to "where did this file come from and what was done to
it?". Without one, a processed dataset is an anonymous array.

Manifest layout::

    data/manifests/<dataset_id>.json

Two rules this module enforces:

1. **Never invent a source URL.** A dataset that arrived as a local file has
   ``source="local_file"`` and ``source_url=None``. Writing a plausible-looking
   URL because the file *looks* like it came from somewhere would be a lie in
   a file whose entire purpose is to be trusted.
2. **Hash the bytes, not the interpretation.** ``file_hash`` is the SHA-256 of
   the file, so it can be recomputed by anyone. Dataset-level ``content_hash``
   is a stable digest of the array values, so it survives a NetCDF rewrite that
   changes the byte layout but not the data.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from extreme_weather.config import KM_PER_DEGREE_LATITUDE, PROJECT_ID, PROJECT_NAME
from extreme_weather.ingestion.base import DataOrigin, utc_now
from extreme_weather.ingestion.exceptions import IngestionError, SourceNotFoundError

__all__ = [
    "MANIFEST_SCHEMA_VERSION",
    "SOURCE_LOCAL_FILE",
    "DatasetManifest",
    "ManifestStore",
    "ProvenanceStep",
    "content_hash",
    "file_hash",
]


#: Bumped when the manifest layout changes incompatibly.
MANIFEST_SCHEMA_VERSION: str = "1.0"

#: The value used when a dataset was supplied as a local file.
SOURCE_LOCAL_FILE: str = "local_file"

#: Read this many bytes at a time when hashing a large file.
_HASH_CHUNK_BYTES: int = 1024 * 1024


def file_hash(path: Path | str, algorithm: str = "sha256") -> str:
    """Return the hex digest of a file, read in chunks.

    Chunked so a multi-gigabyte reanalysis file is hashed without loading it
    into memory.
    """
    digest = hashlib.new(algorithm)
    try:
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
                digest.update(block)
    except OSError as exc:
        raise IngestionError(f"Could not hash '{path}': {exc}") from exc
    return digest.hexdigest()


def content_hash(dataset: xr.Dataset, algorithm: str = "sha256") -> str:
    """Return a stable digest of a dataset's array values.

    Unlike the file hash this survives a rewrite that changes the byte layout
    (compression, chunking, attribute order) but not the data - which is what
    makes it usable for "did normalization change anything?" checks.
    """
    digest = hashlib.new(algorithm)
    for name in sorted(str(key) for key in dataset.data_vars):
        digest.update(name.encode("utf-8"))
        variable = dataset[name]
        values = np.asarray(variable.values)
        digest.update(str(values.dtype).encode("utf-8"))
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ProvenanceStep:
    """One recorded processing step in the ingestion chain.

    Attributes:
        stage: One of ``ingested``, ``validated``, ``normalized``,
            ``processed``, ``analysis``.
        tool: The component that performed the step.
        version: Its version string, when known.
        input_hash: Hash of the input, when one was computed.
        output_hash: Hash of the output, when one was computed.
        configuration: Non-secret settings the step ran with.
        performed_at: When the step ran.
        details: Free-form human-readable notes.
    """

    stage: str
    tool: str
    version: str = ""
    input_hash: str | None = None
    output_hash: str | None = None
    configuration: dict[str, Any] = field(default_factory=dict)
    performed_at: datetime | None = None
    details: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "stage": self.stage,
            "tool": self.tool,
            "version": self.version,
            "input_hash": self.input_hash,
            "output_hash": self.output_hash,
            "configuration": self.configuration,
            "performed_at": self.performed_at.isoformat() if self.performed_at else None,
            "details": self.details,
        }


@dataclass(frozen=True)
class DatasetManifest:
    """Everything known about one ingested dataset.

    The field set is fixed on purpose: a manifest that grows a field per caller
    stops being comparable between datasets.
    """

    dataset_id: str
    source: str
    retrieved_at: datetime
    file_name: str
    file_hash: str
    format: str
    variables: tuple[str, ...] = ()
    dimensions: dict[str, int] = field(default_factory=dict)
    resolution_deg: float | None = None
    resolution_km: float | None = None
    time_start: str | None = None
    time_end: str | None = None
    time_step_seconds: float | None = None
    coordinate_system: str = "latitude_longitude"
    units: dict[str, str] = field(default_factory=dict)
    processing_steps: tuple[ProvenanceStep, ...] = ()
    quality_summary: dict[str, Any] = field(default_factory=dict)
    origin: str = DataOrigin.HISTORICAL
    source_url: str | None = None
    license: str | None = None
    content_hash: str | None = None
    size_bytes: int | None = None
    variables_missing: tuple[str, ...] = ()
    coordinate_actions: tuple[str, ...] = ()
    unit_conversions: dict[str, dict[str, Any]] = field(default_factory=dict)
    notes: str = ""
    schema_version: str = MANIFEST_SCHEMA_VERSION
    project: str = PROJECT_ID
    service: str = PROJECT_NAME

    def __post_init__(self) -> None:
        # Enforce the "never invent a source URL" rule at construction time, so
        # no call site can bypass it.
        if self.source == SOURCE_LOCAL_FILE and self.source_url:
            raise ValueError(
                "A local_file dataset must not carry a source_url; that would "
                "invent provenance. Set source to the real provider name instead."
            )
        if not self.source_url:
            object.__setattr__(self, "source_url", None)

    @property
    def is_synthetic(self) -> bool:
        """True only for data this project generated itself."""
        return self.origin == DataOrigin.SYNTHETIC

    @property
    def is_live(self) -> bool:
        """Always False.

        This project ingests *files*. A file fetched now is still a historical
        artefact, and calling it "live" is exactly the confusion this property
        exists to prevent.
        """
        return False

    @property
    def is_historical(self) -> bool:
        """True for real past data (HISTORICAL or EXTERNAL), false for synthetic."""
        return not self.is_synthetic

    @property
    def status_label(self) -> str:
        """A short badge label for the dashboard.

        Deliberately three distinct words, because the difference matters:
        ``SYNTHETIC`` is generated, ``HISTORICAL`` is a real past period and
        ``EXTERNAL`` is a real source fetched at run time. None of them is
        "live", and none of them is a forecast.
        """
        return {
            DataOrigin.SYNTHETIC: "SYNTHETIC",
            DataOrigin.HISTORICAL: "HISTORICAL",
            DataOrigin.EXTERNAL: "REAL / EXTERNAL",
        }.get(self.origin, self.origin)

    def add_step(self, step: ProvenanceStep) -> "DatasetManifest":
        """Return a copy with one more provenance step appended."""
        return DatasetManifest(
            **{
                **{key: getattr(self, key) for key in self.__dataclass_fields__},
                "processing_steps": (*self.processing_steps, step),
            }
        )


    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary with a stable key order."""
        return {
            "schema_version": self.schema_version,
            "project": self.project,
            "service": self.service,
            "dataset_id": self.dataset_id,
            "source": self.source,
            "source_url": self.source_url,
            "retrieved_at": self.retrieved_at.isoformat(),
            "file_name": self.file_name,
            "file_hash": self.file_hash,
            "file_hash_algorithm": "sha256",
            "content_hash": self.content_hash,
            "size_bytes": self.size_bytes,
            "format": self.format,
            "origin": self.origin,
            "is_synthetic": self.is_synthetic,
            "is_live": self.is_live,
            "status_label": self.status_label,
            "variables": list(self.variables),
            "variables_missing": list(self.variables_missing),
            "dimensions": dict(self.dimensions),
            "resolution_deg": self.resolution_deg,
            "resolution_km": self.resolution_km,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "time_step_seconds": self.time_step_seconds,
            "coordinate_system": self.coordinate_system,
            "units": dict(self.units),
            "unit_conversions": dict(self.unit_conversions),
            "coordinate_actions": list(self.coordinate_actions),
            "license": self.license,
            "quality_summary": dict(self.quality_summary),
            "processing_steps": [step.as_dict() for step in self.processing_steps],
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DatasetManifest":
        """Rebuild a manifest from :meth:`as_dict` output."""
        steps = tuple(
            ProvenanceStep(
                stage=str(item.get("stage", "")),
                tool=str(item.get("tool", "")),
                version=str(item.get("version", "")),
                input_hash=item.get("input_hash"),
                output_hash=item.get("output_hash"),
                configuration=dict(item.get("configuration", {})),
                performed_at=(
                    datetime.fromisoformat(item["performed_at"])
                    if item.get("performed_at")
                    else None
                ),
                details=str(item.get("details", "")),
            )
            for item in payload.get("processing_steps", [])
        )
        return cls(
            dataset_id=str(payload.get("dataset_id", "")),
            source=str(payload.get("source", SOURCE_LOCAL_FILE)),
            retrieved_at=datetime.fromisoformat(
                str(payload.get("retrieved_at") or utc_now().isoformat())
            ),
            file_name=str(payload.get("file_name", "")),
            file_hash=str(payload.get("file_hash", "")),
            format=str(payload.get("format", "")),
            variables=tuple(payload.get("variables", ())),
            dimensions=dict(payload.get("dimensions", {})),
            resolution_deg=payload.get("resolution_deg"),
            resolution_km=payload.get("resolution_km"),
            time_start=payload.get("time_start"),
            time_end=payload.get("time_end"),
            time_step_seconds=payload.get("time_step_seconds"),
            coordinate_system=str(payload.get("coordinate_system", "latitude_longitude")),
            units=dict(payload.get("units", {})),
            processing_steps=steps,
            quality_summary=dict(payload.get("quality_summary", {})),
            origin=str(payload.get("origin", DataOrigin.HISTORICAL)),
            source_url=payload.get("source_url"),
            license=payload.get("license"),
            content_hash=payload.get("content_hash"),
            size_bytes=payload.get("size_bytes"),
            variables_missing=tuple(payload.get("variables_missing", ())),
            coordinate_actions=tuple(payload.get("coordinate_actions", ())),
            unit_conversions=dict(payload.get("unit_conversions", {})),
            notes=str(payload.get("notes", "")),
        )



def build_manifest(
    dataset: xr.Dataset,
    *,
    dataset_id: str,
    source_path: Path | str | None,
    source: str,
    origin: str = DataOrigin.HISTORICAL,
    source_url: str | None = None,
    license_name: str | None = None,
    format_name: str = "netcdf",
    variables_missing: Sequence[str] = (),
    coordinate_actions: Sequence[str] = (),
    unit_conversions: Mapping[str, Mapping[str, Any]] | None = None,
    quality_summary: Mapping[str, Any] | None = None,
    notes: str = "",
    retrieved_at: datetime | None = None,
) -> DatasetManifest:
    """Assemble a :class:`DatasetManifest` from a dataset and its source.

    Coordinate extent, resolution and time range are read from the dataset, so
    the manifest describes what is actually in the file rather than what the
    caller intended to put there.
    """
    path = Path(source_path) if source_path is not None else None
    dimensions = {str(key): int(value) for key, value in dataset.sizes.items()}

    spacings: list[float] = []
    for name in ("latitude", "longitude"):
        if name in dataset.coords and dataset.coords[name].ndim == 1:
            values = np.asarray(dataset.coords[name].values, dtype="float64")
            if values.size >= 2:
                spacings.append(float(np.mean(np.abs(np.diff(values)))))
    resolution_deg = round(float(np.mean(spacings)), 6) if spacings else None

    time_start = time_end = None
    step_seconds = None
    if "time" in dataset.coords and dataset.coords["time"].ndim == 1:
        stamps = np.asarray(dataset.coords["time"].values)
        if stamps.size:
            time_start = str(np.datetime_as_string(stamps[0], unit="s"))
            time_end = str(np.datetime_as_string(stamps[-1], unit="s"))
        if stamps.size >= 2 and np.issubdtype(stamps.dtype, np.datetime64):
            deltas = np.diff(stamps).astype("timedelta64[s]").astype("int64")
            step_seconds = float(np.mean(deltas))

    units = {
        str(name): str(dataset[name].attrs.get("units", ""))
        for name in dataset.data_vars
        if dataset[name].attrs.get("units") not in (None, "")
    }

    return DatasetManifest(
        dataset_id=dataset_id,
        source=source,
        retrieved_at=retrieved_at or utc_now(),
        file_name=path.name if path is not None else "",
        file_hash=file_hash(path) if path is not None and path.is_file() else "",
        content_hash=content_hash(dataset),
        size_bytes=path.stat().st_size if path is not None and path.is_file() else None,
        format=format_name,
        origin=origin,
        source_url=source_url,
        license=license_name,
        variables=tuple(sorted(str(name) for name in dataset.data_vars)),
        variables_missing=tuple(variables_missing),
        dimensions=dimensions,
        resolution_deg=resolution_deg,
        resolution_km=(
            round(resolution_deg * KM_PER_DEGREE_LATITUDE, 3) if resolution_deg else None
        ),
        time_start=time_start,
        time_end=time_end,
        time_step_seconds=step_seconds,
        units=units,
        unit_conversions=dict(unit_conversions or {}),
        coordinate_actions=tuple(coordinate_actions),
        quality_summary=dict(quality_summary or {}),
        notes=notes,
    )



class ManifestStore:
    """Read and write manifests under ``data/manifests``.

    One JSON file per dataset id, so a manifest is inspectable with ``cat`` and
    a single corrupt manifest cannot take down the whole registry.
    """

    def __init__(self, directory: Path | str | None = None) -> None:
        from extreme_weather.config import get_settings

        settings = get_settings()
        self._directory = (
            Path(directory) if directory is not None else settings.manifests_dir
        )

    @property
    def directory(self) -> Path:
        """Directory holding the manifest files."""
        return self._directory

    def path_for(self, dataset_id: str) -> Path:
        """Return the manifest path of one dataset id."""
        return self._directory / f"{dataset_id}.json"

    def save(self, manifest: DatasetManifest) -> Path:
        """Write a manifest, creating the directory as needed.

        Raises:
            IngestionError: If the manifest cannot be written.
        """
        try:
            self._directory.mkdir(parents=True, exist_ok=True)
            target = self.path_for(manifest.dataset_id)
            target.write_text(
                json.dumps(manifest.as_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            raise IngestionError(
                f"Could not write manifest for '{manifest.dataset_id}': {exc}"
            ) from exc
        return self.path_for(manifest.dataset_id)

    def load(self, dataset_id: str) -> DatasetManifest | None:
        """Read one manifest, or return ``None`` when it does not exist.

        Raises:
            IngestionError: If the file exists but cannot be parsed.
        """
        target = self.path_for(dataset_id)
        if not target.is_file():
            return None
        try:
            payload = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise IngestionError(f"Could not read manifest '{target}': {exc}") from exc
        return DatasetManifest.from_dict(payload)

    def list_ids(self) -> list[str]:
        """Return every dataset id that has a manifest, sorted."""
        if not self._directory.is_dir():
            return []
        return sorted(path.stem for path in self._directory.glob("*.json"))

    def list(self) -> list[DatasetManifest]:
        """Read every readable manifest, skipping corrupt ones."""
        manifests: list[DatasetManifest] = []
        for dataset_id in self.list_ids():
            try:
                manifest = self.load(dataset_id)
            except IngestionError:
                continue
            if manifest is not None:
                manifests.append(manifest)
        return manifests

    def remove(self, dataset_id: str) -> bool:
        """Delete a manifest; returns True when a file was actually removed."""
        target = self.path_for(dataset_id)
        if not target.is_file():
            return False
        target.unlink()
        return True

    def find_by_time(self, start: str | None = None, end: str | None = None) -> list[DatasetManifest]:
        """Return manifests whose time range overlaps ``[start, end]``.

        Args:
            start: ISO 8601 inclusive lower bound, or ``None`` for open.
            end: ISO 8601 inclusive upper bound, or ``None`` for open.

        Returns:
            Matching manifests, sorted by dataset id.
        """
        matches: list[DatasetManifest] = []
        for manifest in self.list():
            if _overlaps(manifest.time_start, manifest.time_end, start, end):
                matches.append(manifest)
        return sorted(matches, key=lambda item: item.dataset_id)



def _overlaps(
    dataset_start: str | None,
    dataset_end: str | None,
    query_start: str | None,
    query_end: str | None,
) -> bool:
    """Return True when two ISO 8601 intervals overlap.

    An interval with an unknown end (``None``) is treated as open-ended, so a
    manifest that never recorded a time range is not silently filtered out of
    every time query.
    """
    if dataset_start is None and dataset_end is None:
        return True
    start = dataset_start or query_start
    end = dataset_end or query_end
    if query_start is not None and end is not None and end < query_start:
        return False
    if query_end is not None and start is not None and start > query_end:
        return False
    return True

