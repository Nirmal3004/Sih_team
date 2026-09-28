"""Run provenance for the Phase 7 validation pipeline (SIH26078).

A validation number without provenance is a rumour. This module records, for
every Phase 7 run: which dataset, which split, which software version, which
configuration, and which stages completed - so any figure in the report can be
traced back to the exact bytes and code that produced it.

Records are written to ``data/validation/provenance/<run_id>.json`` so a run
stays auditable even after the working tree moves on.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from extreme_weather.config import PACKAGE_VERSION, PROJECT_ID, PROJECT_NAME
from extreme_weather.ingestion.base import utc_now

__all__ = [
    "PROVENANCE_SCHEMA",
    "ProvenanceRecord",
    "ValidationRunRecord",
    "read_run_record",
    "write_run_record",
]


#: Written into every record so a reader knows what these files are.
PROVENANCE_SCHEMA = "phase7_provenance/1.0"


@dataclass(frozen=True)
class ProvenanceRecord:
    """The provenance of one input to a validation run.

    Attributes:
        dataset_id: Which dataset was used.
        source: Where it came from (``local_file``, a provider name ...).
        source_url: The provider URL, or ``None`` for a local file.
        file_hash: SHA-256 of the file.
        content_hash: Digest of the array values.
        time_start: Earliest timestamp covered.
        time_end: Latest timestamp covered.
        variables: Variables present.
        resolution_deg: Grid resolution in degrees.
        coordinate_system: Coordinate convention.
        origin: ``SYNTHETIC`` / ``HISTORICAL`` / ``EXTERNAL``.
        is_synthetic: True only for project-generated data.
        is_live: Always False: this project ingests files, never a live feed.
        manifest_path: Where the Phase 6 manifest lives.
    """

    dataset_id: str
    source: str = "local_file"
    source_url: str | None = None
    file_hash: str = ""
    content_hash: str | None = None
    time_start: str | None = None
    time_end: str | None = None
    variables: tuple[str, ...] = ()
    resolution_deg: float | None = None
    coordinate_system: str = "latitude_longitude"
    origin: str = "HISTORICAL"
    is_synthetic: bool = False
    is_live: bool = False
    manifest_path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "dataset_id": self.dataset_id,
            "source": self.source,
            "source_url": self.source_url,
            "file_hash": self.file_hash,
            "content_hash": self.content_hash,
            "time_start": self.time_start,
            "time_end": self.time_end,
            "variables": list(self.variables),
            "resolution_deg": self.resolution_deg,
            "coordinate_system": self.coordinate_system,
            "origin": self.origin,
            "is_synthetic": self.is_synthetic,
            "is_live": self.is_live,
            "manifest_path": self.manifest_path,
        }

    @classmethod
    def from_manifest(cls, manifest: Any) -> "ProvenanceRecord":
        """Build a record from a Phase 6 :class:`DatasetManifest`."""
        return cls(
            dataset_id=manifest.dataset_id,
            source=manifest.source,
            source_url=manifest.source_url,
            file_hash=manifest.file_hash,
            content_hash=manifest.content_hash,
            time_start=manifest.time_start,
            time_end=manifest.time_end,
            variables=tuple(manifest.variables),
            resolution_deg=manifest.resolution_deg,
            coordinate_system=manifest.coordinate_system,
            origin=manifest.origin,
            is_synthetic=manifest.is_synthetic,
            is_live=manifest.is_live,
            manifest_path=(
                f"data/manifests/{manifest.dataset_id}.json"
                if manifest.file_name
                else None
            ),
        )


@dataclass
class ValidationRunRecord:
    """Everything needed to reproduce one validation run.

    Not frozen: the pipeline appends stage outcomes as it goes, which is what
    makes a failed run diagnosable after the fact.

    Attributes:
        run_id: Identifier of the run.
        started_at: When the run began.
        dataset: Provenance of the dataset.
        finished_at: When it ended.
        package_version: The project version that produced the numbers.
        split: The split summary.
        configuration: The non-secret settings the run used.
        stages: Stage name -> status (``"ok"``, ``"skipped"``, ``"failed"``).
        stage_details: Stage name -> extra detail.
        environment: Reproducibility facts (Python, NumPy, Xarray versions).
        notes: Anything a reader must know.
    """

    run_id: str
    started_at: datetime
    dataset: ProvenanceRecord
    finished_at: datetime | None = None
    package_version: str = PACKAGE_VERSION
    project: str = PROJECT_ID
    service: str = PROJECT_NAME
    split: dict[str, Any] = field(default_factory=dict)
    configuration: dict[str, Any] = field(default_factory=dict)
    stages: dict[str, str] = field(default_factory=dict)
    stage_details: dict[str, Any] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        """True when every recorded stage finished with ``"ok"``."""
        return bool(self.stages) and all(
            status == "ok" for status in self.stages.values()
        )

    def record_stage(self, name: str, status: str, **details: Any) -> None:
        """Record the outcome of one stage."""
        self.stages[name] = status
        if details:
            self.stage_details[name] = details

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary."""
        return {
            "schema": PROVENANCE_SCHEMA,
            "run_id": self.run_id,
            "project": self.project,
            "service": self.service,
            "package_version": self.package_version,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "complete": self.complete,
            "dataset": self.dataset.as_dict(),
            "split": dict(self.split),
            "configuration": dict(self.configuration),
            "stages": dict(self.stages),
            "stage_details": dict(self.stage_details),
            "environment": dict(self.environment),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ValidationRunRecord":
        """Rebuild a record from :meth:`as_dict` output."""
        dataset = payload.get("dataset", {})
        return cls(
            run_id=str(payload.get("run_id", "unknown")),
            started_at=datetime.fromisoformat(
                str(payload.get("started_at") or utc_now().isoformat())
            ),
            finished_at=(
                datetime.fromisoformat(payload["finished_at"])
                if payload.get("finished_at")
                else None
            ),
            package_version=str(payload.get("package_version", "")),
            project=str(payload.get("project", PROJECT_ID)),
            service=str(payload.get("service", PROJECT_NAME)),
            dataset=ProvenanceRecord(
                dataset_id=str(dataset.get("dataset_id", "")),
                source=str(dataset.get("source", "local_file")),
                source_url=dataset.get("source_url"),
                file_hash=str(dataset.get("file_hash", "")),
                content_hash=dataset.get("content_hash"),
                time_start=dataset.get("time_start"),
                time_end=dataset.get("time_end"),
                variables=tuple(dataset.get("variables", ())),
                resolution_deg=dataset.get("resolution_deg"),
                coordinate_system=str(
                    dataset.get("coordinate_system", "latitude_longitude")
                ),
                origin=str(dataset.get("origin", "HISTORICAL")),
                is_synthetic=bool(dataset.get("is_synthetic", False)),
                is_live=bool(dataset.get("is_live", False)),
                manifest_path=dataset.get("manifest_path"),
            ),
            split=dict(payload.get("split", {})),
            configuration=dict(payload.get("configuration", {})),
            stages=dict(payload.get("stages", {})),
            stage_details=dict(payload.get("stage_details", {})),
            environment=dict(payload.get("environment", {})),
            notes=tuple(payload.get("notes", ())),
        )



def write_run_record(record: ValidationRunRecord, directory: Path | str) -> Path:
    """Write a run record as JSON and return its path.

    Args:
        record: The record to persist.
        directory: Where to write it (created if missing).

    Returns:
        The path written.

    Raises:
        OSError: If the directory or file cannot be written.
    """
    target = Path(directory) / f"{record.run_id}.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(record.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return target


def read_run_record(path: Path | str) -> ValidationRunRecord | None:
    """Read a run record, or return ``None`` when the file does not exist.

    Raises:
        ValueError: If the file exists but cannot be parsed.
    """
    target = Path(path)
    if not target.is_file():
        return None
    return ValidationRunRecord.from_dict(json.loads(target.read_text(encoding="utf-8")))


def environment_facts() -> dict[str, str]:
    """Return the library versions a reproduction would need."""
    import platform

    import numpy
    import xarray

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": numpy.__version__,
        "xarray": xarray.__version__,
    }

