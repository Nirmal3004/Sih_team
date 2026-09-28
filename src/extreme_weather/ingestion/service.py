"""Ingestion service - the business layer for real / historical data (SIH26078, Phase 6).

The service is what the CLI and the API both call, so neither contains pipeline
logic. Its job is to take a *raw* file from any registered source and produce a
**registered, validated, normalized, manifested** dataset that Phase 1-5 can
consume unchanged.

The chain, with what each step guarantees::

    source.fetch()      a real path, or a clear error (no fabricated data)
      -> service.load()  opened in the source's own units
      -> check_quality() a DataQualityReport: valid / errors / warnings
      -> normalize()     standard units and coordinates, fully recorded
      -> build_manifest() provenance, hashes, origin
      -> registry        a queryable entry

Two properties this service is careful about:

* **Synthetic and real never mix.** A caller states the origin, and the service
  writes to ``raw/synthetic`` or ``raw/real`` accordingly. There is no default
  that could quietly put a real file in the synthetic folder.
* **Every write is optional.** ``dry_run`` performs the whole chain and returns
  what *would* happen without touching a single file, so a request can be
  previewed against a populated directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from extreme_weather.config import PACKAGE_VERSION, Settings, get_settings
from extreme_weather.ingestion.base import DataOrigin, WeatherDataSource, utc_now
from extreme_weather.ingestion.exceptions import (
    IngestionError,
    SourceNotFoundError,
)
from extreme_weather.ingestion.manifest import (
    DatasetManifest,
    ManifestStore,
    ProvenanceStep,
    SOURCE_LOCAL_FILE,
    build_manifest,
    content_hash,
)
from extreme_weather.ingestion.normalizer import NormalizationResult, normalize_dataset
from extreme_weather.ingestion.quality import DataQualityReport, check_quality
from extreme_weather.ingestion.registry import DatasetEntry, DatasetRegistry

__all__ = [
    "IngestionRequest",
    "IngestionResult",
    "IngestionService",
    "derive_uv_from_polar_wind",
    "subset_dataset",
]


def derive_uv_from_polar_wind(dataset: xr.Dataset) -> tuple[xr.Dataset, list[str]]:
    """Add ``u_wind``/``v_wind`` when only speed and direction are present.

    Archives such as the Open-Meteo point API publish ``wind_speed`` and
    ``wind_direction`` rather than components, while the Phase 0 data contract -
    and therefore Phases 1-5 - requires ``u_wind``/``v_wind``. The conversion is
    the standard, reversible one (see
    :func:`extreme_weather.data.derived.uv_from_speed_direction`), and it is
    recorded in the returned action list so the manifest states that these
    components were *derived*, not observed.

    A dataset that already carries components, or that carries neither, is
    returned unchanged - the derived variables are a convenience, not a
    requirement, and a missing wind field is reported rather than invented.
    """
    from extreme_weather.data.derived import (
        WIND_DIRECTION_NAME,
        WIND_SPEED_NAME,
        uv_from_speed_direction,
    )

    actions: list[str] = []
    if "u_wind" in dataset.data_vars and "v_wind" in dataset.data_vars:
        return dataset, actions
    if WIND_SPEED_NAME not in dataset.data_vars or WIND_DIRECTION_NAME not in dataset.data_vars:
        return dataset, actions

    u_wind, v_wind = uv_from_speed_direction(
        dataset[WIND_SPEED_NAME], dataset[WIND_DIRECTION_NAME]
    )
    actions.append(
        f"derived {WIND_SPEED_NAME} + {WIND_DIRECTION_NAME} into u_wind/v_wind "
        "(meteorological convention, standard reversible conversion)"
    )
    return dataset.assign(u_wind=u_wind, v_wind=v_wind), actions



def subset_dataset(
    dataset: xr.Dataset,
    *,
    latitude: tuple[float, float] | None = None,
    longitude: tuple[float, float] | None = None,
) -> xr.Dataset:
    """Restrict a dataset to a latitude/longitude box.

    Uses ``.sel(..., method="nearest")`` on the edges and then a boolean cut, so
    the result is a real subset with a slightly larger extent rather than a
    silently shifted one. Returns the dataset unchanged when no box is given.
    """
    result = dataset
    if latitude is not None:
        low, high = latitude
        result = result.sel(latitude=slice(low, high))
    if longitude is not None:
        low, high = longitude
        result = result.sel(longitude=slice(low, high))
    return result


@dataclass(frozen=True)
class IngestionRequest:
    """One ingestion request.

    Attributes:
        dataset_id: Stable id the dataset will be registered under.
        source_name: Which registered source to use (default ``netcdf``).
        path: Local source file, for a file-based source.
        origin: ``SYNTHETIC``, ``HISTORICAL`` or ``EXTERNAL``. Required - there
            is no default, because guessing this is how real data ends up
            presented as synthetic or vice versa.
        variables: Optional whitelist of canonical variable names.
        latitude: Optional ``(min, max)`` latitude box.
        longitude: Optional ``(min, max)`` longitude box.
        strict: Fail when the quality report has errors (default) or only warn.
        dry_run: Perform the whole chain without writing anything.
        source_url: Optional real URL, only for a named remote provider.
        license_name: Optional licence string for the manifest.
        notes: Free-form notes stored in the manifest.
    """

    dataset_id: str
    origin: str
    source_name: str = "netcdf"
    path: Path | str | None = None
    variables: tuple[str, ...] = ()
    latitude: tuple[float, float] | None = None
    longitude: tuple[float, float] | None = None
    strict: bool = True
    dry_run: bool = False
    source_url: str | None = None
    license_name: str | None = None
    notes: str = ""

    def __post_init__(self) -> None:
        if not str(self.dataset_id).strip():
            raise ValueError("dataset_id must not be empty.")
        # Validates the origin eagerly: a bad value must fail at the boundary.
        DataOrigin.validate(self.origin)


@dataclass(frozen=True)
class IngestionResult:
    """Outcome of one ingestion run."""

    dataset_id: str
    manifest: DatasetManifest
    quality: DataQualityReport
    normalization: NormalizationResult | None = None
    entry: DatasetEntry | None = None
    dry_run: bool = False
    output_path: Path | None = None
    steps: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    degraded: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable summary (no array data)."""
        return {
            "dataset_id": self.dataset_id,
            "manifest": self.manifest.as_dict(),
            "quality": self.quality.as_dict(),
            "normalization": (
                self.normalization.as_dict() if self.normalization is not None else None
            ),
            "dry_run": self.dry_run,
            "output_path": str(self.output_path) if self.output_path else None,
            "steps": list(self.steps),
            "warnings": list(self.warnings),
            "degraded": list(self.degraded),
        }


class IngestionService:
    """Ingest a real or synthetic dataset and register it.

    Args:
        sources: The registered sources, by name. Defaults to the built-ins.
        registry: Where manifests are written. Defaults to
            ``<DATA_DIR>/manifests``.
        settings: Application settings (directories, limits, defaults).
    """

    def __init__(
        self,
        sources: Mapping[str, WeatherDataSource] | None = None,
        registry: DatasetRegistry | None = None,
        settings: Settings | None = None,
    ) -> None:
        from extreme_weather.ingestion.registry import default_sources

        self._settings = settings or get_settings()
        self._sources: dict[str, WeatherDataSource] = dict(
            sources if sources is not None else default_sources(self._settings)
        )
        self._registry = registry or DatasetRegistry(
            ManifestStore(self._settings.manifests_dir), settings=self._settings
        )

    @property
    def sources(self) -> dict[str, WeatherDataSource]:
        """The registered sources, by name."""
        return dict(self._sources)

    @property
    def registry(self) -> DatasetRegistry:
        """The dataset registry this service writes to."""
        return self._registry

    def register_source(self, source: WeatherDataSource) -> None:
        """Add or replace a source by its ``name()``."""
        self._sources[source.name()] = source

    def get_source(self, name: str) -> WeatherDataSource:
        """Return one source by name.

        Raises:
            SourceNotFoundError: If the source is not registered.
        """
        source = self._sources.get(name)
        if source is None:
            known = ", ".join(sorted(self._sources)) or "none"
            raise SourceNotFoundError(
                f"Data source '{name}' is not registered. Available: {known}."
            )
        return source

    def _raw_dir(self, origin: str) -> Path:
        """Return the raw directory for an origin, enforcing the separation."""
        return (
            self._settings.synthetic_raw_dir
            if origin == DataOrigin.SYNTHETIC
            else self._settings.real_raw_dir
        )

    def ingest(
        self,
        request: IngestionRequest,
        *,
        now: datetime | None = None,
    ) -> IngestionResult:
        """Run the full ingestion chain for one request.

        Args:
            request: What to ingest and how.
            now: Timestamp for the provenance steps.

        Returns:
            An :class:`IngestionResult` describing what happened.

        Raises:
            SourceNotFoundError: If the source or the file does not exist.
            ValueError: If the request carries no path.
            IngestionError: If the dataset cannot be read, normalized or written.
        """
        stamp = now or utc_now()
        origin = DataOrigin.validate(request.origin)
        source = self.get_source(request.source_name)
        steps: list[str] = []
        warnings: list[str] = []
        degraded: list[str] = []

        if request.path is None:
            raise ValueError(
                "IngestionRequest.path is required; the local-file path is the "
                "supported ingestion route."
            )

        # --- 1. fetch --------------------------------------------------- #
        raw_path = source.fetch(request.path)  # type: ignore[arg-type]
        steps.append(f"ingested via source '{source.name()}'")
        input_digest = ""
        try:
            from extreme_weather.ingestion.manifest import file_hash

            input_digest = file_hash(raw_path)
        except IngestionError:
            degraded.append("Could not hash the source file.")

        # --- 2. load ----------------------------------------------------- #
        opened = source.load(raw_path)
        try:
            loaded = opened.load()
        finally:
            try:
                opened.close()
            except Exception:  # noqa: BLE001 - closing must not mask the body
                pass
        steps.append("loaded in source units")

        # --- 3. subset --------------------------------------------------- #
        latitude = request.latitude or self._settings.ingestion_latitude_range
        longitude = request.longitude or self._settings.ingestion_longitude_range
        if latitude or longitude:
            loaded = subset_dataset(loaded, latitude=latitude, longitude=longitude)
            steps.append(f"subset to lat={latitude} lon={longitude}")

        # --- 4. quality, as delivered ------------------------------------- #
        check_quality(loaded, generated_at=stamp)
        steps.append("quality checked (source units)")

        # --- 5. normalize ------------------------------------------------- #
        variables = request.variables or tuple(self._settings.ingestion_variable_list)
        # Keep the polar wind inputs alongside the whitelist: the derived u/v
        # depend on them, and dropping them first would lose the source's values.
        keep = set(variables)
        if "u_wind" in keep or "v_wind" in keep:
            keep.update({"wind_speed", "wind_direction"})
        normalization = normalize_dataset(
            loaded, variables=tuple(sorted(keep)), drop_unknown=True, generated_at=stamp
        )
        steps.append("normalized to standard units and coordinates")

        # --- 5b. polar wind -> u/v components ----------------------------- #
        # Applied AFTER normalization, so the canonical ``wind_speed`` /
        # ``wind_direction`` names are what the conversion looks for. Some
        # archives publish speed+direction rather than components; the Phase 0
        # contract needs components, so the standard reversible conversion is
        # applied - and recorded, so the manifest never claims the components
        # were observed rather than derived.
        derived_actions: list[str] = []
        if "u_wind" in keep or "v_wind" in keep:
            enriched, derived_actions = derive_uv_from_polar_wind(normalization.dataset)
            normalization = replace(
                normalization,
                dataset=enriched,
                coordinate_actions=(
                    *normalization.coordinate_actions,
                    *derived_actions,
                ),
            )
            steps.extend(derived_actions)

        # The polar intermediates have served their purpose; the contract wants
        # components, and carrying both would be redundant in the processed file.
        drop_after_derivation = [
            name
            for name in ("wind_speed", "wind_direction")
            if name in normalization.dataset.data_vars
            and "u_wind" in normalization.dataset.data_vars
            and "v_wind" in normalization.dataset.data_vars
        ]
        if drop_after_derivation:
            normalization = replace(
                normalization,
                dataset=normalization.dataset.drop_vars(drop_after_derivation),
                dropped_variables=(*normalization.dropped_variables, *drop_after_derivation),
                warnings=(
                    *normalization.warnings,
                    "dropped the polar wind intermediates after deriving u_wind/v_wind",
                ),
            )

        # --- 6. quality, after normalization ------------------------------ #
        report = check_quality(normalization.dataset, generated_at=stamp)
        steps.append("quality checked (normalized)")
        warnings.extend(normalization.warnings)


        # --- 7. write the normalized dataset ------------------------------ #
        target_dir = self._raw_dir(origin)
        target = target_dir / f"{request.dataset_id}.nc"
        output_path: Path | None = None
        if request.dry_run:
            steps.append(f"dry run: would write {target}")
        else:
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                normalization.dataset.to_netcdf(target, mode="w")
            except OSError as exc:
                raise IngestionError(
                    f"Could not write the ingested dataset to '{target}': {exc}"
                ) from exc
            steps.append(f"wrote {target}")
            output_path = target

        # --- 8. manifest and registry ------------------------------------- #
        manifest = self._build_manifest(
            request,
            normalization,
            report,
            stamp,
            input_digest,
            steps,
            source.name(),
            target,
        )
        entry: DatasetEntry | None = None
        if not request.dry_run:
            entry = self._registry.register(manifest, report)
            steps.append("registered in the dataset registry")

        if request.strict and not report.valid:
            degraded.append(
                f"Quality check found {report.error_count} blocking problem(s); "
                "the dataset is registered but must not be analysed."
            )

        return IngestionResult(
            dataset_id=request.dataset_id,
            manifest=manifest,
            quality=report,
            normalization=normalization,
            entry=entry,
            dry_run=request.dry_run,
            output_path=output_path,
            steps=tuple(steps),
            warnings=tuple(warnings),
            degraded=tuple(degraded),
        )

    def _build_manifest(
        self,
        request: IngestionRequest,
        normalization: NormalizationResult,
        report: DataQualityReport,
        stamp: datetime,
        input_digest: str,
        steps: Sequence[str],
        source_name: str,
        target: Path | None = None,
    ) -> DatasetManifest:
        """Assemble the manifest, including the full provenance chain."""
        origin = DataOrigin.validate(request.origin)
        # A local file has no URL. Only a named remote provider may carry one,
        # and the manifest refuses the combination rather than inventing it.
        source = SOURCE_LOCAL_FILE if source_name in ("netcdf", "grib", "nwp") else source_name
        source_url = request.source_url if source != SOURCE_LOCAL_FILE else None

        dataset = normalization.dataset
        manifest = build_manifest(
            dataset,
            dataset_id=request.dataset_id,
            source_path=target,
            source=source,
            origin=origin,
            source_url=source_url,
            license_name=request.license_name,
            format_name="netcdf",
            variables_missing=normalization.missing_variables,
            coordinate_actions=normalization.coordinate_actions,
            unit_conversions=normalization.conversion_map(),
            quality_summary=report.as_dict(),
            notes=request.notes,
            retrieved_at=stamp,
        )

        chain = (
            ProvenanceStep(
                stage="ingested",
                tool=f"extreme_weather.ingestion.{source_name}",
                version=PACKAGE_VERSION,
                input_hash=input_digest or None,
                details=f"Read {request.path} through the '{source_name}' source.",
                performed_at=stamp,
            ),
            ProvenanceStep(
                stage="validated",
                tool="extreme_weather.ingestion.quality",
                version=PACKAGE_VERSION,
                output_hash=content_hash(dataset),
                configuration={"strict": request.strict},
                details=(
                    f"valid={report.valid}, {report.error_count} error(s), "
                    f"{report.warning_count} warning(s)"
                ),
                performed_at=stamp,
            ),
            ProvenanceStep(
                stage="normalized",
                tool="extreme_weather.ingestion.normalizer",
                version=PACKAGE_VERSION,
                output_hash=content_hash(normalization.dataset),
                configuration={
                    "variables": list(request.variables) or "settings default",
                    "latitude": list(request.latitude) if request.latitude else None,
                    "longitude": list(request.longitude) if request.longitude else None,
                },
                details="; ".join(normalization.coordinate_actions) or "no coordinate change",
                performed_at=stamp,
            ),
        )
        for step in chain:
            manifest = manifest.add_step(step)
        return manifest

