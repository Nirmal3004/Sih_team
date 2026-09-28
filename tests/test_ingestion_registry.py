"""Tests for the ingested-dataset registry (SIH26078, Phase 6).

The registry is the index the API and the dashboard read, so what matters is
that it answers questions correctly and - critically - that it keeps **synthetic
and real data apart**. Every filter here exists to make mixing visible rather
than silent.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from extreme_weather.config import Settings
from extreme_weather.ingestion import (
    DataOrigin,
    DatasetManifest,
    DatasetRegistry,
    ManifestStore,
    SOURCE_LOCAL_FILE,
)
from extreme_weather.ingestion.exceptions import SourceNotFoundError

from tests.ingestion_fixtures import make_dataset, write_netcdf

UTC = timezone.utc
NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def make_manifest(dataset_id: str = "demo", **overrides: object) -> DatasetManifest:
    """A minimal manifest for registry tests."""
    payload: dict = {
        "dataset_id": dataset_id,
        "source": SOURCE_LOCAL_FILE,
        "retrieved_at": NOW,
        "file_name": f"{dataset_id}.nc",
        "file_hash": "0" * 64,
        "format": "netcdf",
        "origin": DataOrigin.HISTORICAL,
        "variables": ("precipitation", "temperature"),
        "time_start": "2024-06-01T00:00:00",
        "time_end": "2024-06-01T03:00:00",
    }
    payload.update(overrides)
    return DatasetManifest(**payload)


@pytest.fixture()
def registry(settings: Settings) -> DatasetRegistry:
    """A registry backed by the temporary manifest directory."""
    return DatasetRegistry(ManifestStore(settings.manifests_dir), settings=settings)


def register_ingested(
    settings: Settings,
    registry: DatasetRegistry,
    dataset_id: str = "demo",
    origin: str = DataOrigin.HISTORICAL,
):
    """Write a real dataset file and register its manifest."""
    base = (
        settings.synthetic_raw_dir
        if origin == DataOrigin.SYNTHETIC
        else settings.real_raw_dir
    )
    write_netcdf(make_dataset(), base / f"{dataset_id}.nc")
    return registry.register(make_manifest(dataset_id, origin=origin))


# --------------------------------------------------------------------------- #
# Basic operations
# --------------------------------------------------------------------------- #
def test_a_new_registry_is_empty(registry: DatasetRegistry) -> None:
    assert registry.list() == []
    assert registry.ids() == []


def test_register_then_get(registry: DatasetRegistry) -> None:
    registry.register(make_manifest("alpha"))
    entry = registry.get("alpha")
    assert entry.dataset_id == "alpha"
    assert entry.origin == DataOrigin.HISTORICAL
    assert registry.ids() == ["alpha"]


def test_registering_the_same_id_twice_replaces(registry: DatasetRegistry) -> None:
    registry.register(make_manifest("alpha", file_hash="a" * 64))
    registry.register(make_manifest("alpha", file_hash="b" * 64))
    assert registry.ids() == ["alpha"]
    assert registry.get("alpha").manifest.file_hash == "b" * 64


def test_get_unknown_raises_and_lists_what_exists(registry: DatasetRegistry) -> None:
    registry.register(make_manifest("alpha"))
    with pytest.raises(SourceNotFoundError) as info:
        registry.get("beta")
    assert "not registered" in str(info.value)
    assert "alpha" in str(info.value)


def test_find_returns_none_instead_of_raising(registry: DatasetRegistry) -> None:
    assert registry.find("nope") is None


def test_remove_reports_whether_it_removed_anything(
    registry: DatasetRegistry, settings: Settings
) -> None:
    register_ingested(settings, registry, "alpha")
    data_file = settings.real_raw_dir / "alpha.nc"
    assert registry.remove("alpha") is True
    assert registry.remove("alpha") is False
    # The data file survives: removing a registry entry is not deleting data.
    assert data_file.is_file()


# --------------------------------------------------------------------------- #
# Synthetic / real separation - the point of the registry
# --------------------------------------------------------------------------- #
def test_synthetic_and_real_datasets_are_listed_separately(
    registry: DatasetRegistry, settings: Settings
) -> None:
    register_ingested(settings, registry, "fake", origin=DataOrigin.SYNTHETIC)
    register_ingested(settings, registry, "real", origin=DataOrigin.HISTORICAL)
    register_ingested(settings, registry, "external", origin=DataOrigin.EXTERNAL)

    assert len(registry.list()) == 3
    assert [e.dataset_id for e in registry.list(origin=DataOrigin.SYNTHETIC)] == ["fake"]
    assert [e.dataset_id for e in registry.list(origin=DataOrigin.HISTORICAL)] == ["real"]
    assert [e.dataset_id for e in registry.list(origin=DataOrigin.EXTERNAL)] == ["external"]


def test_an_unknown_origin_filter_is_rejected(registry: DatasetRegistry) -> None:
    with pytest.raises(ValueError, match="Unknown data origin"):
        registry.list(origin="LIVE")


def test_the_synthetic_flag_follows_the_origin(
    registry: DatasetRegistry, settings: Settings
) -> None:
    entry = register_ingested(settings, registry, "fake", origin=DataOrigin.SYNTHETIC)
    assert entry.is_synthetic is True
    real = register_ingested(settings, registry, "real", origin=DataOrigin.HISTORICAL)
    assert real.is_synthetic is False


def test_a_synthetic_file_is_written_to_the_synthetic_directory(
    registry: DatasetRegistry, settings: Settings
) -> None:
    """Real and synthetic data must never share a directory."""
    register_ingested(settings, registry, "fake", origin=DataOrigin.SYNTHETIC)
    assert (settings.synthetic_raw_dir / "fake.nc").is_file()
    assert not (settings.real_raw_dir / "fake.nc").exists()


def test_a_real_file_is_written_to_the_real_directory(
    registry: DatasetRegistry, settings: Settings
) -> None:
    register_ingested(settings, registry, "observed", origin=DataOrigin.HISTORICAL)
    assert (settings.real_raw_dir / "observed.nc").is_file()
    assert not (settings.synthetic_raw_dir / "observed.nc").exists()


def test_filter_by_source(registry: DatasetRegistry) -> None:
    registry.register(make_manifest("local_one", source="local_file"))
    registry.register(make_manifest("era5_one", source="era5"))
    assert [e.dataset_id for e in registry.list(source="era5")] == ["era5_one"]


# --------------------------------------------------------------------------- #
# Time and variable queries
# --------------------------------------------------------------------------- #
def test_find_by_time_returns_overlapping_datasets(registry: DatasetRegistry) -> None:
    registry.register(
        make_manifest("june", time_start="2024-06-01T00:00:00", time_end="2024-06-03T00:00:00")
    )
    registry.register(
        make_manifest(
            "january", time_start="2024-01-01T00:00:00", time_end="2024-01-03T00:00:00"
        )
    )
    found = registry.find_by_time(start="2024-06-02T00:00:00", end="2024-06-05T00:00:00")
    assert [entry.dataset_id for entry in found] == ["june"]


def test_find_by_time_accepts_datetimes(registry: DatasetRegistry) -> None:
    registry.register(
        make_manifest("june", time_start="2024-06-01T00:00:00", time_end="2024-06-03T00:00:00")
    )
    found = registry.find_by_time(
        start=datetime(2024, 6, 2, tzinfo=UTC), end=datetime(2024, 6, 5, tzinfo=UTC)
    )
    assert [entry.dataset_id for entry in found] == ["june"]


def test_find_by_time_rejects_a_bad_timestamp(registry: DatasetRegistry) -> None:
    with pytest.raises(ValueError, match="not a valid ISO 8601"):
        registry.find_by_time(start="yesterday")


def test_find_by_variable_is_case_insensitive(registry: DatasetRegistry) -> None:
    registry.register(make_manifest("with_precip", variables=("precipitation",)))
    registry.register(make_manifest("with_temperature", variables=("temperature",)))
    assert [e.dataset_id for e in registry.find_by_variable("PRECIPITATION")] == [
        "with_precip"
    ]
    assert registry.find_by_variable("hail") == []



# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_validate_rechecks_a_registered_dataset(
    registry: DatasetRegistry, settings: Settings
) -> None:
    register_ingested(settings, registry, "alpha")
    report = registry.validate("alpha")
    assert report.valid is True
    # The report is remembered, so a later get() exposes it without re-reading.
    assert registry.get("alpha").quality_valid is True


def test_validate_on_a_missing_file_raises(registry: DatasetRegistry) -> None:
    registry.register(make_manifest("ghost"))
    with pytest.raises(SourceNotFoundError, match="not found"):
        registry.validate("ghost")


def test_validate_all_turns_a_missing_file_into_an_invalid_report(
    registry: DatasetRegistry, settings: Settings
) -> None:
    register_ingested(settings, registry, "present")
    registry.register(make_manifest("ghost"))
    reports = registry.validate_all()
    assert reports["present"].valid is True
    assert reports["ghost"].valid is False
    assert "not found" in reports["ghost"].errors[0]


def test_valid_only_filters_on_the_last_report(
    registry: DatasetRegistry, settings: Settings
) -> None:
    register_ingested(settings, registry, "good")
    registry.register(make_manifest("never_checked"))
    # Nothing has been validated yet, so nothing is excluded ...
    assert len(registry.list(valid_only=True)) == 2
    registry.validate_all()
    # ... after the sweep only the dataset with a file survives.
    assert [e.dataset_id for e in registry.list(valid_only=True)] == ["good"]


def test_the_entry_serialises_its_quality(
    registry: DatasetRegistry, settings: Settings
) -> None:
    register_ingested(settings, registry, "alpha")
    registry.validate("alpha")
    payload = registry.get("alpha").as_dict()
    assert payload["is_synthetic"] is False
    assert payload["quality_valid"] is True
    assert payload["quality"]["valid"] is True


def test_an_entry_without_a_report_reports_unknown_quality(
    registry: DatasetRegistry,
) -> None:
    registry.register(make_manifest("unchecked"))
    entry = registry.get("unchecked")
    assert entry.quality is None
    assert entry.quality_valid is None

