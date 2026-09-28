"""Tests for dataset manifests, hashes and provenance (SIH26078, Phase 6).

A manifest is the only record of where a file came from and what was done to
it, so the properties asserted here are the ones that make it trustworthy:

* the provenance fields are **never invented** - a local file has no URL, and
  the constructor refuses the combination;
* hashes are real and reproducible - the file hash over bytes, the content hash
  over array values, with the documented difference between them;
* ``is_live`` is always False, and the three origins stay distinguishable.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from extreme_weather.ingestion import (
    SOURCE_LOCAL_FILE,
    DatasetManifest,
    ManifestStore,
    ProvenanceStep,
    build_manifest,
    content_hash,
    file_hash,
)
from extreme_weather.ingestion.base import DataOrigin
from extreme_weather.ingestion.exceptions import IngestionError

from tests.ingestion_fixtures import make_dataset, write_netcdf

UTC = timezone.utc
NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)


def make_manifest(**overrides: object) -> DatasetManifest:
    """Build a minimal valid manifest; every field can be overridden."""
    payload: dict = {
        "dataset_id": "demo",
        "source": SOURCE_LOCAL_FILE,
        "retrieved_at": NOW,
        "file_name": "demo.nc",
        "file_hash": "0" * 64,
        "format": "netcdf",
        "origin": DataOrigin.HISTORICAL,
    }
    payload.update(overrides)
    return DatasetManifest(**payload)


@pytest.fixture()
def store(tmp_path: Path) -> ManifestStore:
    """A manifest store in a temporary folder."""
    return ManifestStore(tmp_path / "manifests")


# --------------------------------------------------------------------------- #
# Hashing
# --------------------------------------------------------------------------- #
def test_file_hash_is_a_sha256_hex_digest(contract_file: Path) -> None:
    digest = file_hash(contract_file)
    assert len(digest) == 64
    assert int(digest, 16) >= 0


def test_file_hash_is_stable_across_calls(contract_file: Path) -> None:
    assert file_hash(contract_file) == file_hash(contract_file)


def test_file_hash_changes_when_the_bytes_change(tmp_path: Path) -> None:
    first = write_netcdf(make_dataset(), tmp_path / "a.nc")
    second = write_netcdf(
        make_dataset(variables={"precipitation": (np.full((4, 3, 3), 9.0), "mm h-1")}),
        tmp_path / "b.nc",
    )
    assert file_hash(first) != file_hash(second)


def test_file_hash_reads_large_files_in_chunks(tmp_path: Path) -> None:
    """Chunked hashing is what keeps a multi-GB file out of RAM."""
    big = tmp_path / "big.bin"
    big.write_bytes(b"x" * (3 * 1024 * 1024))
    assert len(file_hash(big)) == 64


def test_file_hash_on_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(IngestionError, match="Could not hash"):
        file_hash(tmp_path / "nope.bin")


def test_content_hash_is_stable_for_equal_data() -> None:
    first = make_dataset()
    second = make_dataset()
    assert content_hash(first) == content_hash(second)


def test_content_hash_survives_a_rewrite(tmp_path: Path) -> None:
    """Compression/attribute order must not change the data digest."""
    dataset = make_dataset()
    original = content_hash(dataset)
    path = write_netcdf(dataset, tmp_path / "a.nc")
    import xarray as xr

    with xr.open_dataset(path) as reopened:
        assert content_hash(reopened.load()) == original


def test_content_hash_changes_when_the_values_change() -> None:
    base = make_dataset()
    changed = make_dataset(
        variables={"precipitation": (np.full((4, 3, 3), 9.0), "mm h-1")}
    )
    assert content_hash(base) != content_hash(changed)


def test_content_hash_is_sensitive_to_variable_names() -> None:
    first = make_dataset(variables={"precipitation": (np.full((4, 3, 3), 1.0), "mm h-1")})
    second = make_dataset(variables={"temperature": (np.full((4, 3, 3), 1.0), "K")})
    assert content_hash(first) != content_hash(second)


# --------------------------------------------------------------------------- #
# Never invent provenance
# --------------------------------------------------------------------------- #
def test_a_local_file_manifest_must_not_carry_a_url() -> None:
    """The single most important rule in this module."""
    with pytest.raises(ValueError, match="must not carry a source_url"):
        make_manifest(source=SOURCE_LOCAL_FILE, source_url="https://example.invalid/x")


def test_a_named_provider_may_carry_its_url() -> None:
    manifest = make_manifest(
        source="era5", source_url="https://archive-api.open-meteo.com/v1/archive"
    )
    assert manifest.source_url is not None


def test_a_manifest_without_a_url_normalises_it_to_none() -> None:
    assert make_manifest(source="era5").source_url is None


def test_is_live_is_always_false() -> None:
    """A file fetched now is still a historical artefact."""
    for origin in DataOrigin.ALL:
        assert make_manifest(origin=origin).is_live is False


def test_is_synthetic_is_true_only_for_synthetic() -> None:
    assert make_manifest(origin=DataOrigin.SYNTHETIC).is_synthetic is True
    assert make_manifest(origin=DataOrigin.HISTORICAL).is_synthetic is False
    assert make_manifest(origin=DataOrigin.EXTERNAL).is_synthetic is False


def test_the_three_origins_get_three_distinct_badges() -> None:
    labels = {origin: make_manifest(origin=origin).status_label for origin in DataOrigin.ALL}
    assert labels[DataOrigin.SYNTHETIC] == "SYNTHETIC"
    assert labels[DataOrigin.HISTORICAL] == "HISTORICAL"
    assert "EXTERNAL" in labels[DataOrigin.EXTERNAL]
    assert len(set(labels.values())) == 3


# --------------------------------------------------------------------------- #
# Serialisation
# --------------------------------------------------------------------------- #
def test_a_manifest_round_trips_through_json() -> None:
    original = make_manifest(
        variables=("precipitation", "temperature"),
        units={"precipitation": "mm h-1"},
        time_start="2024-06-01T00:00:00",
        time_end="2024-06-01T03:00:00",
    )
    restored = DatasetManifest.from_dict(json.loads(json.dumps(original.as_dict())))
    assert restored.dataset_id == original.dataset_id
    assert restored.variables == original.variables
    assert restored.units == original.units
    assert restored.time_start == original.time_start


def test_the_serialised_manifest_has_every_provenance_field() -> None:
    payload = make_manifest().as_dict()
    for key in (
        "dataset_id",
        "source",
        "source_url",
        "retrieved_at",
        "file_name",
        "file_hash",
        "format",
        "origin",
        "is_synthetic",
        "is_live",
        "variables",
        "dimensions",
        "resolution_deg",
        "coordinate_system",
        "units",
        "processing_steps",
        "quality_summary",
        "license",
    ):
        assert key in payload, f"{key} is missing from the manifest"
    assert payload["file_hash_algorithm"] == "sha256"


# --------------------------------------------------------------------------- #
# Provenance steps
# --------------------------------------------------------------------------- #
def test_provenance_steps_accumulate_and_serialise() -> None:
    first = ProvenanceStep(
        stage="ingested", tool="extreme_weather.ingestion.netcdf", performed_at=NOW
    )
    second = ProvenanceStep(
        stage="normalized",
        tool="extreme_weather.ingestion.normalizer",
        input_hash="a" * 64,
        output_hash="b" * 64,
        performed_at=NOW + timedelta(minutes=1),
    )
    manifest = make_manifest().add_step(first).add_step(second)
    assert [step.stage for step in manifest.processing_steps] == ["ingested", "normalized"]

    payload = manifest.as_dict()
    assert payload["processing_steps"][1]["input_hash"] == "a" * 64

    restored = DatasetManifest.from_dict(payload)
    assert len(restored.processing_steps) == 2
    assert restored.processing_steps[1].output_hash == "b" * 64



# --------------------------------------------------------------------------- #
# build_manifest
# --------------------------------------------------------------------------- #
def test_build_manifest_reads_shape_from_the_dataset(contract_file: Path) -> None:
    import xarray as xr

    with xr.open_dataset(contract_file) as opened:
        manifest = build_manifest(
            opened.load(),
            dataset_id="demo",
            source_path=contract_file,
            source=SOURCE_LOCAL_FILE,
            origin=DataOrigin.HISTORICAL,
        )
    assert manifest.dimensions["time"] == 4
    assert manifest.dimensions["latitude"] == 3
    assert "temperature" in manifest.variables
    assert manifest.resolution_deg == pytest.approx(0.5)
    assert manifest.resolution_km is not None
    assert manifest.time_start == "2024-06-01T00:00:00"
    assert manifest.time_step_seconds == pytest.approx(3600.0)
    assert manifest.units["temperature"] == "K"
    assert len(manifest.file_hash) == 64
    assert manifest.size_bytes == contract_file.stat().st_size


def test_build_manifest_records_missing_variables(contract_file: Path) -> None:
    import xarray as xr

    with xr.open_dataset(contract_file) as opened:
        manifest = build_manifest(
            opened.load(),
            dataset_id="demo",
            source_path=contract_file,
            source=SOURCE_LOCAL_FILE,
            variables_missing=("u_wind", "v_wind"),
        )
    assert manifest.variables_missing == ("u_wind", "v_wind")


# --------------------------------------------------------------------------- #
# ManifestStore
# --------------------------------------------------------------------------- #
def test_the_store_saves_and_loads_a_manifest(store: ManifestStore) -> None:
    store.save(make_manifest(dataset_id="alpha"))
    loaded = store.load("alpha")
    assert loaded is not None
    assert loaded.dataset_id == "alpha"


def test_a_missing_manifest_loads_as_none(store: ManifestStore) -> None:
    assert store.load("never-registered") is None


def test_the_store_lists_and_removes(store: ManifestStore) -> None:
    store.save(make_manifest(dataset_id="alpha"))
    store.save(make_manifest(dataset_id="beta"))
    assert store.list_ids() == ["alpha", "beta"]
    assert len(store.list()) == 2
    assert store.remove("alpha") is True
    assert store.remove("alpha") is False
    assert store.list_ids() == ["beta"]


def test_one_manifest_per_file_so_a_corrupt_one_is_isolated(store: ManifestStore) -> None:
    store.save(make_manifest(dataset_id="alpha"))
    store.save(make_manifest(dataset_id="beta"))
    store.path_for("alpha").write_text("{broken", encoding="utf-8")

    # Reading the broken one raises ...
    with pytest.raises(IngestionError, match="Could not read manifest"):
        store.load("alpha")
    # ... but the listing still yields the good one.
    assert [manifest.dataset_id for manifest in store.list()] == ["beta"]


def test_registering_the_same_id_twice_replaces(store: ManifestStore) -> None:
    store.save(make_manifest(dataset_id="alpha", file_hash="a" * 64))
    store.save(make_manifest(dataset_id="alpha", file_hash="b" * 64))
    assert store.list_ids() == ["alpha"]
    loaded = store.load("alpha")
    assert loaded is not None and loaded.file_hash == "b" * 64


def test_the_store_creates_its_directory(tmp_path: Path) -> None:
    nested = ManifestStore(tmp_path / "a" / "b" / "manifests")
    nested.save(make_manifest(dataset_id="alpha"))
    assert nested.path_for("alpha").is_file()


def test_the_store_finds_by_time_overlap(store: ManifestStore) -> None:
    store.save(
        make_manifest(
            dataset_id="june",
            time_start="2024-06-01T00:00:00",
            time_end="2024-06-03T00:00:00",
        )
    )
    store.save(
        make_manifest(
            dataset_id="january",
            time_start="2024-01-01T00:00:00",
            time_end="2024-01-03T00:00:00",
        )
    )
    found = store.find_by_time(start="2024-06-02T00:00:00", end="2024-06-05T00:00:00")
    assert [manifest.dataset_id for manifest in found] == ["june"]

