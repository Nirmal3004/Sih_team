"""Tests for the Phase 7 dataset builder and provenance.

The builder must be **bounded**. An unbounded request is how a research script
fills a disk, so the configuration rejects one outright rather than warning.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from extreme_weather.config import Settings
from extreme_weather.ingestion import DataOrigin
from extreme_weather.validation.dataset_builder import (
    MAX_REQUEST_DAYS,
    ValidationDatasetBuilder,
    ValidationDatasetConfig,
    ValidationDatasetError,
    validation_dir,
)
from extreme_weather.validation.provenance import (
    ProvenanceRecord,
    ValidationRunRecord,
    environment_facts,
    read_run_record,
    write_run_record,
)

UTC = timezone.utc


def make_config(**overrides) -> ValidationDatasetConfig:
    payload = {
        "dataset_id": "demo",
        "start_date": "2024-06-01",
        "end_date": "2024-06-14",
        "latitudes": (12.0, 12.5, 13.0),
        "longitudes": (76.0, 76.5, 77.0),
    }
    payload.update(overrides)
    return ValidationDatasetConfig(**payload)


# --------------------------------------------------------------------------- #
# Bounding
# --------------------------------------------------------------------------- #
def test_a_bounded_request_is_accepted() -> None:
    config = make_config()
    assert config.day_count() == 14
    assert config.origin == DataOrigin.HISTORICAL


def test_an_unbounded_request_is_rejected() -> None:
    """No box means a global request, which is gigabytes. Refuse it."""
    with pytest.raises(ValidationDatasetError, match="bounding box"):
        make_config(latitudes=(), longitudes=())


def test_a_request_over_the_day_ceiling_is_rejected() -> None:
    with pytest.raises(ValidationDatasetError, match="ceiling"):
        make_config(start_date="2024-01-01", end_date="2024-12-31")


def test_the_day_ceiling_is_configurable() -> None:
    """Raising max_days admits a longer period; lowering it rejects one."""
    config = make_config(start_date="2024-01-01", end_date="2024-01-08", max_days=30)
    assert config.day_count() == 8
    with pytest.raises(ValidationDatasetError, match="ceiling"):
        make_config(start_date="2024-01-01", end_date="2024-01-08", max_days=5)


def test_an_out_of_range_latitude_is_rejected() -> None:
    with pytest.raises(ValidationDatasetError, match="outside"):
        make_config(latitudes=(95.0,))


def test_an_out_of_range_longitude_is_rejected() -> None:
    with pytest.raises(ValidationDatasetError, match="outside"):
        make_config(longitudes=(200.0,))


def test_a_reversed_date_range_is_rejected() -> None:
    with pytest.raises(ValidationDatasetError, match="after end_date"):
        make_config(start_date="2024-06-14", end_date="2024-06-01")


def test_an_empty_dataset_id_is_rejected() -> None:
    with pytest.raises(ValidationDatasetError, match="dataset_id"):
        make_config(dataset_id="  ")


def test_an_invalid_origin_is_rejected() -> None:
    with pytest.raises(ValueError):
        make_config(origin="LIVE")


def test_the_config_serialises_its_request() -> None:
    payload = make_config().as_dict()
    assert payload["days"] == 14
    assert payload["latitudes"] == [12.0, 12.5, 13.0]
    assert payload["origin"] == "HISTORICAL"


# --------------------------------------------------------------------------- #
# Workspace layout
# --------------------------------------------------------------------------- #
def test_the_workspace_lives_under_the_data_directory(
    validation_settings: Settings,
) -> None:
    assert validation_dir(validation_settings) == validation_settings.data_dir / "validation"


def test_the_builder_creates_every_subdirectory(validation_settings: Settings) -> None:
    builder = ValidationDatasetBuilder(validation_settings)
    builder.ensure_directories()
    for directory in (
        builder.raw_dir,
        builder.processed_dir,
        builder.manifests_dir,
        builder.splits_dir,
        builder.reports_dir,
        builder.provenance_dir,
    ):
        assert directory.is_dir(), directory


def test_the_workspace_is_separate_from_the_synthetic_data(
    validation_settings: Settings,
) -> None:
    """Validation data must never land in raw/synthetic or raw/real."""
    builder = ValidationDatasetBuilder(validation_settings)
    assert builder.raw_dir != validation_settings.synthetic_raw_dir
    assert builder.raw_dir != validation_settings.real_raw_dir


def test_the_builder_exposes_its_settings(validation_settings: Settings) -> None:
    assert ValidationDatasetBuilder(validation_settings).settings is validation_settings


# --------------------------------------------------------------------------- #
# Building from an existing file (no network)
# --------------------------------------------------------------------------- #
def test_a_missing_source_file_is_reported(validation_settings: Settings) -> None:
    builder = ValidationDatasetBuilder(validation_settings)
    builder.ensure_directories()
    with pytest.raises(ValidationDatasetError, match="does not exist"):
        builder.build(make_config(), source_path=builder.raw_dir / "absent.nc")


def test_build_without_fetch_or_a_path_is_rejected(validation_settings: Settings) -> None:
    builder = ValidationDatasetBuilder(validation_settings)
    builder.ensure_directories()
    with pytest.raises(ValidationDatasetError, match="fetch=True"):
        builder.build(make_config(), fetch=False)


def test_provenance_records_a_dataset() -> None:
    record = ProvenanceRecord(
        dataset_id="demo",
        source="local_file",
        file_hash="a" * 64,
        time_start="2024-06-01T00:00:00",
        time_end="2024-06-14T00:00:00",
        variables=("precipitation",),
        origin=DataOrigin.HISTORICAL,
    )
    payload = record.as_dict()
    assert payload["dataset_id"] == "demo"
    assert payload["is_synthetic"] is False
    assert payload["is_live"] is False
    assert payload["source_url"] is None


def test_provenance_round_trips(tmp_path: Path) -> None:
    record = ValidationRunRecord(
        run_id="run-1",
        started_at=datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
        dataset=ProvenanceRecord(dataset_id="demo"),
        environment=environment_facts(),
    )
    record.record_stage("dataset", "ok", days=14)
    record.record_stage("split", "ok")

    path = write_run_record(record, tmp_path)
    restored = read_run_record(path)
    assert restored is not None
    assert restored.run_id == "run-1"
    assert restored.stages == {"dataset": "ok", "split": "ok"}
    assert restored.complete is True


def test_an_incomplete_run_is_not_marked_complete(tmp_path: Path) -> None:
    record = ValidationRunRecord(
        run_id="run-2",
        started_at=datetime(2024, 6, 1, 12, 0, tzinfo=UTC),
        dataset=ProvenanceRecord(dataset_id="demo"),
    )
    record.record_stage("dataset", "ok")
    record.record_stage("gnn", "skipped")
    assert record.complete is False
    restored = read_run_record(write_run_record(record, tmp_path))
    assert restored is not None and restored.complete is False


def test_reading_a_missing_record_returns_none(tmp_path: Path) -> None:
    assert read_run_record(tmp_path / "nothing.json") is None


def test_the_environment_records_the_library_versions() -> None:
    facts = environment_facts()
    assert "python" in facts
    assert "numpy" in facts
    assert "xarray" in facts


def test_a_run_record_never_claims_to_be_live() -> None:
    record = ProvenanceRecord(dataset_id="demo", origin=DataOrigin.HISTORICAL)
    assert record.as_dict()["is_live"] is False

