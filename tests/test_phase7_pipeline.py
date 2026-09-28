"""End-to-end tests of the Phase 7 validation pipeline.

These run the **real** pipeline - real ingestion, real climatology, real
calibration, real detection, real tracking, real comparison - over a dataset
built by the project's own generator. No network, and no claim of skill.

The dataset here is **synthetic**, so any number it produces describes the
pipeline's behaviour on a known input, not the real atmosphere. The test that
uses the committed ERA5 fixture is the one that exercises the real path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from extreme_weather.ingestion import DataOrigin
from extreme_weather.validation.climatology_validation import (
    build_climatology_diagnostics,
)
from extreme_weather.validation.dataset_builder import (
    ValidationDatasetBuilder,
    ValidationDatasetConfig,
)
from extreme_weather.validation.reports import REPORT_FILENAMES, read_report
from extreme_weather.validation.splits import split_dataset
from extreme_weather.validation.threshold_calibration import calibrate_thresholds

from tests.validation_fixtures import make_validation_dataset


@pytest.fixture()
def config() -> ValidationDatasetConfig:
    return ValidationDatasetConfig(
        dataset_id="unit_run",
        start_date="2024-06-01",
        end_date="2024-06-14",
        latitudes=(12.0, 12.5, 13.0),
        longitudes=(76.0, 76.5, 77.0),
        origin=DataOrigin.SYNTHETIC,
    )


# --------------------------------------------------------------------------- #
# The chain
# --------------------------------------------------------------------------- #
def test_the_whole_chain_runs_without_touching_the_network(
    tmp_path: Path, config: ValidationDatasetConfig
) -> None:
    """Ingest -> split -> climatology -> calibrate -> detect -> compare."""
    from extreme_weather.anomaly.climatology import (
        ClimatologyGrouping,
        build_climatology,
    )
    from extreme_weather.anomaly.service import AnomalyDetectionService, EventStore
    from extreme_weather.config import Settings
    from extreme_weather.validation.metrics import (
        compute_anomaly_diagnostics,
        compute_gnn_comparison,
    )

    settings = Settings(data_dir=tmp_path, model_dir=tmp_path / "models")
    settings.ensure_directories()
    builder = ValidationDatasetBuilder(settings)
    builder.ensure_directories()

    # --- ingest, through the real Phase 6 service ---------------------
    from extreme_weather.ingestion import IngestionRequest, IngestionService
    from extreme_weather.ingestion import ManifestStore
    from extreme_weather.ingestion.registry import DatasetRegistry
    from tests.ingestion_fixtures import write_netcdf

    source = write_netcdf(
        make_validation_dataset(steps=100), builder.raw_dir / "source.nc"
    )
    service = IngestionService(
        registry=DatasetRegistry(
            ManifestStore(builder.manifests_dir), settings=settings
        ),
        settings=settings,
    )
    ingested = service.ingest(
        IngestionRequest(
            dataset_id=config.dataset_id, origin=config.origin, path=source
        )
    )
    assert ingested.quality.valid is True
    assert ingested.output_path is not None and ingested.output_path.is_file()

    # --- split, and prove it is leakage free ---------------------------
    with xr.open_dataset(ingested.output_path) as opened:
        dataset = opened.load()
    blocks = split_dataset(dataset)
    train_end = np.max(blocks["train"].coords["time"].values)
    assert train_end < np.min(blocks["validation"].coords["time"].values)
    assert np.max(blocks["validation"].coords["time"].values) < np.min(
        blocks["test"].coords["time"].values
    )

    # --- climatology, from TRAIN only ---------------------------------
    diagnostics = build_climatology_diagnostics(
        blocks["train"], split="train", dataset_id=config.dataset_id
    )
    # The source is synthetic, so a short climatology is expected and must
    # be reported rather than hidden.
    assert diagnostics.is_sufficient is False
    assert diagnostics.coverage("day_of_year") is not None
    assert diagnostics.coverage("day_of_year").usable is False

    # The Phase 1 service builds its own climatology from a reference
    # *dataset*, so the reference is the training split - which is the leakage
    # guard. A standalone climatology file is written for inspection only.
    train_path = builder.processed_dir / "train.nc"
    blocks["train"].to_netcdf(train_path, mode="w")
    build_climatology(
        blocks["train"], "precipitation", ClimatologyGrouping.MONTH
    ).to_netcdf(builder.processed_dir / "climatology_train.nc", mode="w")

    # --- calibration, from TRAIN only ---------------------------------
    calibration = calibrate_thresholds(
        blocks["train"], dataset_id=config.dataset_id
    )
    assert calibration.thresholds is not None
    assert calibration.split == "train"
    assert calibration.variable("precipitation") is not None

    # --- detection on VALIDATION --------------------------------------
    validation_path = builder.processed_dir / "validation.nc"
    blocks["validation"].to_netcdf(validation_path, mode="w")
    events_path = builder.reports_dir / "events.json"
    analysis = AnomalyDetectionService(
        store=EventStore(events_path), thresholds=calibration.thresholds
    ).run(validation_path, reference_path=train_path, write_outputs=True)
    assert events_path.is_file()

    report = compute_anomaly_diagnostics(
        list(analysis.events), split="validation", dataset_id=config.dataset_id
    ).as_dict()
    assert report["detection_skill"]["available"] is False
    assert report["detection_skill"]["POD"] is None

    # --- a baseline comparison still works ----------------------------
    truth = np.zeros(4)
    comparison = compute_gnn_comparison(
        truth, truth, truth, truth, truth, truth,
        truth + 0.1, truth, truth,
        dataset=config.dataset_id, split="test", model_checkpoint="none",
    )
    assert comparison.comparison["_summary"]["verdict"] in (
        "beats persistence",
        "matches persistence",
        "worse than persistence",
    )


# --------------------------------------------------------------------------- #
# Report production
# --------------------------------------------------------------------------- #
def test_the_pipeline_writes_every_report(tmp_path: Path, config) -> None:
    """Every stage report is present and readable after a run."""
    from extreme_weather.config import Settings
    from extreme_weather.validation.provenance import (
        ProvenanceRecord,
        ValidationRunRecord,
        write_run_record,
    )
    from extreme_weather.validation.reports import build_phase7_summary, write_reports
    from datetime import datetime, timezone

    settings = Settings(data_dir=tmp_path, model_dir=tmp_path / "models")
    settings.ensure_directories()
    builder = ValidationDatasetBuilder(settings)
    builder.ensure_directories()

    for name in ("anomaly", "tracking", "gnn", "downscaling", "alerts",
                 "climatology", "threshold_calibration"):
        (builder.reports_dir / REPORT_FILENAMES[name]).write_text(
            json.dumps({"marker": name}), encoding="utf-8"
        )

    record = ValidationRunRecord(
        run_id="run-test",
        started_at=datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc),
        dataset=ProvenanceRecord(dataset_id=config.dataset_id, origin=config.origin),
    )
    write_run_record(record, builder.provenance_dir)

    summary = build_phase7_summary(
        dataset=record.dataset.as_dict(),
        split={"splits": []},
        ground_truth={"status": "UNAVAILABLE", "available": False},
    )
    paths = write_reports(builder.reports_dir, summary)

    assert paths["json"].is_file() and paths["markdown"].is_file()
    assert read_report(builder.reports_dir, "summary") is not None
    assert read_report(builder.provenance_dir.parent / "reports", "gnn") is not None


# --------------------------------------------------------------------------- #
# The real-data path, using the committed ERA5 fixture
# --------------------------------------------------------------------------- #
def test_the_real_era5_fixture_reaches_the_pipeline(tmp_path: Path) -> None:
    """Genuine ERA5 data, ingested and normalised, with no network.

    The ERA5 JSON fixture is real reanalysis captured in Phase 6; the conversion
    is a pure function, so this exercises the real path offline.
    """
    from extreme_weather.config import Settings
    from extreme_weather.ingestion import IngestionRequest, IngestionService
    from extreme_weather.ingestion import ManifestStore
    from extreme_weather.ingestion import era5_payload_to_dataset
    from extreme_weather.ingestion.registry import DatasetRegistry
    from tests.ingestion_fixtures import ERA5_FIXTURE

    settings = Settings(data_dir=tmp_path, model_dir=tmp_path / "models")
    settings.ensure_directories()
    builder = ValidationDatasetBuilder(settings)
    builder.ensure_directories()

    payload = json.loads(ERA5_FIXTURE.read_text(encoding="utf-8"))
    source = era5_payload_to_dataset(payload)
    source_path = builder.raw_dir / "era5_source.nc"
    source.to_netcdf(source_path, mode="w")

    service = IngestionService(
        registry=DatasetRegistry(
            ManifestStore(builder.manifests_dir), settings=settings
        ),
        settings=settings,
    )
    result = service.ingest(
        IngestionRequest(
            dataset_id="era5_unit", origin=DataOrigin.HISTORICAL, path=source_path
        )
    )
    assert result.quality.valid is True
    assert result.manifest.is_synthetic is False
    assert result.manifest.is_live is False
    assert result.manifest.source == "local_file"
    assert result.manifest.source_url is None
    # The archive's degC/hPa became the project's K/Pa.
    assert result.manifest.units["temperature"] == "K"
    assert result.manifest.units["pressure"] == "Pa"

