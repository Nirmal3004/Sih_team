"""End-to-end tests of the real / historical data path (SIH26078, Phase 6).

This module runs the **whole** Phase 6 -> Phase 5 chain on genuine ERA5
reanalysis data, with **no network access**: the archive payload is a committed
fixture and the conversion from it is a pure function.

```text
ERA5 fixture -> xarray -> ingest -> normalise -> manifest -> register
             -> Phase 0 preprocessing -> Phase 1 events -> Phase 2 tracks
             -> Phase 5 alerts
```

Every stage is the *real* project code, so a pass here means the real pipeline
consumes real data unchanged. What is deliberately **not** asserted is anything
about scientific skill: there is no ground truth in this repository, so no test
claims a detection was correct, and none claims a score is accurate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from extreme_weather.alerts.service import AlertService
from extreme_weather.alerts.store import AlertStore
from extreme_weather.anomaly.service import AnomalyDetectionService, EventStore
from extreme_weather.config import Settings
from extreme_weather.ingestion import (
    DataOrigin,
    IngestionRequest,
    IngestionService,
    content_hash,
    era5_payload_to_dataset,
    normalize_dataset,
)
from extreme_weather.tracking.service import TrackingService
from extreme_weather.tracking.store import TrackStore


# --------------------------------------------------------------------------- #
# The fixture is genuine, and genuinely not synthetic
# --------------------------------------------------------------------------- #
def test_the_fixture_is_real_era5_not_synthetic(era5_dataset: xr.Dataset) -> None:
    assert era5_dataset.attrs["is_synthetic"] == "false"
    assert era5_dataset.attrs["dataset_type"] == "historical_reanalysis"
    assert "ERA5" in era5_dataset.attrs["provider_attribution"]
    assert "Copernicus" in era5_dataset.attrs["license"]


def test_the_fixture_carries_the_archives_own_units(era5_dataset: xr.Dataset) -> None:
    """The point of using real data: the units are not the project's units."""
    assert era5_dataset["temperature_2m"].attrs["units"] == "degC"
    assert era5_dataset["surface_pressure"].attrs["units"] == "hPa"
    assert era5_dataset["relative_humidity_2m"].attrs["units"] == "%"


def test_the_conversion_is_deterministic(era5_payload) -> None:
    first = era5_payload_to_dataset(era5_payload)
    second = era5_payload_to_dataset(era5_payload)
    assert content_hash(first) == content_hash(second)
    assert dict(first.sizes) == dict(second.sizes)


def test_the_conversion_rejects_a_malformed_payload() -> None:
    from extreme_weather.ingestion.exceptions import IngestionError

    with pytest.raises(IngestionError):
        era5_payload_to_dataset("not a payload")
    with pytest.raises(IngestionError):
        era5_payload_to_dataset([{"latitude": 1.0}])  # no hourly block


# --------------------------------------------------------------------------- #
# Normalisation of real data
# --------------------------------------------------------------------------- #
def test_era5_normalises_into_the_project_contract(era5_dataset: xr.Dataset) -> None:
    result = normalize_dataset(era5_dataset, drop_unknown=True)
    assert "temperature" in result.dataset.data_vars
    assert result.dataset["temperature"].attrs["units"] == "K"
    assert result.dataset["pressure"].attrs["units"] == "Pa"
    # 27 degC .. 35 degC in the tropics: a plausible Kelvin range for ERA5.
    kelvin = float(result.dataset["temperature"].mean())
    assert 280.0 < kelvin < 320.0
    # hPa -> Pa puts surface pressure in the 90-105 kPa band.
    pascals = float(result.dataset["pressure"].mean())
    assert 85_000.0 < pascals < 110_000.0


def test_era5_reports_the_units_it_converted(era5_dataset: xr.Dataset) -> None:
    conversions = normalize_dataset(era5_dataset).conversion_map()
    assert conversions["temperature"]["original_units"] == "degC"
    assert conversions["pressure"]["scale"] == 100.0
    assert conversions["relative_humidity"]["conversion_method"] == "identity"


def test_era5_has_no_wind_components_and_that_is_reported(
    era5_dataset: xr.Dataset,
) -> None:
    """The archive gives speed + direction, not u/v: a missing variable."""
    result = normalize_dataset(era5_dataset, drop_unknown=True)
    assert "u_wind" not in result.dataset.data_vars
    assert "v_wind" not in result.dataset.data_vars
    assert "u_wind" in result.missing_variables


# --------------------------------------------------------------------------- #
# The full chain, from a real payload to real alerts
# --------------------------------------------------------------------------- #
@pytest.fixture()
def ingested_era5(era5_dataset: xr.Dataset, settings: Settings) -> Path:
    """Write the real payload to disk the way an archive download would."""
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    path = settings.cache_dir / "era5_real.nc"
    era5_dataset.to_netcdf(path, mode="w")
    return path


def test_real_data_ingests_and_registers(
    ingest_svc: IngestionService, ingested_era5: Path, settings: Settings
) -> None:
    result = ingest_svc.ingest(
        IngestionRequest(
            dataset_id="era5_real",
            origin=DataOrigin.HISTORICAL,
            path=ingested_era5,
        )
    )
    assert result.quality.valid is True
    assert result.manifest.is_synthetic is False
    assert result.manifest.is_live is False
    assert result.manifest.source == "local_file"
    assert "era5_real" in ingest_svc.registry.ids()
    # Real data lands under raw/real, never raw/synthetic.
    assert result.output_path == settings.real_raw_dir / "era5_real.nc"
    assert result.output_path.is_file()


def test_real_data_derives_wind_components_from_polar_wind(
    ingest_svc: IngestionService, ingested_era5: Path
) -> None:
    """ERA5 publishes speed + direction; the contract needs u/v."""
    result = ingest_svc.ingest(
        IngestionRequest(
            dataset_id="era5_wind",
            origin=DataOrigin.HISTORICAL,
            path=ingested_era5,
        )
    )
    variables = set(result.manifest.variables)
    assert "u_wind" in variables
    assert "v_wind" in variables
    # The components are labelled as derived, not observed.
    assert any("derived" in action for action in result.manifest.coordinate_actions)


def test_real_data_survives_the_phase0_preprocessing(
    ingest_svc: IngestionService, ingested_era5: Path, settings: Settings
) -> None:
    """The existing Phase 0 pipeline must accept an ingested real dataset."""
    from extreme_weather.data.preprocessing import preprocess_file

    result = ingest_svc.ingest(
        IngestionRequest(
            dataset_id="era5_pp",
            origin=DataOrigin.HISTORICAL,
            path=ingested_era5,
        )
    )
    processed = preprocess_file(
        result.output_path, settings.processed_dir / "era5_pp_processed.nc", max_gap=3
    )
    with xr.open_dataset(processed.output_path) as opened:
        dataset = opened.load()
    # All six contract variables survive; the pipeline added the derived wind.
    for name in (
        "temperature",
        "precipitation",
        "pressure",
        "relative_humidity",
        "u_wind",
        "v_wind",
    ):
        assert name in dataset.data_vars, f"{name} is missing after preprocessing"
    assert dataset.attrs["is_synthetic"] == "false"



def test_real_data_flows_through_phases_1_2_and_5(
    ingest_svc: IngestionService, ingested_era5: Path, settings: Settings
) -> None:
    """ingest -> Phase 0 -> Phase 1 events -> Phase 2 tracks -> Phase 5 alerts.

    This is the end-to-end claim of Phase 6: a real historical dataset runs
    through the *unchanged* Phase 1-5 pipeline. What is asserted is that each
    stage produces real output - not that the output is scientifically correct,
    because this repository contains no ground truth to check it against.
    """
    from extreme_weather.data.preprocessing import preprocess_file

    # --- Phase 6: ingest -------------------------------------------------
    result = ingest_svc.ingest(
        IngestionRequest(
            dataset_id="era5_chain",
            origin=DataOrigin.HISTORICAL,
            path=ingested_era5,
        )
    )
    assert result.quality.valid is True

    # --- Phase 0: preprocess ---------------------------------------------
    processed = preprocess_file(
        result.output_path, settings.processed_dir / "era5_chain_processed.nc", max_gap=3
    )

    # --- Phase 1: anomaly detection ---------------------------------------
    events_store = EventStore(settings.processed_dir / "era5_chain_events.json")
    detection = AnomalyDetectionService(store=events_store)
    # ``write_outputs=True`` so the events file Phase 2 reads actually exists:
    # a run that detected nothing still has to leave a valid (empty) analysis.
    analysis = detection.run(processed.output_path, write_outputs=True)
    assert analysis.event_count == len(analysis.events)
    assert events_store.exists()
    for event in analysis.events:
        assert -90.0 <= event.latitude <= 90.0
        assert -180.0 <= event.longitude <= 180.0
        assert event.intensity >= 0.0
        assert event.event_id.startswith("EVT-")

    # --- Phase 2: tracking -----------------------------------------------
    tracks_store = TrackStore(settings.processed_dir / "era5_chain_tracks.json")
    tracking = TrackingService(event_store=events_store, track_store=tracks_store)
    tracked = tracking.run(write_output=True)
    assert tracked.track_count == len(tracked.tracks)
    for track in tracked.tracks:
        assert track.end_time >= track.start_time
        assert track.max_speed_kmh >= 0.0
        assert track.duration_hours >= 0.0

    # --- Phase 5: alerting ------------------------------------------------
    alerts = AlertService(
        alert_store=AlertStore(settings.processed_dir / "era5_chain_alerts.json"),
        event_store=events_store,
        track_store=tracks_store,
    )
    generated = alerts.generate()
    assert generated.stored.alert_count == len(generated.stored.alerts)
    for alert in generated.stored.alerts:
        assert 0.0 <= alert.risk.score <= 100.0
        assert alert.risk.explanation
        # Real data is analysed with whatever models are available, and the
        # alert must say so rather than implying a prediction it lacks.
        assert "events" in alert.source_components
        assert "NOT an official warning" in alert.notice


def test_no_ground_truth_is_claimed_anywhere_in_the_real_chain(
    ingest_svc: IngestionService, ingested_era5: Path
) -> None:
    """Without a reference dataset the project must not score itself."""
    result = ingest_svc.ingest(
        IngestionRequest(
            dataset_id="era5_honest",
            origin=DataOrigin.HISTORICAL,
            path=ingested_era5,
        )
    )
    quality = str(result.manifest.quality_summary).lower()
    for forbidden in ("accuracy", "skill", "true_positive", "verified"):
        assert forbidden not in quality
    assert "synthetic" not in result.manifest.status_label.lower()


# --------------------------------------------------------------------------- #
# The synthetic path still works, through the same service
# --------------------------------------------------------------------------- #
def test_synthetic_data_still_ingests_as_synthetic(
    ingest_svc: IngestionService, settings: Settings
) -> None:
    """Phase 0-5's own generator must keep working through Phase 6."""
    from extreme_weather.data.generator import (
        ExtremeEventConfig,
        SyntheticConfig,
        generate_and_save,
    )

    config = SyntheticConfig(
        seed=4242,
        time_steps=6,
        start_time="2024-06-01T00:00:00",
        time_step_hours=1,
        latitude_min=12.0,
        latitude_max=13.5,
        longitude_min=76.5,
        longitude_max=77.5,
        resolution_deg=0.25,
        event=ExtremeEventConfig(
            enabled=True,
            center_latitude=12.5,
            center_longitude=77.0,
            start_index=2,
            duration_steps=2,
        ),
    )
    generated = generate_and_save(
        settings.synthetic_raw_dir / "generated.nc", config=config
    )
    result = ingest_svc.ingest(
        IngestionRequest(
            dataset_id="generated_demo",
            origin=DataOrigin.SYNTHETIC,
            path=generated.dataset_path,
        )
    )
    assert result.manifest.is_synthetic is True
    assert result.manifest.status_label == "SYNTHETIC"
    assert result.output_path == settings.synthetic_raw_dir / "generated_demo.nc"
    # And it is kept apart from the real datasets in the registry.
    ids = {
        entry.dataset_id
        for entry in ingest_svc.registry.list(origin=DataOrigin.SYNTHETIC)
    }
    assert "generated_demo" in ids

