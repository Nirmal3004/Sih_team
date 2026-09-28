"""Tests for the ingestion service (SIH26078, Phase 6).

The service is what the CLI and the API both call, so these tests assert the
whole chain rather than one step: a real file goes in and a registered,
normalised, manifested, validated dataset comes out - and a dry run goes
through every one of those steps while touching nothing on disk.

The two properties worth reading the whole file for:

* **synthetic and real data are written to different directories**, decided by
  the request's ``origin``, with no default to fall back on;
* **ingestion is deterministic**: the same input file produces the same content
  hash, so re-ingesting is a no-op rather than a source of drift.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from extreme_weather.config import Settings
from extreme_weather.ingestion import (
    DataOrigin,
    IngestionRequest,
    IngestionService,
    derive_uv_from_polar_wind,
    subset_dataset,
)
from extreme_weather.ingestion.exceptions import SourceNotFoundError

from tests.ingestion_fixtures import (
    make_dataset,
    service as ingest_svc,
    write_netcdf,
)

UTC = timezone.utc
NOW = datetime(2024, 6, 1, 12, 0, tzinfo=UTC)
SHAPE = (4, 3, 3)


def request_for(path: Path, dataset_id: str = "demo", **overrides) -> IngestionRequest:
    """Build an ingest request for a local file."""
    payload: dict = {"dataset_id": dataset_id, "origin": DataOrigin.HISTORICAL, "path": path}
    payload.update(overrides)
    return IngestionRequest(**payload)


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #
def test_a_request_validates_its_origin_eagerly() -> None:
    """A bad origin must fail at the boundary, not halfway through a write."""
    with pytest.raises(ValueError, match="Unknown data origin"):
        request_for(Path("x.nc"), origin="LIVE")


def test_a_request_rejects_an_empty_dataset_id() -> None:
    with pytest.raises(ValueError, match="dataset_id"):
        request_for(Path("x.nc"), dataset_id="  ")


def test_a_request_without_a_path_is_rejected_at_ingest(ingest_svc: IngestionService) -> None:
    request = IngestionRequest(dataset_id="demo", origin=DataOrigin.HISTORICAL)
    with pytest.raises(ValueError, match="path is required"):
        ingest_svc.ingest(request)


# --------------------------------------------------------------------------- #
# The happy path
# --------------------------------------------------------------------------- #
def test_a_full_ingest_produces_a_registered_dataset(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    result = ingest_svc.ingest(request_for(contract_file, "contract"))

    assert result.entry is not None
    assert result.manifest.dataset_id == "contract"
    assert result.quality.valid is True
    assert result.output_path is not None and result.output_path.is_file()
    assert "contract" in ingest_svc.registry.ids()

    joined = " ".join(result.steps)
    for stage in ("ingested", "loaded", "normalized", "registered"):
        assert stage in joined


def test_the_manifest_records_the_standard_units(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    result = ingest_svc.ingest(request_for(contract_file, "contract"))
    units = result.manifest.units
    assert units["temperature"] == "K"
    assert units["pressure"] == "Pa"
    assert units["precipitation"] == "mm h-1"


def test_the_manifest_records_the_provenance_chain(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    result = ingest_svc.ingest(request_for(contract_file, "contract"))
    stages = [step.stage for step in result.manifest.processing_steps]
    assert stages == ["ingested", "validated", "normalized"]
    ingested = result.manifest.processing_steps[0]
    assert len(ingested.input_hash or "") == 64
    assert "netcdf" in ingested.tool


def test_a_local_file_ingest_never_invents_a_url(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    result = ingest_svc.ingest(request_for(contract_file, "contract"))
    assert result.manifest.source == "local_file"
    assert result.manifest.source_url is None
    assert result.manifest.is_live is False


def test_ingestion_is_deterministic(
    ingest_svc: IngestionService, source_file: Path, settings: Settings
) -> None:
    """Re-ingesting the same file must not change the data.

    The source lives outside the directories ingestion writes to, so the second
    run reads the same input rather than its own output.
    """
    first = ingest_svc.ingest(request_for(source_file, "repeat"))
    second = ingest_svc.ingest(request_for(source_file, "repeat"))
    # The content hash digests the array values, so it survives the container
    # metadata NetCDF writes. (The *file* hash legitimately differs between
    # two writes, which is exactly why the content hash exists.)
    assert first.manifest.content_hash == second.manifest.content_hash
    assert first.output_path == second.output_path == settings.real_raw_dir / "repeat.nc"
    assert second.manifest.processing_steps[0].input_hash == first.manifest.processing_steps[0].input_hash


# --------------------------------------------------------------------------- #
# Synthetic / real separation
# --------------------------------------------------------------------------- #
def test_real_data_is_written_under_raw_real(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    result = ingest_svc.ingest(
        request_for(
            contract_file, "real_one", origin=DataOrigin.HISTORICAL
        )
    )
    assert result.output_path == settings.real_raw_dir / "real_one.nc"
    assert result.output_path.is_file()
    assert result.manifest.is_synthetic is False


def test_synthetic_data_is_written_under_raw_synthetic(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    result = ingest_svc.ingest(
        request_for(
            contract_file, "fake_one", origin=DataOrigin.SYNTHETIC
        )
    )
    assert result.output_path == settings.synthetic_raw_dir / "fake_one.nc"
    assert result.output_path.is_file()
    assert result.manifest.is_synthetic is True
    # The two origins land in different places even for the same input file.
    assert not (settings.real_raw_dir / "fake_one.nc").exists()


# --------------------------------------------------------------------------- #
# Dry run
# --------------------------------------------------------------------------- #
def test_a_dry_run_writes_nothing_and_registers_nothing(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    result = ingest_svc.ingest(
        request_for(contract_file, "preview", dry_run=True)
    )
    assert result.dry_run is True
    assert result.output_path is None
    assert result.entry is None
    assert ingest_svc.registry.ids() == []
    assert not (settings.real_raw_dir / "preview.nc").exists()
    # The work was still done, so the preview is informative.
    assert result.quality.valid is True
    assert any("would write" in step for step in result.steps)


def test_a_dry_run_leaves_a_populated_registry_untouched(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    ingest_svc.ingest(request_for(contract_file, "real_one"))
    before = ingest_svc.registry.ids()
    manifests = sorted(path.name for path in settings.manifests_dir.glob("*.json"))

    ingest_svc.ingest(
        request_for(contract_file, "real_one", dry_run=True)
    )
    assert ingest_svc.registry.ids() == before
    assert sorted(path.name for path in settings.manifests_dir.glob("*.json")) == manifests


# --------------------------------------------------------------------------- #
# Failure modes
# --------------------------------------------------------------------------- #
def test_an_unknown_source_is_rejected(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    with pytest.raises(SourceNotFoundError, match="not registered"):
        ingest_svc.ingest(
            request_for(contract_file, "x", source_name="nope")
        )


def test_a_missing_file_is_rejected(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    with pytest.raises(SourceNotFoundError):
        ingest_svc.ingest(request_for(settings.real_raw_dir / "gone.nc", "x"))



# --------------------------------------------------------------------------- #
# Unit conversion through the whole service
# --------------------------------------------------------------------------- #
def test_the_service_converts_celsius_and_hectopascals(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    path = write_netcdf(
        make_dataset(
            variables={
                "temperature_2m": (np.full(SHAPE, 27.0), "degC"),
                "precipitation": (np.full(SHAPE, 3.0), "mm"),
                "surface_pressure": (np.full(SHAPE, 1013.0), "hPa"),
            }
        ),
        settings.real_raw_dir / "celsius.nc",
    )
    result = ingest_svc.ingest(request_for(path, "celsius"))
    conversions = result.manifest.unit_conversions

    assert conversions["temperature"]["original_units"] == "degC"
    assert conversions["temperature"]["normalized_units"] == "K"
    assert "273.15" in conversions["temperature"]["conversion_method"]
    assert conversions["pressure"]["conversion_method"] == "hPa -> Pa (x100)"

    with xr.open_dataset(result.output_path) as written:
        assert float(written["temperature"].values.mean()) == pytest.approx(300.15, abs=1e-4)


def test_missing_variables_are_reported_not_invented(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    path = write_netcdf(
        make_dataset(variables={"precipitation": (np.full(SHAPE, 3.0), "mm h-1")}),
        settings.real_raw_dir / "partial.nc",
    )
    result = ingest_svc.ingest(request_for(path, "partial"))
    assert set(result.manifest.variables_missing) >= {"temperature", "u_wind", "v_wind"}
    assert any("absent" in warning for warning in result.warnings)


def test_the_variable_whitelist_is_applied(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    result = ingest_svc.ingest(
        request_for(
            contract_file, "subset", variables=("precipitation",)
        )
    )
    assert set(result.manifest.variables) == {"precipitation"}


# --------------------------------------------------------------------------- #
# Subsets
# --------------------------------------------------------------------------- #
def test_subset_dataset_restricts_the_box() -> None:
    dataset = make_dataset(latitude=np.array([10.0, 11.0, 12.0]))
    subset = subset_dataset(dataset, latitude=(11.0, 12.0))
    assert list(subset.latitude.values) == [11.0, 12.0]


def test_subset_dataset_is_a_no_op_without_bounds() -> None:
    dataset = make_dataset()
    assert subset_dataset(dataset).sizes == dataset.sizes


def test_an_ingest_can_apply_a_spatial_subset(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    result = ingest_svc.ingest(
        request_for(contract_file, "boxed", latitude=(12.4, 13.0))
    )
    assert "subset to lat=" in " ".join(result.steps)


# --------------------------------------------------------------------------- #
# Polar wind -> u/v components
# --------------------------------------------------------------------------- #
def test_polar_wind_is_converted_to_components() -> None:
    dataset = make_dataset(
        variables={
            "wind_speed": (np.full(SHAPE, 10.0), "m s-1"),
            "wind_direction": (np.full(SHAPE, 0.0), "degree"),
        }
    )
    enriched, actions = derive_uv_from_polar_wind(dataset)
    assert "u_wind" in enriched.data_vars
    assert "v_wind" in enriched.data_vars
    # Wind from the north blows towards the south: u=0, v=-speed.
    assert float(enriched["u_wind"].mean()) == pytest.approx(0.0, abs=1e-9)
    assert float(enriched["v_wind"].mean()) == pytest.approx(-10.0)
    assert actions and "u_wind/v_wind" in actions[0]


def test_polar_wind_conversion_is_a_no_op_when_components_exist() -> None:
    dataset = make_dataset()
    enriched, actions = derive_uv_from_polar_wind(dataset)
    assert list(enriched.data_vars) == list(dataset.data_vars)
    assert actions == []


def test_polar_wind_is_a_no_op_when_neither_is_present() -> None:
    dataset = make_dataset(variables={"precipitation": (np.full(SHAPE, 1.0), "mm h-1")})
    enriched, actions = derive_uv_from_polar_wind(dataset)
    assert "u_wind" not in enriched.data_vars
    assert actions == []


def test_the_service_records_that_the_components_were_derived(
    ingest_svc: IngestionService, contract_file: Path, settings: Settings
) -> None:
    """A derived component must be labelled as derived, never as observed."""
    path = write_netcdf(
        make_dataset(
            variables={
                "precipitation": (np.full(SHAPE, 3.0), "mm h-1"),
                "temperature": (np.full(SHAPE, 300.0), "K"),
                "pressure": (np.full(SHAPE, 101_000.0), "Pa"),
                "relative_humidity": (np.full(SHAPE, 50.0), "%"),
                "wind_speed": (np.full(SHAPE, 8.0), "m s-1"),
                "wind_direction": (np.full(SHAPE, 270.0), "degree"),
            }
        ),
        settings.real_raw_dir / "polar.nc",
    )
    result = ingest_svc.ingest(request_for(path, "polar"))
    assert "u_wind" in result.manifest.variables
    assert "v_wind" in result.manifest.variables
    assert any("derived" in action for action in result.manifest.coordinate_actions)
    # The intermediates are dropped once the components exist.
    assert "wind_speed" not in result.manifest.variables

