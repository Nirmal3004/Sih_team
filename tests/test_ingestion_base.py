"""Tests for the Phase 6 data source interface and registry of sources.

Covers the provider-independent contract: every source answers ``name``,
``available``, ``fetch``, ``load`` and ``metadata``; an unavailable source says
so without raising; and a source that needs credentials names the environment
variable rather than inventing data.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from extreme_weather.config import Settings
from extreme_weather.ingestion import (
    DataOrigin,
    ERA5Source,
    GRIBSource,
    IngestionService,
    NetCDFSource,
    NWPSource,
    OpenMeteoRequest,
    NWPFormRequest,
    SourceInfo,
    WeatherDataSource,
    default_sources,
    grib_backend_available,
    grib_backend_name,
)
from extreme_weather.ingestion.exceptions import (
    DataSourceUnavailableError,
    ProviderCredentialsMissingError,
    SourceNotFoundError,
)

SOURCES = (NetCDFSource(), GRIBSource(), ERA5Source(), NWPSource())


# --------------------------------------------------------------------------- #
# The interface
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.name())
def test_every_source_satisfies_the_interface(source: object) -> None:
    assert isinstance(source, WeatherDataSource)
    for method in ("name", "available", "fetch", "load", "metadata", "default_origin"):
        assert callable(getattr(source, method))


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.name())
def test_name_is_a_non_empty_stable_string(source: object) -> None:
    name = source.name()
    assert isinstance(name, str) and name
    assert name == source.name()
    assert name.islower()


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.name())
def test_available_never_raises_and_returns_a_bool(source: object) -> None:
    """A dashboard must be able to ask "can I use this?" without a try/except."""
    assert isinstance(source.available(), bool)


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.name())
def test_metadata_is_a_serialisable_source_info(source: object) -> None:
    info = source.metadata()
    assert isinstance(info, SourceInfo)
    assert info.name == source.name()
    assert info.kind in ("file", "remote")
    assert isinstance(info.as_dict()["formats"], list)


@pytest.mark.parametrize("source", SOURCES, ids=lambda s: s.name())
def test_default_origin_is_never_synthetic_for_real_sources(source: object) -> None:
    """NetCDF/GRIB/ERA5/NWP all describe real or undetermined data, not synthetic."""
    assert DataOrigin.validate(source.default_origin()) in (
        DataOrigin.HISTORICAL,
        DataOrigin.EXTERNAL,
    )


# --------------------------------------------------------------------------- #
# No source claims to be live
# --------------------------------------------------------------------------- #
def test_era5_is_labelled_historical_not_live() -> None:
    info = ERA5Source().metadata()
    assert info.default_origin == DataOrigin.HISTORICAL
    assert "not live" in info.notes.lower()
    assert "not a forecast" in info.notes.lower()


def test_era5_requires_no_credentials() -> None:
    info = ERA5Source().metadata()
    assert info.requires_credentials is False
    assert info.required_environment == ()


def test_era5_records_its_attribution_and_licence() -> None:
    source = ERA5Source()
    assert "ERA5" in source.attribution
    assert "ECMWF" in source.license
    assert "reanalysis" in source.metadata().description.lower()


# --------------------------------------------------------------------------- #
# GRIB optional dependency
# --------------------------------------------------------------------------- #
def test_grib_availability_tracks_the_optional_engine() -> None:
    expected = grib_backend_available()
    assert GRIBSource().available() is expected
    assert (grib_backend_name() is not None) is expected


def test_grib_metadata_names_the_engine_or_how_to_install_it() -> None:
    info = GRIBSource().metadata()
    if grib_backend_available():
        assert info.optional_dependency == grib_backend_name()
    else:
        assert "pip install" in (info.optional_dependency or "")
        assert "cfgrib" in (info.optional_dependency or "")


def test_missing_grib_engine_raises_a_clear_error(tmp_path: Path) -> None:
    """A GRIB file without its backend must not silently yield an empty dataset."""
    if grib_backend_available():
        pytest.skip("A GRIB engine is installed, so the missing-engine path cannot run.")
    path = tmp_path / "field.grib"
    path.write_bytes(b"GRIB\x00\x00\x00\x00")
    with pytest.raises(Exception) as info:
        GRIBSource().load(path)
    assert "cfgrib" in str(info.value).lower()


# --------------------------------------------------------------------------- #
# Credentials
# --------------------------------------------------------------------------- #
def test_nwp_metadata_reports_configuration_without_reading_a_token(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NWP_API_URL", raising=False)
    monkeypatch.delenv("NWP_API_TOKEN", raising=False)
    info = NWPSource().metadata()
    assert "Remote endpoint configured: False" in info.notes
    assert "never logged" in info.notes


def test_nwp_without_a_token_reports_the_missing_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A protected endpoint without a token must say which variable to set."""
    monkeypatch.delenv("NWP_API_TOKEN", raising=False)
    request = NWPFormRequest(
        url="https://api.example.invalid/grib", lat0=10.0, lat1=12.0, lon0=70.0, lon1=72.0
    )
    with pytest.raises(ProviderCredentialsMissingError) as info:
        NWPSource(cache_dir=tmp_path).fetch(request)
    message = str(info.value)
    assert "NWP_API_TOKEN" in message
    assert "Bearer" not in message


def test_nwp_uses_a_token_when_present(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With a token set, the request is attempted rather than refused."""
    monkeypatch.setenv("NWP_API_TOKEN", "test-token-value")
    request = NWPFormRequest(
        url="https://api.example.invalid/grib", lat0=10.0, lat1=12.0, lon0=70.0, lon1=72.0
    )
    # No network in tests: the call must fail as a *network* problem, which
    # proves the credential gate was passed rather than tripped.
    with pytest.raises(DataSourceUnavailableError):
        NWPSource(cache_dir=tmp_path).fetch(request)


def test_nwp_fetch_requires_a_request_for_the_remote_route(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="needs an OpenMeteoRequest|exactly one"):
        ERA5Source(cache_dir=tmp_path).fetch()  # type: ignore[call-arg]


# --------------------------------------------------------------------------- #
# The default registry
# --------------------------------------------------------------------------- #
def test_default_sources_registers_all_four(settings: Settings) -> None:
    assert set(default_sources(settings)) == {"netcdf", "grib", "era5", "nwp"}


def test_default_sources_include_unavailable_ones(settings: Settings) -> None:
    """An unusable source is still listed so the API can explain why."""
    if not grib_backend_available():
        assert "grib" in default_sources(settings)


def test_service_accepts_a_custom_source(settings: Settings) -> None:
    service = IngestionService(settings=settings)
    service.register_source(NetCDFSource())
    assert service.get_source("netcdf").name() == "netcdf"


def test_service_rejects_an_unknown_source(settings: Settings) -> None:
    with pytest.raises(SourceNotFoundError, match="not registered"):
        IngestionService(settings=settings).get_source("definitely-not-a-source")


# --------------------------------------------------------------------------- #
# The ERA5 request object (no network)
# --------------------------------------------------------------------------- #
def test_era5_request_validates_its_bounds() -> None:
    with pytest.raises(ValueError, match="outside"):
        OpenMeteoRequest(
            latitudes=(95.0,),
            longitudes=(0.0,),
            start_date="2024-01-01",
            end_date="2024-01-02",
        )
    with pytest.raises(ValueError, match="after end_date"):
        OpenMeteoRequest(
            latitudes=(0.0,),
            longitudes=(0.0,),
            start_date="2024-01-05",
            end_date="2024-01-02",
        )
    with pytest.raises(ValueError, match="at least one point"):
        OpenMeteoRequest(
            latitudes=(),
            longitudes=(0.0,),
            start_date="2024-01-01",
            end_date="2024-01-02",
        )


def test_era5_request_url_is_bounded_and_has_no_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OPEN_METEO_API_KEY", raising=False)
    request = OpenMeteoRequest(
        latitudes=(12.0, 13.0),
        longitudes=(77.0,),
        start_date="2024-06-01",
        end_date="2024-06-02",
        variables=("precipitation",),
    )
    url = request.build_url()
    assert request.point_count() == 2
    assert "archive-api.open-meteo.com" in url
    assert "apikey" not in url
    assert "2024-06-01" in url and "2024-06-02" in url


def test_era5_url_requests_the_units_the_project_expects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Asking the API for m/s avoids a km/h conversion on the way in."""
    monkeypatch.delenv("OPEN_METEO_API_KEY", raising=False)
    url = OpenMeteoRequest(
        latitudes=(12.0,),
        longitudes=(77.0,),
        start_date="2024-06-01",
        end_date="2024-06-01",
    ).build_url()
    assert "wind_speed_unit=ms" in url
    assert "temperature_unit=celsius" in url
    assert "timezone=UTC" in url


# --------------------------------------------------------------------------- #
# Origin semantics
# --------------------------------------------------------------------------- #
def test_data_origin_validates_its_values() -> None:
    assert DataOrigin.validate("synthetic") == DataOrigin.SYNTHETIC
    assert DataOrigin.validate("HISTORICAL") == DataOrigin.HISTORICAL
    assert DataOrigin.validate(" external ") == DataOrigin.EXTERNAL
    with pytest.raises(ValueError, match="Unknown data origin"):
        DataOrigin.validate("LIVE")


def test_the_three_origins_are_distinct_words() -> None:
    """SYNTHETIC, HISTORICAL and EXTERNAL must never be conflated."""
    assert len(set(DataOrigin.ALL)) == 3
    assert "LIVE" not in DataOrigin.ALL
    assert "REALTIME" not in DataOrigin.ALL

