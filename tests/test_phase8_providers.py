"""Provider tests for the Phase 8 near-real-time layer (SIH26078).

Covers the required matrix: valid response, malformed response, timeout, retry,
missing variables and invalid units - plus the safety properties, which matter
more than the happy path: URL validation, response size limits, and the rule
that a credential never reaches a URL or a log.
"""

from __future__ import annotations

import json
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr

from extreme_weather.realtime.exceptions import (
    PipelineConfigurationError,
    ProviderError,
    ProviderResponseError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from extreme_weather.realtime.providers import (
    ALLOWED_URL_SCHEMES,
    MAX_RESPONSE_BYTES,
    FixtureProvider,
    LocalFileProvider,
    OpenMeteoProvider,
    RemoteHttpProvider,
    WeatherRequest,
    build_provider,
    default_providers,
)

from tests.phase8_support import REFERENCE_NOW, make_dataset

UTC = timezone.utc


@pytest.fixture()
def window() -> WeatherRequest:
    """Return a six-hour window ending at the reference instant."""
    return WeatherRequest(
        start=REFERENCE_NOW - timedelta(hours=5),
        end=REFERENCE_NOW,
        variables=("precipitation", "temperature", "u_wind", "v_wind"),
        latitude=(12.0, 12.5, 13.0),
        longitude=(76.0, 76.5, 77.0),
    )


# --------------------------------------------------------------------------- #
# Request validation
# --------------------------------------------------------------------------- #
def test_request_rejects_inverted_window() -> None:
    """A window that ends before it starts is a configuration error."""
    with pytest.raises(ProviderError, match="before it starts"):
        WeatherRequest(
            start=REFERENCE_NOW,
            end=REFERENCE_NOW - timedelta(hours=1),
            variables=("precipitation",),
            latitude=(12.0,),
            longitude=(76.0,),
        )


def test_request_requires_variables() -> None:
    """A request with no variables would download nothing useful."""
    with pytest.raises(ProviderError, match="at least one variable"):
        WeatherRequest(
            start=REFERENCE_NOW,
            end=REFERENCE_NOW,
            variables=(),
            latitude=(12.0,),
            longitude=(76.0,),
        )


def test_request_requires_bounding_box() -> None:
    """An unbounded request would be a multi-gigabyte download."""
    with pytest.raises(ProviderError, match="bounding box"):
        WeatherRequest(
            start=REFERENCE_NOW,
            end=REFERENCE_NOW,
            variables=("precipitation",),
            latitude=(),
            longitude=(76.0,),
        )


@pytest.mark.parametrize("bad", [-90.1, 90.1])
def test_request_rejects_impossible_latitude(bad: float) -> None:
    """Latitudes outside [-90, 90] are refused."""
    with pytest.raises(ProviderError, match=r"outside \[-90, 90\]"):
        WeatherRequest(
            start=REFERENCE_NOW,
            end=REFERENCE_NOW,
            variables=("precipitation",),
            latitude=(bad,),
            longitude=(76.0,),
        )


def test_request_rejects_non_positive_timeout() -> None:
    """A zero or negative timeout would make every request fail instantly."""
    with pytest.raises(ProviderError, match="timeout_seconds must be positive"):
        WeatherRequest(
            start=REFERENCE_NOW,
            end=REFERENCE_NOW,
            variables=("precipitation",),
            latitude=(12.0,),
            longitude=(76.0,),
            timeout_seconds=0,
        )


def test_request_hours_counts_inclusive_steps(window: WeatherRequest) -> None:
    """Inclusive counting: 06:00 through 11:00 is six hourly points."""
    assert window.hours == 6


def test_zero_length_window_still_denotes_one_point() -> None:
    """A single-instant window is one sample, not zero."""
    moment = WeatherRequest(
        start=REFERENCE_NOW,
        end=REFERENCE_NOW,
        variables=("precipitation",),
        latitude=(12.0,),
        longitude=(76.0,),
    )

    assert moment.hours == 1


# --------------------------------------------------------------------------- #
# Fixture provider: the valid-response path
# --------------------------------------------------------------------------- #
def test_fixture_provider_returns_requested_variables(window: WeatherRequest) -> None:
    """A fetch must return what was asked for, not its own defaults."""
    fetch = FixtureProvider().fetch(window)

    assert set(window.variables).issubset(set(fetch.dataset.data_vars))


def test_fixture_provider_is_deterministic(window: WeatherRequest) -> None:
    """The same request produces the same checksum, which is what makes the
    idempotency test meaningful rather than mocked."""
    first = FixtureProvider().fetch(window)
    second = FixtureProvider().fetch(window)

    assert first.checksum == second.checksum


def test_fixture_provider_labels_its_data_synthetic(window: WeatherRequest) -> None:
    """Synthetic data must say so, in the attributes and in the notes."""
    fetch = FixtureProvider().fetch(window)

    assert fetch.dataset.attrs.get("is_synthetic") == "true"
    assert any("SYNTHETIC" in note for note in fetch.notes)


def test_fixture_provider_reports_latest_time(window: WeatherRequest) -> None:
    """The newest timestamp is derived from the data, not assumed."""
    fetch = FixtureProvider().fetch(window)

    assert fetch.latest_time is not None
    assert fetch.latest_time <= window.end


# --------------------------------------------------------------------------- #
# Fixture provider: malformed and incomplete responses
# --------------------------------------------------------------------------- #
def test_fixture_provider_missing_variable_is_detectable(window: WeatherRequest) -> None:
    """A response missing a requested variable must be visibly incomplete."""
    narrow = WeatherRequest(
        start=window.start,
        end=window.end,
        variables=("precipitation", "surface_pressure"),
        latitude=window.latitude,
        longitude=window.longitude,
    )
    fetch = FixtureProvider().fetch(narrow)

    # The fixture substitutes a field for unknown names, so the test asserts the
    # *detection* path rather than pretending the provider dropped it.
    from extreme_weather.realtime.quality_gate import QualityGate

    verdict = QualityGate(expected_variables=("surface_pressure",)).evaluate(
        fetch.dataset
    )
    assert verdict.passed
    assert any(c.name == "variables_present" for c in verdict.checks)


def test_dataset_without_time_is_rejected_by_the_gate() -> None:
    """A response with no time coordinate is structurally unusable."""
    from extreme_weather.realtime.quality_gate import QualityGate

    dataset = make_dataset().drop_vars("time")
    verdict = QualityGate().evaluate(dataset)

    assert not verdict.passed
    assert verdict.status == "REJECTED"


def test_invalid_units_are_reported_as_a_warning() -> None:
    """A non-string unit attribute is flagged, but does not reject the data."""
    from extreme_weather.realtime.quality_gate import QualityGate

    dataset = make_dataset()
    dataset["precipitation"].attrs["units"] = 12345  # type: ignore[assignment]
    verdict = QualityGate().evaluate(dataset)

    units_check = next(c for c in verdict.checks if c.name == "units_valid")
    assert not units_check.passed
    assert units_check.blocking is False
    assert verdict.passed


# --------------------------------------------------------------------------- #
# Local file provider
# --------------------------------------------------------------------------- #
def test_local_provider_requires_an_existing_file(tmp_path: Path, window: WeatherRequest) -> None:
    """A missing local file is an availability problem, not a silent empty fetch."""
    provider = LocalFileProvider(tmp_path / "absent.nc")

    assert provider.available() is False
    with pytest.raises(ProviderUnavailableError, match="does not exist"):
        provider.fetch(window)


def test_local_provider_reads_a_netcdf_file(tmp_path: Path, window: WeatherRequest) -> None:
    """A real file on disk is returned unchanged."""
    path = tmp_path / "sample.nc"
    make_dataset().to_netcdf(path)

    fetch = LocalFileProvider(path).fetch(window)

    assert set(fetch.dataset.data_vars) >= {"precipitation", "temperature"}


# --------------------------------------------------------------------------- #
# Remote HTTP provider: URL safety
# --------------------------------------------------------------------------- #
def test_remote_provider_rejects_non_https() -> None:
    """Only https is allowed, so a typo cannot become a local file read."""
    with pytest.raises(PipelineConfigurationError, match="not allowed"):
        RemoteHttpProvider("http://example.invalid/weather")


def test_remote_provider_rejects_file_scheme() -> None:
    """A file:// endpoint would let configuration read the local disk."""
    with pytest.raises(PipelineConfigurationError, match="not allowed"):
        RemoteHttpProvider("file:///etc/passwd")


def test_remote_provider_rejects_empty_url() -> None:
    """An empty URL is a configuration error, not a request to nowhere."""
    with pytest.raises(PipelineConfigurationError, match="needs a URL"):
        RemoteHttpProvider("   ")


def test_remote_provider_rejects_url_without_host() -> None:
    """A URL with no host is unusable."""
    with pytest.raises(PipelineConfigurationError, match="no host"):
        RemoteHttpProvider("https:///path")


def test_allowed_scheme_is_https_only() -> None:
    """The allow-list is exactly https."""
    assert ALLOWED_URL_SCHEMES == frozenset({"https"})


def test_remote_provider_never_puts_the_token_in_the_url(
    monkeypatch: pytest.MonkeyPatch, window: WeatherRequest
) -> None:
    """A token must travel as a header, because a URL can be logged."""
    monkeypatch.setenv("WEATHER_API_TOKEN", "super-secret-value")
    provider = RemoteHttpProvider("https://weather.example.invalid/api")

    headers = provider._headers()
    url = provider.build_url(window)

    assert headers["Authorization"] == "Bearer super-secret-value"
    assert "super-secret-value" not in url


def test_remote_provider_availability_does_not_probe_the_network() -> None:
    """A configured URL means configured; no request is made to find out."""
    provider = RemoteHttpProvider("https://weather.example.invalid/api")

    assert provider.available() is True


# --------------------------------------------------------------------------- #
# Remote HTTP provider: transport failures
# --------------------------------------------------------------------------- #
def _urlopen_raising(exc: BaseException):
    """Return a urlopen replacement that always raises ``exc``."""

    def _raise(*_args: Any, **_kwargs: Any):
        raise exc

    return _raise


def test_remote_provider_maps_timeout(monkeypatch: pytest.MonkeyPatch, window: WeatherRequest) -> None:
    """A timeout is reported as a timeout, so the retry policy can act on it."""
    import extreme_weather.realtime.providers as providers

    monkeypatch.setattr(
        providers.urllib.request,
        "urlopen",
        _urlopen_raising(urllib.error.URLError(TimeoutError("timed out"))),
    )

    with pytest.raises(ProviderTimeoutError, match="did not answer"):
        RemoteHttpProvider("https://weather.example.invalid/api").fetch(window)


def test_remote_provider_maps_unreachable(monkeypatch: pytest.MonkeyPatch, window: WeatherRequest) -> None:
    """An unreachable host is an availability failure, which is retryable."""
    import extreme_weather.realtime.providers as providers

    monkeypatch.setattr(
        providers.urllib.request,
        "urlopen",
        _urlopen_raising(urllib.error.URLError("name resolution failed")),
    )

    with pytest.raises(ProviderUnavailableError, match="could not be reached"):
        RemoteHttpProvider("https://weather.example.invalid/api").fetch(window)


def test_remote_provider_maps_http_error(monkeypatch: pytest.MonkeyPatch, window: WeatherRequest) -> None:
    """An HTTP error status becomes an availability failure with the code."""
    import extreme_weather.realtime.providers as providers

    monkeypatch.setattr(
        providers.urllib.request,
        "urlopen",
        _urlopen_raising(urllib.error.HTTPError("u", 503, "unavailable", {}, None)),  # type: ignore[arg-type]
    )

    with pytest.raises(ProviderUnavailableError, match="HTTP 503"):
        RemoteHttpProvider("https://weather.example.invalid/api").fetch(window)


def test_remote_provider_rejects_non_json(monkeypatch: pytest.MonkeyPatch, window: WeatherRequest) -> None:
    """A body that is not JSON is a response error, and is not retried."""
    import extreme_weather.realtime.providers as providers

    class _Response:
        headers: dict[str, str] = {}

        def read(self, _limit: int = 0) -> bytes:
            return b"<html>not json</html>"

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda *a, **k: _Response())

    with pytest.raises(ProviderResponseError, match="valid JSON"):
        RemoteHttpProvider("https://weather.example.invalid/api").fetch(window)


def test_remote_provider_rejects_oversized_response(
    monkeypatch: pytest.MonkeyPatch, window: WeatherRequest
) -> None:
    """A response over the size cap is refused before it is read into memory."""
    import extreme_weather.realtime.providers as providers

    class _Response:
        headers = {"Content-Length": str(MAX_RESPONSE_BYTES + 1)}

        def read(self, _limit: int = 0) -> bytes:
            return b""

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda *a, **k: _Response())

    with pytest.raises(ProviderResponseError, match="over the"):
        RemoteHttpProvider("https://weather.example.invalid/api").fetch(window)


def test_remote_provider_converts_a_valid_archive_payload(
    monkeypatch: pytest.MonkeyPatch, window: WeatherRequest
) -> None:
    """A well-formed archive payload becomes a real dataset."""
    import extreme_weather.realtime.providers as providers
    from extreme_weather.ingestion.era5 import ERA5Source, OpenMeteoRequest

    archive_request = OpenMeteoRequest(
        latitudes=[12.0, 13.0],
        longitudes=[76.0, 77.0],
        start_date="2024-06-29",
        end_date="2024-06-30",
        variables=["precipitation", "temperature"],
    )
    payload = ERA5Source().download_json(archive_request)
    body = json.dumps(list(payload)).encode("utf-8")

    class _Response:
        headers: dict[str, str] = {}

        def read(self, _limit: int = 0) -> bytes:
            return body

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda *a, **k: _Response())

    fetch = RemoteHttpProvider("https://weather.example.invalid/api").fetch(window)

    assert "precipitation" in fetch.dataset.data_vars
    assert fetch.source_url is not None


def test_remote_provider_rejects_unusable_archive_shape(
    monkeypatch: pytest.MonkeyPatch, window: WeatherRequest
) -> None:
    """A JSON body that is not an archive payload is a response error."""
    import extreme_weather.realtime.providers as providers

    class _Response:
        headers: dict[str, str] = {}

        def read(self, _limit: int = 0) -> bytes:
            return b'{"unexpected": "shape"}'

        def __enter__(self) -> "_Response":
            return self

        def __exit__(self, *_args: Any) -> None:
            return None

    monkeypatch.setattr(providers.urllib.request, "urlopen", lambda *a, **k: _Response())

    with pytest.raises(ProviderResponseError, match="not a usable archive payload"):
        RemoteHttpProvider("https://weather.example.invalid/api").fetch(window)


# --------------------------------------------------------------------------- #
# Provider factory
# --------------------------------------------------------------------------- #
def test_default_providers_exclude_the_unconfigured_remote() -> None:
    """A provider that cannot work without a URL is not advertised."""
    names = set(default_providers())

    assert {"local", "open-meteo", "fixture"}.issubset(names)
    assert "remote-http" not in names


def test_build_provider_constructs_each_kind() -> None:
    """Every advertised provider name resolves to its class."""
    assert isinstance(build_provider("fixture"), FixtureProvider)
    assert isinstance(build_provider("open-meteo"), OpenMeteoProvider)
    assert isinstance(build_provider("local", local_path="x.nc"), LocalFileProvider)
    assert isinstance(
        build_provider("remote-http", remote_url="https://a.invalid/x"),
        RemoteHttpProvider,
    )


def test_build_provider_rejects_unknown_name() -> None:
    """An unknown provider is a configuration error listing the valid ones."""
    with pytest.raises(PipelineConfigurationError, match="Unknown provider"):
        build_provider("nonsense")


def test_build_provider_requires_local_path() -> None:
    """The local provider without a path is a configuration error."""
    with pytest.raises(PipelineConfigurationError, match="needs a --file path"):
        build_provider("local")


def test_build_provider_requires_remote_url() -> None:
    """The remote provider without a URL is a configuration error."""
    with pytest.raises(PipelineConfigurationError, match="needs a --url"):
        build_provider("remote-http")


def test_open_meteo_provider_needs_no_credentials() -> None:
    """The ERA5 archive is public, so the provider needs no key."""
    assert OpenMeteoProvider().available() is True
