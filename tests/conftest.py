"""Shared pytest fixtures for the DeepWeather AI (SIH26078) test suite.

The fixtures use intentionally small grids (16 x 9 x 9) so the whole suite runs
in a couple of seconds, while still exercising every code path of the pipeline.
All data used by the tests is synthetic and generated through the public API.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path
from typing import Iterator

import pytest
from fastapi.testclient import TestClient

# Make `src/` importable when pytest is invoked without the pyproject
# `pythonpath` setting (for example with an unusual rootdir).
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extreme_weather.api.main import create_app  # noqa: E402
from extreme_weather.config import PACKAGE_VERSION, Settings, get_settings  # noqa: E402
from extreme_weather.realtime.orchestrator import PipelineOrchestrator  # noqa: E402
from extreme_weather.realtime.providers import WeatherRequest  # noqa: E402

# The Phase 3 GNN fixtures live in `tests/gnn_fixtures.py`; re-exporting them here
# makes them available to every test module without each one importing them.
# The Phase 9 evaluation fixtures live in `tests/phase9_support.py`; they are
# re-exported here for the same reason as the GNN fixtures above.
from tests.phase9_support import (  # noqa: E402,F401
    event,
    samples,
    track,
    validation_samples,
)


from tests.gnn_fixtures import (  # noqa: E402,F401
    many_tracks,
    single_track,
    two_far_tracks,
    two_nearby_tracks,
)

# The Phase 4 downscaling fixtures live in `tests/downscaling_fixtures.py`.
from tests.downscaling_fixtures import (  # noqa: E402,F401
    ds_coarse_field,
    tiny_downscaling_dataset,
)
# The Phase 5 alerting fixtures live in `tests/alert_fixtures.py`; re-exporting
# them here makes them available to every Phase 5 test module.
from tests.alert_fixtures import (  # noqa: E402,F401
    alert_client,
    alert_settings,
    build_service,
    empty_alert_client,
    generated_alerts,
    make_event,
    seeded_pipeline,
    service,
    storm_events,
    write_events,
)
# The Phase 6 ingestion fixtures live in `tests/ingestion_fixtures.py`.
from tests.ingestion_fixtures import (  # noqa: E402,F401
    contract_dataset,
    contract_file,
    data_client,
    era5_dataset,
    era5_payload,
    make_dataset,
    service as ingest_svc,
    service_without_grib,
    settings,
    source_file,
    write_netcdf,
)
# The Phase 7 validation fixtures live in 	ests/validation_fixtures.py.
from tests.validation_fixtures import (  # noqa: E402,F401
    long_dataset,
    make_times,
    make_validation_dataset,
    reports_dir,
    short_dataset,
    validation_client,
    validation_dataset,
    validation_settings,
)
from extreme_weather.data.generator import (  # noqa: E402
    ExtremeEventConfig,
    GenerationResult,
    SyntheticConfig,
    generate_and_save,
    generate_synthetic_dataset,
)
from extreme_weather.data.preprocessing import PreprocessingResult, preprocess_file  # noqa: E402

#: Seed used by every test fixture (kept explicit so failures are reproducible).
TEST_SEED = 4242

#: Event configuration used by the small test grid.
TEST_EVENT = ExtremeEventConfig(
    enabled=True,
    name="test_monsoon_burst",
    center_latitude=13.0,
    center_longitude=77.0,
    radius_deg=0.5,
    start_index=4,
    duration_steps=6,
    peak_precipitation_mm_per_hour=55.0,
    temperature_anomaly_k=-4.0,
)


@pytest.fixture(scope="session")
def small_config() -> SyntheticConfig:
    """Small deterministic synthetic configuration (16 time steps, 0.25 deg)."""
    return SyntheticConfig(
        seed=TEST_SEED,
        time_steps=16,
        start_time="2024-06-01T00:00:00",
        time_step_hours=1,
        latitude_min=12.0,
        latitude_max=14.0,
        longitude_min=76.0,
        longitude_max=78.0,
        resolution_deg=0.25,
        event=TEST_EVENT,
    )


@pytest.fixture(scope="session")
def quiet_config() -> SyntheticConfig:
    """Same configuration as :func:`small_config` but without the event."""
    return SyntheticConfig(
        seed=TEST_SEED,
        time_steps=16,
        start_time="2024-06-01T00:00:00",
        time_step_hours=1,
        latitude_min=12.0,
        latitude_max=14.0,
        longitude_min=76.0,
        longitude_max=78.0,
        resolution_deg=0.25,
        event=ExtremeEventConfig(
            enabled=False,
            center_latitude=13.0,
            center_longitude=77.0,
            radius_deg=0.5,
            start_index=4,
            duration_steps=6,
        ),
    )


@pytest.fixture(scope="session")
def small_dataset(small_config: SyntheticConfig):
    """In-memory synthetic dataset *with* the injected extreme event."""
    return generate_synthetic_dataset(small_config, created_utc="2024-06-01T00:00:00Z")


@pytest.fixture(scope="session")
def quiet_dataset(quiet_config: SyntheticConfig):
    """In-memory synthetic dataset *without* the injected extreme event."""
    return generate_synthetic_dataset(quiet_config, created_utc="2024-06-01T00:00:00Z")


@pytest.fixture(scope="session")
def raw_generation(tmp_path_factory: pytest.TempPathFactory, small_config: SyntheticConfig) -> GenerationResult:
    """Raw NetCDF file + manifest written to a temporary directory."""
    output_dir = tmp_path_factory.mktemp("raw")
    return generate_and_save(
        output_dir / "sample_history.nc",
        config=small_config,
        created_utc="2024-06-01T00:00:00Z",
    )


@pytest.fixture(scope="session")
def processed_result(
    tmp_path_factory: pytest.TempPathFactory, raw_generation: GenerationResult
) -> PreprocessingResult:
    """Result of running the preprocessing pipeline over :func:`raw_generation`."""
    output_dir = tmp_path_factory.mktemp("processed")
    return preprocess_file(
        raw_generation.dataset_path,
        output_dir / "sample_history_processed.nc",
        max_gap=3,
    )


@pytest.fixture()
def api_client(tmp_path: Path, small_config: SyntheticConfig) -> Iterator[TestClient]:
    """FastAPI test client whose data directory is a temporary folder.

    The fixture writes one synthetic dataset (``raw/sample_history.nc``) first so
    the dataset endpoints have something real to inspect, then overrides the
    ``get_settings`` dependency so the application never touches the repository
    ``data/`` directory.
    """
    settings = Settings(data_dir=tmp_path, model_dir=tmp_path / "models")
    settings.ensure_directories()
    generate_and_save(settings.raw_dir / "sample_history.nc", config=small_config)

    application = create_app(settings)
    application.dependency_overrides[get_settings] = lambda: settings
    try:
        with TestClient(application) as client:
            yield client
    finally:
        application.dependency_overrides.clear()


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Isolated settings whose every directory lives under ``tmp_path``.

    Used by the Phase 8 near-real-time tests, which write run records and
    incremental state and must never touch the real ``data/`` directory.
    """
    isolated = Settings(data_dir=tmp_path / "data", model_dir=tmp_path / "models")
    isolated.ensure_directories()
    return isolated


@pytest.fixture()
def window() -> "WeatherRequest":
    """A six-hour window ending at a fixed reference instant (Phase 8)."""
    from tests.phase8_support import REFERENCE_NOW, DEFAULT_VARIABLES

    return WeatherRequest(
        start=REFERENCE_NOW - timedelta(hours=5),
        end=REFERENCE_NOW,
        variables=DEFAULT_VARIABLES,
        latitude=(12.0, 12.5, 13.0),
        longitude=(76.0, 76.5, 77.0),
    )


@pytest.fixture()
def pipeline_orchestrator(settings: Settings) -> Iterator[PipelineOrchestrator]:
    """An orchestrator on the fixture provider, with no real backoff sleeps."""
    from extreme_weather.realtime.orchestrator import PipelineOrchestrator
    from extreme_weather.realtime.providers import FixtureProvider
    from extreme_weather.realtime.retry import RetryPolicy

    yield PipelineOrchestrator(
        FixtureProvider(),
        settings=settings,
        retry_policy=RetryPolicy(max_attempts=3, base_delay=0.0),
        sleep=lambda _seconds: None,
    )


def package_version() -> str:
    """Version reported by the API/package metadata."""
    return PACKAGE_VERSION
