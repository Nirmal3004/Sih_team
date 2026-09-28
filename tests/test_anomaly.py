"""Tests for anomaly scores, thresholds, grid-cell detection and the event model.

Everything runs on synthetic data: the reference distributions are prototype
statistics and the thresholds are prototype values, not official warning criteria.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pytest
import xarray as xr

from extreme_weather.anomaly.climatology import build_climatology
from extreme_weather.anomaly.detector import DetectionError, detect_extreme_cells, detection_summary
from extreme_weather.anomaly.scores import (
    EFI_FORMULA,
    EFI_MAX,
    EFI_MIN,
    SCORE_FUNCTIONS,
    ScoreError,
    absolute_anomaly,
    compute_score,
    efi_score,
    extremeness_score,
    percentile_score,
    standardized_anomaly,
)
from extreme_weather.anomaly.thresholds import (
    PROTOTYPE_THRESHOLD_WARNING,
    ThresholdLadder,
    ThresholdSet,
    ThresholdsError,
    classify_score,
    classify_scores,
    default_thresholds,
)
from extreme_weather.schemas.event import (
    EventType,
    ExtremeWeatherEvent,
    Severity,
)

TIMES = np.arange(np.datetime64("2024-06-01"), np.datetime64("2024-06-11"), np.timedelta64(1, "D"))
LATITUDES = np.array([10.0, 10.5, 11.0])
LONGITUDES = np.array([70.0, 70.5, 71.0])


def make_field(values: np.ndarray, *, name: str = "precipitation", units: str = "mm h-1") -> xr.DataArray:
    """Wrap a ``(latitude, longitude)`` or ``(time, latitude, longitude)`` array."""
    array = np.asarray(values, dtype="float32")
    if array.ndim == 2:
        array = np.broadcast_to(array, (TIMES.size, *array.shape)).copy()
    return xr.DataArray(
        array,
        dims=("time", "latitude", "longitude"),
        coords={"time": TIMES, "latitude": LATITUDES, "longitude": LONGITUDES},
        name=name,
        attrs={"units": units},
    )


def reference_climatology(*, spread: float = 1.0, constant: bool = False) -> xr.Dataset:
    """Build a reference dataset (and climatology) with a known distribution."""
    rng = np.random.default_rng(7)
    days = TIMES.size
    base = 10.0 + np.linspace(0.0, 4.0, LATITUDES.size)[:, None] * np.ones((1, LONGITUDES.size))
    if constant:
        data = np.broadcast_to(base, (days, LATITUDES.size, LONGITUDES.size)).copy()
    else:
        noise = rng.normal(0.0, spread, size=(days, LATITUDES.size, LONGITUDES.size))
        data = base + noise
    dataset = xr.Dataset(
        {"precipitation": make_field(data)},
        coords={"time": TIMES, "latitude": LATITUDES, "longitude": LONGITUDES},
    )
    return build_climatology(dataset, "precipitation", "all")


# --------------------------------------------------------------------------- #
# A. Absolute anomaly
# --------------------------------------------------------------------------- #
def test_absolute_anomaly_is_value_minus_reference_mean() -> None:
    climatology = reference_climatology()
    mean = climatology["mean"].isel(group=0).values
    values = mean + 3.0

    anomaly = absolute_anomaly(make_field(values), climatology)

    assert np.allclose(anomaly.values, 3.0, atol=1e-4)
    assert anomaly.dims == ("time", "latitude", "longitude")
    assert anomaly.attrs["units"] == "mm h-1"
    assert "value - climatology_mean" in anomaly.attrs["formula"]


def test_absolute_anomaly_is_signed() -> None:
    climatology = reference_climatology()
    mean = climatology["mean"].isel(group=0).values

    above = absolute_anomaly(make_field(mean + 5.0), climatology)
    below = absolute_anomaly(make_field(mean - 5.0), climatology)

    assert np.all(above.values > 0.0)
    assert np.all(below.values < 0.0)
    assert np.allclose(above.values, -below.values, atol=1e-4)


def test_absolute_anomaly_propagates_nan() -> None:
    climatology = reference_climatology()
    mean = climatology["mean"].isel(group=0).values
    values = np.broadcast_to(mean, (TIMES.size, LATITUDES.size, LONGITUDES.size)).copy()
    values[0, 0, 0] = np.nan

    anomaly = absolute_anomaly(make_field(values), climatology)

    assert np.isnan(anomaly.values[0, 0, 0])
    assert np.isfinite(anomaly.values[1:]).all()


# --------------------------------------------------------------------------- #
# B. Standardized anomaly
# --------------------------------------------------------------------------- #
def test_standardized_anomaly_matches_numpy() -> None:
    climatology = reference_climatology()
    mean = climatology["mean"].isel(group=0).values
    std = climatology["std"].isel(group=0).values

    zscore = standardized_anomaly(make_field(mean + 2.0 * std), climatology)

    assert np.allclose(zscore.values, 2.0, atol=1e-3)
    assert zscore.attrs["units"] == "1"


def test_standardized_anomaly_handles_zero_variance() -> None:
    climatology = reference_climatology(constant=True)
    mean = climatology["mean"].isel(group=0).values
    assert float(climatology["std"].max()) == 0.0

    nan_result = standardized_anomaly(make_field(mean + 4.0), climatology)
    assert np.isnan(nan_result.values).all()

    zero_result = standardized_anomaly(
        make_field(mean + 4.0), climatology, zero_variance="zero"
    )
    assert np.all(zero_result.values == 0.0)


def test_standardized_anomaly_uses_the_min_std_floor() -> None:
    climatology = reference_climatology(spread=1e-9)
    mean = climatology["mean"].isel(group=0).values

    zscore = standardized_anomaly(make_field(mean), climatology, min_std=1e-6)

    assert np.isnan(zscore.values).all()


def test_standardized_anomaly_rejects_invalid_arguments() -> None:
    climatology = reference_climatology()
    values = make_field(np.zeros((TIMES.size, LATITUDES.size, LONGITUDES.size)))

    with pytest.raises(ScoreError, match="zero_variance"):
        standardized_anomaly(values, climatology, zero_variance="whatever")
    with pytest.raises(ScoreError, match="min_std"):
        standardized_anomaly(values, climatology, min_std=-1.0)


# --------------------------------------------------------------------------- #
# C. Percentile and extremeness
# --------------------------------------------------------------------------- #
def test_percentile_is_monotonic_and_bounded() -> None:
    climatology = reference_climatology()
    mean = climatology["mean"].isel(group=0).values
    std = climatology["std"].isel(group=0).values

    # Inside the reference spread the percentile must increase strictly.
    inside = [mean - 3.0 * std, mean - std, mean, mean + std, mean + 3.0 * std]
    percentiles = [
        float(percentile_score(make_field(level), climatology).values[0, 0, 0])
        for level in inside
    ]
    assert all(earlier < later for earlier, later in zip(percentiles, percentiles[1:]))
    assert percentiles[0] < 0.2
    assert percentiles[-1] > 0.8

    # Beyond the stored quantiles the score saturates instead of growing.
    saturated_low = float(percentile_score(make_field(mean - 50.0), climatology).values[0, 0, 0])
    saturated_high = float(percentile_score(make_field(mean + 50.0), climatology).values[0, 0, 0])
    assert saturated_low == pytest.approx(0.01, abs=0.02)
    assert saturated_high == pytest.approx(0.99, abs=0.02)


def test_extremeness_is_bounded_between_zero_and_one() -> None:
    climatology = reference_climatology()
    mean = climatology["mean"].isel(group=0).values

    low = extremeness_score(make_field(mean), climatology)
    high = extremeness_score(make_field(mean + 50.0), climatology)

    assert float(low.values.min()) >= 0.0
    assert float(high.values.max()) <= 1.0
    assert float(high.values.min()) > float(low.values.max())
    assert high.attrs["units"] == "1"


def test_percentile_requires_a_quantile_variable() -> None:
    climatology = reference_climatology().drop_vars("quantiles")
    with pytest.raises(ScoreError, match="no 'quantiles' variable"):
        percentile_score(make_field(np.zeros((TIMES.size, 3, 3))), climatology)


# --------------------------------------------------------------------------- #
# D. EFI-style score
# --------------------------------------------------------------------------- #
def test_efi_score_is_bounded_and_monotonic() -> None:
    climatology = reference_climatology()
    mean = climatology["mean"].isel(group=0).values
    shape = (TIMES.size, LATITUDES.size, LONGITUDES.size)

    typical = efi_score(make_field(mean), climatology)
    extreme = efi_score(make_field(mean + 100.0), climatology)

    assert float(typical.values.min()) >= EFI_MIN
    assert float(extreme.values.max()) <= EFI_MAX
    assert np.isfinite(typical.values).all()
    assert np.isfinite(extreme.values).all()
    assert float(extreme.values.min()) > float(typical.values.max())
    # The score is negative at/below the climatological percentile by construction.
    assert float(typical.values.max()) == pytest.approx(EFI_MIN, abs=1e-6)


def test_efi_score_documents_its_definition() -> None:
    climatology = reference_climatology()
    score = efi_score(make_field(np.zeros((TIMES.size, 3, 3))), climatology)

    assert "Phi^-1" in EFI_FORMULA
    assert "NOT" in EFI_FORMULA and "operational" in EFI_FORMULA
    assert score.attrs["formula"] == EFI_FORMULA
    assert score.attrs["climatology_percentile"] == pytest.approx(0.9)
    assert score.attrs["score_min"] == EFI_MIN
    assert score.attrs["score_max"] == EFI_MAX


def test_efi_score_saturates_at_the_climatological_percentile() -> None:
    climatology = reference_climatology()
    p90_value = climatology["quantiles"].sel(quantile=0.9, group=0)
    p99_value = climatology["quantiles"].sel(quantile=0.99, group=0)

    at_p90 = efi_score(make_field(p90_value.values), climatology, climatology_percentile=0.9)
    at_p99 = efi_score(make_field(p99_value.values), climatology, climatology_percentile=0.9)

    # At exactly p_clim the ratio is clipped to 0, so the score saturates at the
    # documented lower bound; above it the score becomes positive.
    assert float(at_p90.values.mean()) == pytest.approx(EFI_MIN, abs=1e-6)
    assert float(at_p99.values.mean()) > 0.0
    assert float(at_p99.values.mean()) <= EFI_MAX


def test_efi_score_custom_percentile_and_bounds() -> None:
    climatology = reference_climatology()
    mean = climatology["mean"].isel(group=0).values

    score = efi_score(
        make_field(mean + 100.0), climatology, climatology_percentile=0.5, bounds=(-1.0, 1.0)
    )

    assert float(score.values.max()) <= 1.0
    assert score.attrs["climatology_percentile"] == pytest.approx(0.5)

    with pytest.raises(ScoreError, match="climatology_percentile"):
        efi_score(make_field(mean), climatology, climatology_percentile=1.5)
    with pytest.raises(ScoreError, match="bounds must be increasing"):
        efi_score(make_field(mean), climatology, bounds=(2.0, 1.0))


def test_efi_score_handles_nan_inputs() -> None:
    climatology = reference_climatology()
    values = np.zeros((TIMES.size, LATITUDES.size, LONGITUDES.size))
    values[0, 0, 0] = np.nan

    score = efi_score(make_field(values), climatology)

    assert np.isnan(score.values[0, 0, 0])
    assert np.isfinite(score.values[1:]).all()


def test_score_registry_covers_every_documented_score() -> None:
    climatology = reference_climatology()
    mean = climatology["mean"].isel(group=0).values
    values = make_field(mean + 5.0)

    assert set(SCORE_FUNCTIONS) == {
        "absolute",
        "standardized",
        "percentile",
        "extremeness",
        "efi",
    }
    for name in SCORE_FUNCTIONS:
        score = compute_score(name, values, climatology)
        assert score.dims == values.dims
        assert score.shape == values.shape

    with pytest.raises(ScoreError, match="Unknown anomaly score"):
        compute_score("vibes", values, climatology)


# --------------------------------------------------------------------------- #
# Thresholds
# --------------------------------------------------------------------------- #
def make_ladder(steps: list[tuple[float, str]]) -> ThresholdLadder:
    """Build a small ladder for boundary tests."""
    return ThresholdLadder(
        name="test_ladder",
        event_type=EventType.EXTREME_RAINFALL,
        variable="precipitation",
        score_name="absolute",
        steps=steps,
    )


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (-5.0, Severity.NORMAL),
        (0.0, Severity.NORMAL),
        (9.99, Severity.NORMAL),
        (10.0, Severity.WATCH),  # boundary: equal to the threshold
        (10.01, Severity.WATCH),
        (25.0, Severity.MODERATE),
        (40.0, Severity.SEVERE),
        (49.99, Severity.SEVERE),
        (50.0, Severity.EXTREME),
        (1000.0, Severity.EXTREME),
    ],
)
def test_threshold_boundaries(score: float, expected: Severity) -> None:
    ladder = make_ladder(
        [(10.0, "WATCH"), (25.0, "MODERATE"), (40.0, "SEVERE"), (50.0, "EXTREME")]
    )
    assert classify_score(score, ladder) is expected
    assert ladder.classify(score) is expected


def test_nan_score_is_normal() -> None:
    ladder = make_ladder([(1.0, "WATCH"), (2.0, "EXTREME")])
    assert ladder.classify(float("nan")) is Severity.NORMAL


def test_classify_scores_matches_the_scalar_classifier() -> None:
    ladder = make_ladder([(10.0, "WATCH"), (25.0, "MODERATE"), (40.0, "SEVERE"), (50.0, "EXTREME")])
    # 39.9 would still be MODERATE (SEVERE starts at 40), so use 45.0 here.
    grid = np.array([[-1.0, 9.99, 10.0], [25.0, 45.0, 50.0], [120.0, np.nan, 0.0]])
    scores = score_field(grid)

    classified = classify_scores(scores, ladder)
    flat = np.asarray(classified.values)[0].reshape(-1)

    assert [Severity(item) for item in flat] == [
        Severity.NORMAL,
        Severity.NORMAL,
        Severity.WATCH,
        Severity.MODERATE,
        Severity.SEVERE,
        Severity.EXTREME,
        Severity.EXTREME,
        Severity.NORMAL,
        Severity.NORMAL,
    ]
    assert classified.dims == scores.dims
    assert classified.attrs["ladder"] == "test_ladder"


def test_ladder_validation_rejects_broken_configurations() -> None:
    with pytest.raises(ValueError, match="no steps"):
        make_ladder([])
    with pytest.raises(ValueError, match="strictly increasing"):
        make_ladder([(10.0, "WATCH"), (5.0, "MODERATE"), (50.0, "EXTREME")])
    with pytest.raises(ValueError, match="must not contain NORMAL"):
        make_ladder([(10.0, "NORMAL"), (50.0, "EXTREME")])
    with pytest.raises(ValueError, match="must be EXTREME"):
        make_ladder([(10.0, "WATCH"), (50.0, "SEVERE")])


def test_default_ladders_are_prototype_values_with_a_disclaimer() -> None:
    thresholds = default_thresholds()

    assert thresholds.for_variable("precipitation").event_type is EventType.EXTREME_RAINFALL
    assert thresholds.for_variable("wind_speed").event_type is EventType.EXTREME_WIND
    assert thresholds.for_variable("temperature").event_type is EventType.EXTREME_TEMPERATURE
    assert thresholds.for_variable("unknown") is None
    for ladder in thresholds.ladders:
        assert "NOT official warning thresholds" in ladder.note
        assert ladder.note == PROTOTYPE_THRESHOLD_WARNING

    # A supported type without a ladder in the set is a configuration error.
    with pytest.raises(ThresholdsError, match="No threshold ladder"):
        default_thresholds([EventType.EXTREME_RAINFALL]).for_event_type(EventType.EXTREME_WIND)
    # An unknown type is rejected by the enum itself.
    with pytest.raises(ValueError, match="Unknown event type"):
        thresholds.for_event_type("EXTREME_HAIL")


def test_threshold_set_can_be_restricted_to_event_types() -> None:
    thresholds = default_thresholds([EventType.EXTREME_RAINFALL])

    assert len(thresholds.ladders) == 1
    with pytest.raises(ThresholdsError, match="No threshold ladder"):
        thresholds.for_event_type(EventType.EXTREME_WIND)


def test_custom_ladder_is_accepted_from_mapping_input() -> None:
    ladder = ThresholdLadder(
        name="mapping_ladder",
        event_type=EventType.EXTREME_WIND,
        variable="wind_speed",
        steps={2.0: "watch", 6.0: "severe", 9.0: "extreme"},
    )

    assert ladder.thresholds == (2.0, 6.0, 9.0)
    assert ladder.classify(9.5) is Severity.EXTREME
    assert ThresholdSet(ladders=(ladder,)).for_variable("wind_speed") is ladder


# --------------------------------------------------------------------------- #
# Grid-cell detection
# --------------------------------------------------------------------------- #
def score_field(values: np.ndarray) -> xr.DataArray:
    """Build a score-like field (same helper, different name for readability)."""
    return make_field(values, name="score", units="1")


def test_detection_returns_nothing_without_anomalies() -> None:
    ladder = make_ladder([(10.0, "WATCH"), (50.0, "EXTREME")])
    scores = score_field(np.zeros((TIMES.size, LATITUDES.size, LONGITUDES.size)))

    mask = detect_extreme_cells(scores, ladder, event_type=EventType.EXTREME_RAINFALL)

    assert mask.dtype == bool
    assert mask.dims == scores.dims
    assert not mask.values.any()
    assert mask.attrs["event_type"] == "EXTREME_RAINFALL"
    assert mask.attrs["threshold"] == 10.0


def test_detection_marks_single_and_multiple_cells() -> None:
    ladder = make_ladder([(10.0, "WATCH"), (50.0, "EXTREME")])
    values = np.zeros((TIMES.size, LATITUDES.size, LONGITUDES.size))
    values[2, 1, 1] = 12.0
    values[2, 1, 2] = 30.0
    values[5, 0, 0] = 60.0

    mask = detect_extreme_cells(score_field(values), ladder)

    assert mask.values[2, 1, 1]
    assert mask.values[2, 1, 2]
    assert mask.values[5, 0, 0]
    assert int(mask.values.sum()) == 3
    # Coordinates must survive detection unchanged.
    assert np.array_equal(mask["latitude"].values, LATITUDES)
    assert np.array_equal(mask["longitude"].values, LONGITUDES)


def test_detection_ignores_nan_cells() -> None:
    ladder = make_ladder([(10.0, "WATCH"), (50.0, "EXTREME")])
    values = np.full((TIMES.size, LATITUDES.size, LONGITUDES.size), np.nan)
    values[0, 0, 0] = 100.0

    mask = detect_extreme_cells(score_field(values), ladder)

    assert int(mask.values.sum()) == 1
    assert bool(mask.values[0, 0, 0])


def test_detection_preserves_name_and_attributes() -> None:
    ladder = make_ladder([(10.0, "WATCH"), (50.0, "EXTREME")])
    scores = score_field(np.zeros((TIMES.size, LATITUDES.size, LONGITUDES.size)))

    mask = detect_extreme_cells(scores, ladder, name="precipitation_extreme_mask")

    assert mask.name == "precipitation_extreme_mask"
    assert list(mask.coords) == list(scores.coords)
    assert np.array_equal(mask["time"].values, scores["time"].values)
    assert "flag_meanings" in mask.attrs


def test_detection_rejects_mismatched_event_type() -> None:
    ladder = make_ladder([(10.0, "WATCH"), (50.0, "EXTREME")])
    scores = score_field(np.zeros((TIMES.size, LATITUDES.size, LONGITUDES.size)))

    with pytest.raises(DetectionError, match="does not match the ladder"):
        detect_extreme_cells(scores, ladder, event_type=EventType.EXTREME_WIND)

    with pytest.raises(DetectionError, match="at least one step"):
        detect_extreme_cells(scores, object())


def test_detection_summary_counts_cells_per_time_step() -> None:
    ladder = make_ladder([(10.0, "WATCH"), (50.0, "EXTREME")])
    values = np.zeros((TIMES.size, LATITUDES.size, LONGITUDES.size))
    values[0, 0, 0] = 12.0
    values[1, 0, 0] = 12.0
    values[1, 0, 1] = 12.0
    scores = score_field(values)

    mask = detect_extreme_cells(scores, ladder)
    summary = detection_summary(mask, classify_scores(scores, ladder))

    assert summary["total"] == 3
    assert summary["2024-06-01T00:00:00"] == 1
    assert summary["2024-06-02T00:00:00"] == 2
    assert summary["severity_WATCH"] == 3
    assert summary["severity_EXTREME"] == 0


# --------------------------------------------------------------------------- #
# Event model
# --------------------------------------------------------------------------- #
def valid_event(**overrides) -> ExtremeWeatherEvent:
    """Return a valid event, optionally with field overrides."""
    payload = {
        "event_id": "EVT-EXTREME_RAINFALL-20240601T0000-001",
        "event_type": EventType.EXTREME_RAINFALL,
        "timestamp": datetime(2024, 6, 1, tzinfo=timezone.utc),
        "latitude": 13.0,
        "longitude": 77.0,
        "intensity": 55.0,
        "anomaly_score": 54.0,
        "severity": Severity.EXTREME,
        "affected_radius_km": 42.5,
        "cell_count": 12,
        "variable": "precipitation",
        "units": "mm h-1",
        "max_value": 55.0,
        "mean_value": 30.0,
    }
    payload.update(overrides)
    return ExtremeWeatherEvent(**payload)


def test_valid_event_round_trips_through_json() -> None:
    event = valid_event()
    payload = event.model_dump(mode="json")

    assert payload["event_type"] == "EXTREME_RAINFALL"
    assert payload["severity"] == "EXTREME"
    assert payload["timestamp"].startswith("2024-06-01T00:00:00")
    assert ExtremeWeatherEvent.model_validate(payload) == event


def test_event_normalises_naive_timestamps_to_utc() -> None:
    event = valid_event(timestamp=datetime(2024, 6, 1))
    assert event.timestamp.tzinfo is not None
    assert event.timestamp.utcoffset().total_seconds() == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"latitude": 95.0}, "latitude"),
        ({"latitude": -120.0}, "latitude"),
        ({"longitude": 181.0}, "longitude"),
        ({"longitude": -200.0}, "longitude"),
        ({"affected_radius_km": -0.1}, "affected_radius_km"),
        ({"cell_count": 0}, "cell_count"),
        ({"event_id": ""}, "event_id"),
        ({"event_type": "EXTREME_HAIL"}, "event_type"),
        ({"severity": "CATASTROPHIC"}, "severity"),
    ],
)
def test_event_validation_rejects_invalid_fields(overrides: dict, message: str) -> None:
    with pytest.raises(ValueError) as error:
        valid_event(**overrides)
    assert message in str(error.value)


def test_event_accepts_boundary_coordinates() -> None:
    assert valid_event(latitude=90.0, longitude=180.0, affected_radius_km=0.0, cell_count=1)
    assert valid_event(latitude=-90.0, longitude=-180.0)


def test_severity_and_event_type_parsing() -> None:
    assert Severity.from_value(" extreme ") is Severity.EXTREME
    assert EventType.from_value("extreme_wind") is EventType.EXTREME_WIND
    assert Severity.SEVERE.rank > Severity.WATCH.rank
    assert Severity.EXTREME.is_extreme
    assert not Severity.NORMAL.is_extreme

    with pytest.raises(ValueError, match="Unknown severity"):
        Severity.from_value("nope")
    with pytest.raises(ValueError, match="Unknown event type"):
        EventType.from_value("tornado")


# --------------------------------------------------------------------------- #
# Integration: the Phase 0 synthetic extreme event must be detected
# --------------------------------------------------------------------------- #
def test_injected_synthetic_event_is_detected(small_dataset, quiet_dataset, small_config) -> None:
    """Background -> injected event -> high anomaly -> detected cells -> event.

    The climatology is built from the *event-free* synthetic run, so the injected
    event is a genuine anomaly with respect to the reference.
    """
    from extreme_weather.anomaly.clustering import cluster_extreme_events

    event = small_config.event
    climatology = build_climatology(quiet_dataset, "precipitation", "month")
    scores = compute_score("absolute", small_dataset["precipitation"], climatology)
    ladder = default_thresholds().for_variable("precipitation")
    mask = detect_extreme_cells(scores, ladder, event_type=ladder.event_type)
    events = cluster_extreme_events(
        mask,
        scores,
        ladder=ladder,
        event_type=ladder.event_type,
        values=small_dataset["precipitation"],
        variable="precipitation",
        units="mm h-1",
    )

    # 1. The background alone produces no detection.
    quiet_scores = compute_score("absolute", quiet_dataset["precipitation"], climatology)
    quiet_mask = detect_extreme_cells(quiet_scores, ladder, event_type=ladder.event_type)
    assert int(quiet_mask.values.sum()) == 0

    # 2. The injected event produces a high anomaly score near its centre.
    window = slice(event.start_index, event.start_index + event.duration_steps)
    assert float(scores.values[window].max()) > 50.0 - float(climatology["mean"].max())

    # 3. Detected cells only exist inside the injection window.
    outside = np.ones(small_dataset.sizes["time"], dtype=bool)
    outside[window] = False
    assert int(mask.values[outside].sum()) == 0
    assert int(mask.values[window].sum()) > 0

    # 4. The clustered event is located at the injected centre (tolerant of the
    #    centroid being a grid-point average rather than the exact centre).
    assert events, "the injected extreme rainfall event must be detected"
    strongest = max(events, key=lambda item: item.anomaly_score)
    assert abs(strongest.latitude - event.center_latitude) <= 0.5
    assert abs(strongest.longitude - event.center_longitude) <= 0.5
    assert strongest.event_type.value == "EXTREME_RAINFALL"
    assert strongest.cell_count >= 1
    assert strongest.intensity > 50.0
    assert strongest.severity.value in {"MODERATE", "SEVERE", "EXTREME"}
    assert strongest.affected_radius_km > 0.0
    assert all(event_item.variable == "precipitation" for event_item in events)


def test_service_detects_the_injected_event_and_stores_the_result(
    tmp_path, small_dataset, quiet_dataset
) -> None:
    from extreme_weather.anomaly.service import AnomalyDetectionService, EventStore

    store = EventStore(tmp_path / "events.json")
    service = AnomalyDetectionService(store=store)

    result = service.analyse(
        small_dataset,
        reference=quiet_dataset,
        variables=("precipitation",),
        source_path="raw/sample_history.nc",
        write_outputs=True,
        keep_fields=True,
    )

    assert result.event_count > 0
    assert result.variables == ("precipitation",)
    assert result.score_name == "absolute"
    assert "operational forecast" in result.metadata["data_disclaimer"]
    assert store.path.is_file()
    assert (store.path.parent / "anomaly.nc").is_file()

    reloaded = store.load()
    assert reloaded is not None
    assert reloaded.event_count == result.event_count
    assert reloaded.events[0].event_id == result.events[0].event_id


# --------------------------------------------------------------------------- #
# Visualisation (optional diagnostic aid)
# --------------------------------------------------------------------------- #
def test_visualization_writes_an_overview_figure(tmp_path, small_dataset, quiet_dataset) -> None:
    from extreme_weather.anomaly.clustering import cluster_extreme_events
    from extreme_weather.anomaly.visualization import plot_analysis_overview

    climatology = build_climatology(quiet_dataset, "precipitation", "month")
    values = small_dataset["precipitation"]
    scores = compute_score("absolute", values, climatology)
    ladder = default_thresholds().for_variable("precipitation")
    mask = detect_extreme_cells(scores, ladder, event_type=ladder.event_type)
    events = cluster_extreme_events(
        mask, scores, ladder=ladder, event_type=ladder.event_type, values=values
    )

    figure = plot_analysis_overview(
        values, scores, mask, events, output_path=tmp_path / "overview.png"
    )

    assert figure is not None
    assert figure.is_file()
    assert figure.stat().st_size > 0


def test_visualization_rejects_mismatched_fields() -> None:
    from extreme_weather.anomaly.visualization import VisualizationError, plot_analysis_overview

    values = make_field(np.zeros((TIMES.size, LATITUDES.size, LONGITUDES.size)))
    mask = xr.DataArray(
        np.zeros((TIMES.size, 2, 2), dtype=bool),
        dims=("time", "latitude", "longitude"),
        name="mask",
    )

    with pytest.raises(VisualizationError, match="must match"):
        plot_analysis_overview(values, values, mask)
