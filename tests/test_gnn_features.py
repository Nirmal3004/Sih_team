"""Node and edge feature tests (SIH26078, Phase 3)."""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

import pytest

from extreme_weather.gnn.features import (
    EDGE_FEATURE_NAMES,
    EDGE_SPATIAL,
    EDGE_TEMPORAL,
    EVENT_TYPE_ORDER,
    NODE_FEATURE_NAMES,
    TARGET_NAMES,
    FeatureConfig,
    edge_feature_vector,
    event_type_columns,
    node_feature_vector,
    target_vector,
)
from extreme_weather.schemas.event import EventType, Severity
from extreme_weather.schemas.track import TrackPoint

BASE_TIME = datetime(2024, 6, 2, 2, 0, tzinfo=timezone.utc)


def make_point(**overrides) -> TrackPoint:
    """Build a valid track point, overriding the fields a test needs."""
    payload = {
        "event_id": "EVT-1",
        "timestamp": BASE_TIME,
        "latitude": 13.0,
        "longitude": 77.0,
        "intensity": 50.0,
        "anomaly_score": 0.8,
        "severity": Severity.SEVERE,
        "speed_kmh": 10.0,
        "bearing_degrees": 90.0,
        "direction": "E",
        "cell_count": 12,
    }
    payload.update(overrides)
    return TrackPoint(**payload)


# --- dimensions -------------------------------------------------------------


def test_node_feature_vector_has_the_documented_dimension():
    assert len(node_feature_vector(make_point(), EventType.EXTREME_RAINFALL)) == len(
        NODE_FEATURE_NAMES
    )


def test_edge_feature_vector_has_the_documented_dimension():
    target = make_point(timestamp=BASE_TIME + timedelta(hours=1))
    vector = edge_feature_vector(make_point(), target, EDGE_TEMPORAL)
    assert len(vector) == len(EDGE_FEATURE_NAMES)


def test_feature_names_are_unique():
    assert len(set(NODE_FEATURE_NAMES)) == len(NODE_FEATURE_NAMES)
    assert len(set(EDGE_FEATURE_NAMES)) == len(EDGE_FEATURE_NAMES)
    assert len(set(TARGET_NAMES)) == len(TARGET_NAMES)


def test_all_node_features_are_finite():
    vector = node_feature_vector(make_point(), EventType.EXTREME_WIND)
    assert all(math.isfinite(value) for value in vector)


# --- normalisation ----------------------------------------------------------


def test_coordinates_do_not_dominate_the_feature_space():
    """Raw degrees are present, but the scaled copies are O(1)."""
    vector = node_feature_vector(
        make_point(latitude=13.0, longitude=77.0), EventType.EXTREME_RAINFALL
    )
    assert vector[NODE_FEATURE_NAMES.index("latitude_scaled")] == pytest.approx(13.0 / 90.0)
    assert vector[NODE_FEATURE_NAMES.index("longitude_scaled")] == pytest.approx(77.0 / 180.0)
    assert abs(vector[NODE_FEATURE_NAMES.index("latitude_scaled")]) <= 1.0
    assert abs(vector[NODE_FEATURE_NAMES.index("longitude_scaled")]) <= 1.0


def test_intensity_is_scaled_by_the_configured_divisor():
    point = make_point(intensity=55.0)
    index = NODE_FEATURE_NAMES.index("intensity_scaled")
    default = node_feature_vector(point, EventType.EXTREME_RAINFALL)
    custom = node_feature_vector(
        point, EventType.EXTREME_RAINFALL, FeatureConfig(intensity_scale=10.0)
    )
    assert default[index] == pytest.approx(0.55)
    assert custom[index] == pytest.approx(5.5)


def test_speed_is_scaled_by_the_configured_divisor():
    index = NODE_FEATURE_NAMES.index("speed_scaled")
    fast = node_feature_vector(make_point(speed_kmh=60.0), EventType.EXTREME_RAINFALL)
    assert fast[index] == pytest.approx(0.5)  # 60 / 120
    custom = node_feature_vector(
        make_point(speed_kmh=60.0),
        EventType.EXTREME_RAINFALL,
        FeatureConfig(speed_scale=60.0),
    )
    assert custom[index] == pytest.approx(1.0)


def test_cell_count_is_log_scaled_and_capped():
    index = NODE_FEATURE_NAMES.index("cell_count_scaled")
    small = node_feature_vector(make_point(cell_count=9), EventType.EXTREME_RAINFALL)
    large = node_feature_vector(make_point(cell_count=100000), EventType.EXTREME_RAINFALL)
    assert 0.0 < small[index] < 1.0
    assert large[index] == 1.0


def test_missing_cell_count_maps_to_zero():
    index = NODE_FEATURE_NAMES.index("cell_count_scaled")
    vector = node_feature_vector(make_point(cell_count=None), EventType.EXTREME_RAINFALL)
    assert vector[index] == 0.0


# --- angle encodings --------------------------------------------------------


def test_bearing_is_encoded_as_sin_cos():
    sin_index = NODE_FEATURE_NAMES.index("bearing_sin")
    cos_index = NODE_FEATURE_NAMES.index("bearing_cos")
    north = node_feature_vector(
        make_point(bearing_degrees=0.0, direction="N"), EventType.EXTREME_RAINFALL
    )
    east = node_feature_vector(
        make_point(bearing_degrees=90.0, direction="E"), EventType.EXTREME_RAINFALL
    )
    assert north[sin_index] == pytest.approx(0.0, abs=1e-12)
    assert north[cos_index] == pytest.approx(1.0)
    assert east[sin_index] == pytest.approx(1.0)
    assert east[cos_index] == pytest.approx(0.0, abs=1e-12)


def test_undefined_bearing_maps_to_the_neutral_direction():
    """A CALM point has no direction: (sin, cos) = (0, 1)."""
    vector = node_feature_vector(
        make_point(bearing_degrees=None, speed_kmh=0.0, direction="CALM"),
        EventType.EXTREME_RAINFALL,
    )
    assert vector[NODE_FEATURE_NAMES.index("bearing_sin")] == 0.0
    assert vector[NODE_FEATURE_NAMES.index("bearing_cos")] == 1.0
    assert vector[NODE_FEATURE_NAMES.index("is_stationary")] == 1.0


def test_bearing_encoding_is_continuous_across_the_wrap():
    """359 deg and 1 deg are neighbours, not opposite ends of the range."""
    just_below = node_feature_vector(
        make_point(bearing_degrees=359.0, direction="N"), EventType.EXTREME_RAINFALL
    )
    just_above = node_feature_vector(
        make_point(bearing_degrees=1.0, direction="N"), EventType.EXTREME_RAINFALL
    )
    for name in ("bearing_sin", "bearing_cos"):
        index = NODE_FEATURE_NAMES.index(name)
        assert just_below[index] == pytest.approx(just_above[index], abs=0.04)


def test_hour_encoding_is_continuous_across_midnight():
    before = node_feature_vector(
        make_point(timestamp=BASE_TIME.replace(hour=23, minute=59)), EventType.EXTREME_RAINFALL
    )
    after = node_feature_vector(
        make_point(timestamp=BASE_TIME.replace(hour=0, minute=1)), EventType.EXTREME_RAINFALL
    )
    for name in ("hour_sin", "hour_cos"):
        index = NODE_FEATURE_NAMES.index(name)
        assert before[index] == pytest.approx(after[index], abs=0.02)


def test_hour_angles_stay_on_the_unit_circle():
    vector = node_feature_vector(make_point(timestamp=BASE_TIME), EventType.EXTREME_RAINFALL)
    sin_hour = vector[NODE_FEATURE_NAMES.index("hour_sin")]
    cos_hour = vector[NODE_FEATURE_NAMES.index("hour_cos")]
    assert math.hypot(sin_hour, cos_hour) == pytest.approx(1.0)


# --- event type encoding ----------------------------------------------------


def test_event_type_encoding_is_one_hot():
    for position, event_type in enumerate(EVENT_TYPE_ORDER):
        columns = event_type_columns(event_type)
        assert len(columns) == len(EVENT_TYPE_ORDER)
        assert sum(columns) == 1.0
        assert columns[position] == 1.0


def test_event_type_columns_reach_the_node_vector():
    vector = node_feature_vector(make_point(), EventType.EXTREME_WIND)
    assert vector[NODE_FEATURE_NAMES.index("is_extreme_wind")] == 1.0
    assert vector[NODE_FEATURE_NAMES.index("is_extreme_rainfall")] == 0.0


def test_unknown_event_type_is_not_mapped_onto_a_real_type():
    assert event_type_columns("NOT_A_TYPE") == [0.0, 0.0, 0.0]
    assert event_type_columns(None) == [0.0, 0.0, 0.0]


# --- edge features ----------------------------------------------------------


def test_temporal_edge_flags_the_edge_kind():
    target = make_point(timestamp=BASE_TIME + timedelta(hours=1))
    vector = edge_feature_vector(make_point(), target, EDGE_TEMPORAL)
    assert vector[EDGE_FEATURE_NAMES.index("is_temporal")] == 1.0
    assert vector[EDGE_FEATURE_NAMES.index("is_spatial")] == 0.0


def test_spatial_edge_flags_the_edge_kind():
    target = make_point(timestamp=BASE_TIME, longitude=77.1)
    vector = edge_feature_vector(make_point(), target, EDGE_SPATIAL)
    assert vector[EDGE_FEATURE_NAMES.index("is_spatial")] == 1.0
    assert vector[EDGE_FEATURE_NAMES.index("is_temporal")] == 0.0


def test_edge_distance_and_time_delta_are_scaled():
    config = FeatureConfig(edge_distance_scale=100.0, edge_time_scale=2.0)
    # ~11 km apart, 2 h apart -> 0.11 and 1.0 after scaling.
    target = make_point(timestamp=BASE_TIME + timedelta(hours=2), latitude=13.1)
    vector = edge_feature_vector(make_point(), target, EDGE_TEMPORAL, config)
    assert vector[EDGE_FEATURE_NAMES.index("distance_scaled")] == pytest.approx(0.111, abs=0.005)
    assert vector[EDGE_FEATURE_NAMES.index("time_delta_scaled")] == pytest.approx(1.0)


def test_unknown_edge_kind_is_rejected():
    with pytest.raises(ValueError, match="Unknown edge kind"):
        edge_feature_vector(make_point(), make_point(), "diagonal")


def test_edge_features_are_finite_for_a_stationary_pair():
    vector = edge_feature_vector(make_point(), make_point(timestamp=BASE_TIME), EDGE_SPATIAL)
    assert all(math.isfinite(value) for value in vector)


# --- target -----------------------------------------------------------------


def test_target_vector_is_the_delta_to_the_next_state():
    current = make_point(latitude=13.0, longitude=77.0, intensity=50.0)
    following = make_point(latitude=13.1, longitude=77.2, intensity=45.0)
    assert target_vector(current, following) == pytest.approx([0.1, 0.2, -5.0])


def test_target_vector_is_zero_for_a_stationary_step():
    point = make_point()
    assert target_vector(point, point) == pytest.approx([0.0, 0.0, 0.0])


# --- config validation ------------------------------------------------------


@pytest.mark.parametrize(
    "field", ["intensity_scale", "speed_scale", "cell_count_scale", "edge_distance_scale"]
)
def test_feature_config_rejects_non_positive_divisors(field):
    with pytest.raises(ValueError, match=field):
        FeatureConfig(**{field: 0.0})


def test_features_never_read_the_future_target():
    """A point's features do not depend on whatever comes after it."""
    point = make_point()
    following = make_point(
        timestamp=BASE_TIME + timedelta(hours=1), intensity=99.0, latitude=20.0
    )
    before = node_feature_vector(point, EventType.EXTREME_RAINFALL)
    # Recomputing the same point must give the same vector, whatever follows it.
    assert node_feature_vector(point, EventType.EXTREME_RAINFALL) == before
    # Sanity: the following point really does differ, so the test is meaningful.
    assert node_feature_vector(following, EventType.EXTREME_RAINFALL) != before
