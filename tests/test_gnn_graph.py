"""Graph construction tests (SIH26078, Phase 3)."""

from __future__ import annotations

import pytest

from extreme_weather.gnn.features import EDGE_FEATURE_NAMES, EDGE_SPATIAL, EDGE_TEMPORAL
from extreme_weather.gnn.graph import GraphConfig, build_graph, sort_tracks_for_graph

from .gnn_fixtures import build_track


def test_each_track_point_becomes_one_node(single_track):
    batch = build_graph([single_track])
    assert batch.node_count == len(single_track.points)
    assert len(batch.node_features) == batch.node_count


def test_only_the_last_point_of_a_track_has_no_target(single_track):
    batch = build_graph([single_track])
    assert batch.targets[-1] is None
    assert all(value is not None for value in batch.targets[:-1])
    assert batch.target_count == len(single_track.points) - 1


def test_targets_are_the_deltas_to_the_next_point(single_track):
    batch = build_graph([single_track])
    first, second = single_track.points[0], single_track.points[1]
    assert batch.targets[0] == pytest.approx(
        [
            second.latitude - first.latitude,
            second.longitude - first.longitude,
            second.intensity - first.intensity,
        ]
    )


def test_temporal_edges_connect_consecutive_points(single_track):
    batch = build_graph([single_track])
    count = len(single_track.points)
    assert batch.temporal_edge_count() == count - 1
    assert batch.edge_index[0] == list(range(count - 1))
    assert batch.edge_index[1] == list(range(1, count))
    assert set(batch.edge_kinds) == {EDGE_TEMPORAL}


def test_spatial_edge_joins_concurrent_close_tracks(two_nearby_tracks):
    batch = build_graph(two_nearby_tracks, GraphConfig(max_spatial_distance_km=50.0))
    assert batch.spatial_edge_count() == 3  # one per concurrent pair per timestamp
    assert batch.temporal_edge_count() == 4  # two per track


def test_no_spatial_edge_between_distant_tracks(two_far_tracks):
    batch = build_graph(two_far_tracks, GraphConfig(max_spatial_distance_km=150.0))
    assert batch.spatial_edge_count() == 0
    assert batch.temporal_edge_count() == 4


def test_spatial_distance_threshold_is_respected(two_far_tracks):
    """The same pair is linked at a large threshold and not at a small one."""
    loose = build_graph(two_far_tracks, GraphConfig(max_spatial_distance_km=2000.0))
    tight = build_graph(two_far_tracks, GraphConfig(max_spatial_distance_km=100.0))
    assert loose.spatial_edge_count() == 3
    assert tight.spatial_edge_count() == 0


def test_spatial_edges_can_be_disabled_entirely(two_nearby_tracks):
    batch = build_graph(two_nearby_tracks, GraphConfig(include_spatial_edges=False))
    assert batch.spatial_edge_count() == 0
    assert batch.temporal_edge_count() == 4


def test_zero_threshold_disables_spatial_edges(two_nearby_tracks):
    batch = build_graph(two_nearby_tracks, GraphConfig(max_spatial_distance_km=0.0))
    assert batch.spatial_edge_count() == 0


def test_no_duplicate_edges_are_created(two_nearby_tracks):
    """Same-track pairs are already temporal; a second edge would double-count."""
    batch = build_graph(two_nearby_tracks, GraphConfig(max_spatial_distance_km=2000.0))
    pairs = list(zip(*batch.edge_index))
    assert len(pairs) == len(set(pairs))
    assert all(source != target for source, target in pairs)


def test_every_edge_lands_in_the_node_range(two_nearby_tracks):
    batch = build_graph(two_nearby_tracks, GraphConfig(max_spatial_distance_km=2000.0))
    for source, target in zip(*batch.edge_index):
        assert 0 <= source < batch.node_count
        assert 0 <= target < batch.node_count


def test_node_and_edge_feature_widths_match_the_contract(single_track):
    batch = build_graph([single_track])
    assert batch.node_dim == 16
    assert batch.edge_dim == 6
    assert all(len(features) == batch.node_dim for features in batch.node_features)
    assert all(len(features) == batch.edge_dim for features in batch.edge_features)


def test_graph_construction_is_deterministic(single_track, two_nearby_tracks):
    first = build_graph([single_track], GraphConfig(max_spatial_distance_km=500.0))
    second = build_graph([single_track], GraphConfig(max_spatial_distance_km=500.0))
    assert first.node_features == second.node_features
    assert first.edge_features == second.edge_features
    assert first.edge_index == second.edge_index

    nearby_first = build_graph(two_nearby_tracks, GraphConfig(max_spatial_distance_km=500.0))
    nearby_second = build_graph(two_nearby_tracks, GraphConfig(max_spatial_distance_km=500.0))
    assert nearby_first.edge_index == nearby_second.edge_index


def test_input_order_does_not_change_the_graph(two_nearby_tracks):
    forwards = build_graph(two_nearby_tracks, GraphConfig(max_spatial_distance_km=500.0))
    backwards = build_graph(
        list(reversed(two_nearby_tracks)), GraphConfig(max_spatial_distance_km=500.0)
    )
    assert forwards.node_features == backwards.node_features
    assert forwards.edge_index == backwards.edge_index
    assert forwards.node_track_ids == backwards.node_track_ids


def test_tracks_are_sorted_canonically():
    later = build_track("TRK-B", [(13.0, 77.0, 10.0), (13.1, 77.1, 11.0)])
    earlier = build_track("TRK-A", [(13.0, 77.0, 10.0), (13.1, 77.1, 11.0)])
    ordered = sort_tracks_for_graph([later, earlier])
    assert [track.track_id for track in ordered] == ["TRK-A", "TRK-B"]


def test_empty_input_produces_an_empty_graph():
    batch = build_graph([])
    assert batch.node_count == 0
    assert batch.edge_count == 0
    assert batch.target_count == 0
    assert batch.node_dim == 0


def test_single_point_track_has_no_edges_and_no_target():
    batch = build_graph([build_track("TRK-SINGLE", [(13.0, 77.0, 10.0)])])
    assert batch.node_count == 1
    assert batch.edge_count == 0
    assert batch.targets == [None]


def test_duplicate_track_ids_are_rejected(single_track):
    with pytest.raises(ValueError, match="Duplicate track id"):
        build_graph([single_track, single_track])


def test_negative_spatial_threshold_is_rejected():
    with pytest.raises(ValueError, match="max_spatial_distance_km"):
        GraphConfig(max_spatial_distance_km=-1.0)


def test_incoming_degree_counts_edges(two_nearby_tracks):
    batch = build_graph(two_nearby_tracks, GraphConfig(max_spatial_distance_km=2000.0))
    degree = batch.incoming_degree()
    assert len(degree) == batch.node_count
    assert sum(degree) == batch.edge_count
    # Nodes 0-2 are track A, nodes 3-5 track B. Temporal edges give nodes 1,2,4,5
    # one in-edge; the spatial edges add one more to nodes 3,4,5.
    assert degree == [0, 1, 1, 1, 2, 2]


def test_edge_kinds_agree_with_the_feature_flags(two_nearby_tracks):
    batch = build_graph(two_nearby_tracks, GraphConfig(max_spatial_distance_km=2000.0))
    temporal_index = EDGE_FEATURE_NAMES.index("is_temporal")
    spatial_index = EDGE_FEATURE_NAMES.index("is_spatial")
    for position, kind in enumerate(batch.edge_kinds):
        features = batch.edge_features[position]
        assert features[temporal_index] == (1.0 if kind == EDGE_TEMPORAL else 0.0)
        assert features[spatial_index] == (1.0 if kind == EDGE_SPATIAL else 0.0)
