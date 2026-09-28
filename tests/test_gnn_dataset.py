"""Dataset and chronological-split tests (SIH26078, Phase 3).

The leakage tests here are the important ones: a shuffled split would produce a
model that scores well and forecasts badly, and nothing in the metrics would
reveal it.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from extreme_weather.gnn.dataset import (
    GraphDataset,
    SplitConfig,
    build_dataset,
    build_dataset_from_file,
    load_tracks,
    split_dataset,
    split_ratios_from_timestamps,
)
from extreme_weather.gnn.graph import GraphConfig, build_graph

BASE_TIME = datetime(2024, 6, 2, 0, 0, tzinfo=timezone.utc)


# --- split configuration ----------------------------------------------------


def test_default_split_is_sixty_twenty_twenty():
    config = SplitConfig()
    assert (config.train, config.validation, config.test) == (0.6, 0.2, 0.2)
    assert config.train_end == pytest.approx(0.6)
    assert config.validation_end == pytest.approx(0.8)


def test_split_fractions_must_sum_to_one():
    with pytest.raises(ValueError, match="sum to 1.0"):
        SplitConfig(train=0.5, validation=0.2, test=0.2)


def test_split_rejects_negative_fractions():
    with pytest.raises(ValueError, match=">= 0"):
        SplitConfig(train=-0.1, validation=0.5, test=0.6)


def test_split_requires_a_non_empty_train_and_test():
    with pytest.raises(ValueError, match="training fraction"):
        SplitConfig(train=0.0, validation=0.5, test=0.5)
    with pytest.raises(ValueError, match="test fraction"):
        SplitConfig(train=0.8, validation=0.2, test=0.0)


# --- chronological separation ----------------------------------------------


def test_blocks_are_strictly_ordered_in_time(many_tracks):
    """The core invariant: train < validation < test, with no shuffling."""
    dataset = build_dataset(many_tracks)
    batch = dataset.batch
    train_times = [batch.node_timestamps[i] for i in dataset.split.train]
    validation_times = [batch.node_timestamps[i] for i in dataset.split.validation]
    test_times = [batch.node_timestamps[i] for i in dataset.split.test]

    assert max(train_times) <= min(validation_times)
    assert max(validation_times) <= min(test_times)


def test_every_block_is_populated(many_tracks):
    dataset = build_dataset(many_tracks)
    assert dataset.sizes["train"] > 0
    assert dataset.sizes["validation"] > 0
    assert dataset.sizes["test"] > 0


def test_blocks_partition_the_samples_without_overlap(many_tracks):
    dataset = build_dataset(many_tracks)
    flat = [
        index
        for block in (dataset.split.train, dataset.split.validation, dataset.split.test)
        for index in block
    ]
    assert len(flat) == len(set(flat))
    assert sorted(flat) == sorted(dataset.batch.target_indices())
    assert dataset.split.total == dataset.sample_count


def test_no_future_sample_is_in_the_training_block(many_tracks):
    """No training sample may share a timestamp with, or follow, a test sample."""
    dataset = build_dataset(many_tracks)
    batch = dataset.batch
    earliest_test = min(batch.node_timestamps[i] for i in dataset.split.test)
    for index in dataset.split.train:
        assert batch.node_timestamps[index] < earliest_test


def test_the_last_chronological_sample_is_in_the_test_block(many_tracks):
    dataset = build_dataset(many_tracks)
    latest = max(dataset.batch.node_timestamps[i] for i in dataset.split.test)
    assert latest == max(
        dataset.batch.node_timestamps[i] for i in dataset.batch.target_indices()
    )


def test_custom_fractions_change_the_block_sizes(many_tracks):
    generous = build_dataset(many_tracks, split_config=SplitConfig(0.8, 0.1, 0.1))
    modest = build_dataset(many_tracks, split_config=SplitConfig(0.4, 0.3, 0.3))
    assert generous.sizes["train"] > modest.sizes["train"]
    assert modest.sizes["test"] > generous.sizes["test"]


def test_split_is_deterministic(many_tracks):
    first = build_dataset(many_tracks)
    second = build_dataset(many_tracks)
    assert first.split.train == second.split.train
    assert first.split.validation == second.split.validation
    assert first.split.test == second.split.test


def test_split_ignores_the_input_order_of_tracks(many_tracks):
    forwards = build_dataset(many_tracks)
    backwards = build_dataset(list(reversed(many_tracks)))
    assert forwards.split.test == backwards.split.test


# --- split helpers ----------------------------------------------------------


def test_split_requires_at_least_two_timestamps():
    with pytest.raises(ValueError, match="At least 2 timestamps"):
        split_ratios_from_timestamps([BASE_TIME])


def test_split_rejects_identical_timestamps():
    with pytest.raises(ValueError, match="same timestamp"):
        split_ratios_from_timestamps([BASE_TIME, BASE_TIME, BASE_TIME])


def test_split_of_two_samples_still_fills_train_and_test():
    ratios = split_ratios_from_timestamps([BASE_TIME, BASE_TIME + timedelta(hours=5)])
    assert ratios.train
    assert ratios.test


def test_tiny_dataset_does_not_crash(many_tracks):
    """A single-node graph (no targets at all) degrades gracefully."""
    batch = build_graph(many_tracks)
    trimmed = type(batch)(
        node_features=batch.node_features[:1],
        edge_features=[],
        edge_index=([], []),
        node_track_ids=batch.node_track_ids[:1],
        node_timestamps=batch.node_timestamps[:1],
        node_event_ids=batch.node_event_ids[:1],
        node_indices=batch.node_indices[:1],
        targets=[None],
        track_ids=batch.track_ids[:1],
    )
    ratios = split_dataset(trimmed)
    assert ratios.train == []
    assert ratios.validation == []


def test_dataset_counts_match_the_graph(many_tracks):
    dataset = build_dataset(many_tracks)
    assert dataset.node_count == dataset.batch.node_count
    assert dataset.edge_count == dataset.batch.edge_count
    assert dataset.track_count == len(many_tracks)
    assert dataset.sample_count == sum(len(track.points) - 1 for track in many_tracks)


def test_dataset_nodes_of_rejects_an_unknown_block(many_tracks):
    dataset = build_dataset(many_tracks)
    with pytest.raises(ValueError, match="Unknown split block"):
        dataset.nodes_of("holdout")


def test_dataset_summary_is_json_serialisable(many_tracks):
    payload = build_dataset(many_tracks, source="tracks.json").as_dict()
    assert json.loads(json.dumps(payload))["split"]["train"] > 0
    assert payload["node_count"] == 40


# --- loading ----------------------------------------------------------------


def test_load_tracks_round_trips_a_tracking_file(tmp_path, many_tracks):
    path = tmp_path / "tracks.json"
    path.write_text(
        json.dumps({"tracks": [track.model_dump(mode="json") for track in many_tracks]}),
        encoding="utf-8",
    )
    loaded = load_tracks(path)
    assert [track.track_id for track in loaded] == [t.track_id for t in many_tracks]
    assert build_dataset_from_file(path).track_count == len(many_tracks)


def test_load_tracks_reports_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="track_events.py"):
        load_tracks(tmp_path / "absent.json")


def test_load_tracks_rejects_invalid_json(tmp_path):
    path = tmp_path / "tracks.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        load_tracks(path)


def test_load_tracks_rejects_a_document_without_tracks(tmp_path):
    path = tmp_path / "tracks.json"
    path.write_text(json.dumps({"schema_version": "1.0"}), encoding="utf-8")
    with pytest.raises(ValueError, match="does not contain a 'tracks' list"):
        load_tracks(path)


def test_build_dataset_from_file_does_not_modify_the_source(tmp_path, many_tracks):
    path = tmp_path / "tracks.json"
    original = json.dumps({"tracks": [t.model_dump(mode="json") for t in many_tracks]})
    path.write_text(original, encoding="utf-8")
    build_dataset_from_file(path)
    assert path.read_text(encoding="utf-8") == original


def test_spatial_edges_flow_through_to_the_dataset(two_nearby_tracks):
    dataset = build_dataset(two_nearby_tracks, GraphConfig(max_spatial_distance_km=2000.0))
    assert dataset.batch.spatial_edge_count() == 3


def test_dataset_is_a_graph_dataset(many_tracks):
    assert isinstance(build_dataset(many_tracks), GraphDataset)
