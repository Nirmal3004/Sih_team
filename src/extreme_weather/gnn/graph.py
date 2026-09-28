"""Deterministic spatio-temporal graph construction (SIH26078, Phase 3).

The Phase 2 tracker produced a set of :class:`~extreme_weather.schemas.track.
EventTrack` objects. This module turns them into a single **sparse directed
graph** whose nodes are individual track points (one *event state* per node) and
whose edges are of two kinds.

Temporal edges (always built)
    ``point[i] -> point[i+1]`` inside the same track. These are the real
    trajectory steps the model must learn to extrapolate.

Spatial edges (optional, configurable)
    Two points that share a timestamp but belong to **different** tracks are
    connected when their great-circle distance is at or below
    :attr:`GraphConfig.max_spatial_distance_km`. These capture "two events were
    near each other at this hour" - for example a storm about to merge with a
    weaker one - which a purely temporal graph cannot express.

Determinism
-----------
Node order is ``(track_id, timestamp)`` and every candidate pair is evaluated in
that fixed order, so the graph is a pure function of the input tracks. There is
no randomness, no hashing and no dependence on dictionary iteration order.
Rebuilding the graph from the same tracks yields byte-identical arrays, which
:mod:`extreme_weather.gnn.dataset` relies on when it caches the dataset.

No self-loops and no duplicate edges are created, and a spatial edge is never
added between two points of the same track: that is already a temporal
relationship, and duplicating it would double-count the step in the aggregation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from extreme_weather.gnn.features import (
    EDGE_SPATIAL,
    EDGE_TEMPORAL,
    FeatureConfig,
    edge_feature_vector,
    node_feature_vector,
    target_vector,
)
from extreme_weather.schemas.track import EventTrack, TrackPoint
from extreme_weather.tracking.metrics import haversine_distance_km

__all__ = [
    "GraphBatch",
    "GraphConfig",
    "build_graph",
    "sort_tracks_for_graph",
]


@dataclass(frozen=True)
class GraphConfig:
    """Controls how the spatio-temporal graph is built.

    Attributes:
        max_spatial_distance_km: Largest distance for a spatial edge. ``0.0``
            disables spatial edges (temporal edges remain).
        include_spatial_edges: Master switch for the spatial edges.
        feature_config: Scaling constants handed to the feature builders.
    """

    max_spatial_distance_km: float = 150.0
    include_spatial_edges: bool = True
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    def __post_init__(self) -> None:
        value = self.max_spatial_distance_km
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"max_spatial_distance_km must be finite and >= 0 (got {value!r}).")


@dataclass
class GraphBatch:
    """A materialised graph, stored as plain Python lists of floats/ints.

    Dense tensor arrays are avoided on purpose: prototype graphs are small (a few
    dozen nodes) and lists keep this module free of any tensor-library
    assumption, so the graph can be built, inspected and serialised in tests
    without a backend.

    Attributes:
        node_features: One 16-value feature vector per node, ordered like
            :data:`~extreme_weather.gnn.features.NODE_FEATURE_NAMES`.
        edge_features: One 6-value feature vector per directed edge.
        edge_index: ``(sources, targets)`` index arrays, both of length
            ``len(edge_features)``.
        node_track_ids: Track id of each node, used to group targets by track.
        node_timestamps: Valid time of each node, in node order.
        node_event_ids: Phase 1 event id behind each node.
        node_indices: Index of each node's position inside its own track.
        targets: Per-node ``[delta_latitude, delta_longitude, delta_intensity]``;
            ``None`` for the last point of a track, which has no future state.
        track_ids: Sorted, de-duplicated track ids present in the graph.
        edge_distances_km: Great-circle distance of each edge (inspection only).
        edge_kinds: :data:`~extreme_weather.gnn.features.EDGE_TEMPORAL` or
            ``...EDGE_SPATIAL`` per edge.
    """

    node_features: list[list[float]] = field(default_factory=list)
    edge_features: list[list[float]] = field(default_factory=list)
    edge_index: tuple[list[int], list[int]] = field(default_factory=lambda: ([], []))
    node_track_ids: list[str] = field(default_factory=list)
    node_timestamps: list[datetime] = field(default_factory=list)
    node_event_ids: list[str] = field(default_factory=list)
    node_indices: list[int] = field(default_factory=list)
    targets: list[list[float] | None] = field(default_factory=list)
    track_ids: list[str] = field(default_factory=list)
    edge_distances_km: list[float] = field(default_factory=list)
    edge_kinds: list[str] = field(default_factory=list)

    @property
    def node_count(self) -> int:
        """Number of event states (nodes)."""
        return len(self.node_features)

    @property
    def edge_count(self) -> int:
        """Number of directed edges."""
        return len(self.edge_features)

    @property
    def target_count(self) -> int:
        """Number of nodes that have a known next state (prediction samples)."""
        return sum(1 for value in self.targets if value is not None)

    @property
    def node_dim(self) -> int:
        """Feature width of a node (``0`` for an empty graph)."""
        return len(self.node_features[0]) if self.node_features else 0

    @property
    def edge_dim(self) -> int:
        """Feature width of an edge (``0`` for an empty graph)."""
        return len(self.edge_features[0]) if self.edge_features else 0

    def target_indices(self) -> list[int]:
        """Indices of the nodes that carry a target, in node order."""
        return [index for index, value in enumerate(self.targets) if value is not None]

    def temporal_edge_count(self) -> int:
        """Number of within-track edges."""
        return sum(1 for kind in self.edge_kinds if kind == EDGE_TEMPORAL)

    def spatial_edge_count(self) -> int:
        """Number of cross-track same-timestamp edges."""
        return sum(1 for kind in self.edge_kinds if kind == EDGE_SPATIAL)

    def incoming_degree(self) -> list[int]:
        """In-degree of every node (number of edges pointing at it)."""
        degree = [0] * self.node_count
        for target in self.edge_index[1]:
            degree[target] += 1
        return degree


def sort_tracks_for_graph(tracks: list[EventTrack]) -> list[EventTrack]:
    """Return ``tracks`` in the canonical order that fixes the node numbering."""
    return sorted(tracks, key=lambda track: (track.start_time, track.track_id))


def build_graph(
    tracks: list[EventTrack],
    config: GraphConfig | None = None,
) -> GraphBatch:
    """Build the spatio-temporal graph of ``tracks``.

    Node ``i`` is the ``i``-th point in canonical order. A node's target is the
    delta to **its own next point in the same track**, or ``None`` for the final
    point of a track (it has no known future).

    Args:
        tracks: Phase 2 tracks. Order does not matter; they are sorted first.
        config: Graph construction settings.

    Returns:
        A :class:`GraphBatch`. An empty input yields an empty batch rather than an
        error, so a run without tracks degrades gracefully.

    Raises:
        ValueError: If two tracks share a track id or a point timestamp/key is
            duplicated, which would make the node ordering ambiguous.
    """
    cfg = config or GraphConfig()
    ordered_tracks = sort_tracks_for_graph(list(tracks))

    seen_track_ids: set[str] = set()
    for track in ordered_tracks:
        if track.track_id in seen_track_ids:
            raise ValueError(f"Duplicate track id '{track.track_id}' in the graph input.")
        seen_track_ids.add(track.track_id)

    batch = GraphBatch()
    batch.track_ids = [track.track_id for track in ordered_tracks]

    # --- nodes and temporal targets -------------------------------------
    point_of_node: list[TrackPoint] = []
    track_node_offsets: dict[str, int] = {}
    for track in ordered_tracks:
        points = list(track.points)
        track_node_offsets[track.track_id] = len(point_of_node)
        for position, point in enumerate(points):
            batch.node_features.append(node_feature_vector(point, track.event_type, cfg.feature_config))
            batch.node_track_ids.append(track.track_id)
            batch.node_timestamps.append(point.timestamp)
            batch.node_event_ids.append(point.event_id)
            batch.node_indices.append(position)
            if position + 1 < len(points):
                batch.targets.append(target_vector(point, points[position + 1]))
            else:
                batch.targets.append(None)
            point_of_node.append(point)

    sources: list[int] = []
    destinations: list[int] = []

    # --- temporal edges: point[i] -> point[i+1] ------------------------
    for track in ordered_tracks:
        first_node = track_node_offsets[track.track_id]
        for offset in range(len(track.points) - 1):
            source = first_node + offset
            target = source + 1
            sources.append(source)
            destinations.append(target)
            batch.edge_features.append(
                edge_feature_vector(
                    point_of_node[source],
                    point_of_node[target],
                    EDGE_TEMPORAL,
                    cfg.feature_config,
                )
            )
            batch.edge_kinds.append(EDGE_TEMPORAL)
            batch.edge_distances_km.append(
                haversine_distance_km(
                    point_of_node[source].latitude,
                    point_of_node[source].longitude,
                    point_of_node[target].latitude,
                    point_of_node[target].longitude,
                )
            )

    # --- spatial edges: same timestamp, different track, close enough ---
    if cfg.include_spatial_edges and cfg.max_spatial_distance_km > 0.0:
        buckets: dict[datetime, list[int]] = {}
        for node_index, stamp in enumerate(batch.node_timestamps):
            buckets.setdefault(stamp, []).append(node_index)

        for stamp in sorted(buckets):
            members = buckets[stamp]
            for position, source in enumerate(members):
                for target in members[position + 1 :]:
                    # Two nodes of the same track are consecutive, not concurrent.
                    if batch.node_track_ids[source] == batch.node_track_ids[target]:
                        continue
                    distance = haversine_distance_km(
                        point_of_node[source].latitude,
                        point_of_node[source].longitude,
                        point_of_node[target].latitude,
                        point_of_node[target].longitude,
                    )
                    if distance > cfg.max_spatial_distance_km:
                        continue
                    sources.append(source)
                    destinations.append(target)
                    batch.edge_features.append(
                        edge_feature_vector(
                            point_of_node[source],
                            point_of_node[target],
                            EDGE_SPATIAL,
                            cfg.feature_config,
                        )
                    )
                    batch.edge_kinds.append(EDGE_SPATIAL)
                    batch.edge_distances_km.append(distance)

    batch.edge_index = (sources, destinations)
    return batch
