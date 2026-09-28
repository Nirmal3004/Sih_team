"""Graph dataset assembly and the chronological train/val/test split.

Why a chronological split
-------------------------
Weather events evolve forward in time. If samples were shuffled, a model could be
trained on a storm's *later* states and evaluated on its *earlier* states, which
is temporal leakage: the reported error would be optimistic and meaningless for
forecasting. So this module never shuffles. Samples are sorted by the timestamp
of the **current** node and cut into three contiguous blocks:

```text
   earliest .................................................. latest
   |------------------ TRAIN -----------------|
                                  |--- VALIDATION ---|
                                                |-- TEST --|
```

The cut points are chosen by *time* (fractions of the full timestamp range), not
by sample count, so the blocks always correspond to physically distinct periods.
The invariant the tests assert is
``max(train) <= min(validation) <= min(test)``.

A note on the target: a node's target is the delta to its own next point, i.e. a
future observation. That is the *label*, not an input - the feature builders in
:mod:`extreme_weather.gnn.features` never read it. A training sample therefore
exposes the model only information from its own timestamp, and the split keeps
whole *periods* apart rather than merely splitting sample lists.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Sequence

from extreme_weather.gnn.features import FeatureConfig
from extreme_weather.gnn.graph import GraphBatch, GraphConfig, build_graph
from extreme_weather.schemas.track import EventTrack

__all__ = [
    "GraphDataset",
    "SplitConfig",
    "SplitRatios",
    "build_dataset",
    "build_dataset_from_file",
    "load_tracks",
    "split_dataset",
    "split_ratios_from_timestamps",
]


@dataclass(frozen=True)
class SplitConfig:
    """Fractions of the time span assigned to each block.

    They must be non-negative and sum to 1.0; ``train`` and ``test`` are
    required to be non-empty by construction (``0.0`` train or ``0.0`` test
    raises), while validation may legitimately be empty on tiny fixtures - the
    split degrades gracefully instead of failing.
    """

    train: float = 0.6
    validation: float = 0.2
    test: float = 0.2

    def __post_init__(self) -> None:
        values = (self.train, self.validation, self.test)
        if any(value < 0.0 for value in values):
            raise ValueError(f"Split fractions must be >= 0 (got {values}).")
        total = sum(values)
        if abs(total - 1.0) > 1e-9:
            raise ValueError(f"Split fractions must sum to 1.0 (got {total!r}).")
        if self.train <= 0.0:
            raise ValueError("The training fraction must be > 0.")
        if self.test <= 0.0:
            raise ValueError("The test fraction must be > 0.")

    @property
    def train_end(self) -> float:
        """Cumulative fraction marking the end of the training period."""
        return self.train

    @property
    def validation_end(self) -> float:
        """Cumulative fraction marking the end of the validation period."""
        return self.train + self.validation


@dataclass(frozen=True)
class SplitRatios:
    """The realised outcome of a split, useful for the CLI report and tests.

    Attributes:
        train: Node indices in the training block.
        validation: Node indices in the validation block.
        test: Node indices in the test block.
        train_end_time: Last timestamp assigned to training.
        validation_end_time: Last timestamp assigned to validation.
        earliest_time: First timestamp in the whole dataset.
        latest_time: Last timestamp in the whole dataset.
    """

    train: list[int] = field(default_factory=list)
    validation: list[int] = field(default_factory=list)
    test: list[int] = field(default_factory=list)
    train_end_time: datetime | None = None
    validation_end_time: datetime | None = None
    earliest_time: datetime | None = None
    latest_time: datetime | None = None

    @property
    def sizes(self) -> dict[str, int]:
        """Sample count of each block."""
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    @property
    def total(self) -> int:
        """Total number of samples across the three blocks."""
        return len(self.train) + len(self.validation) + len(self.test)


def split_ratios_from_timestamps(
    timestamps: Sequence[datetime],
    config: SplitConfig | None = None,
) -> SplitRatios:
    """Split sample indices into three chronological blocks.

    Only nodes that actually carry a target are candidates. They are ordered by
    their **current** timestamp, then split at the fractional cut points of the
    observed time range, so each block is a distinct period.

    Args:
        timestamps: Valid time of every candidate sample (the current node).
        config: Block fractions; defaults to 60/20/20.

    Returns:
        A :class:`SplitRatios`. When the timestamps are empty every block is
        empty; when validation would be empty it is allowed to be.

    Raises:
        ValueError: If fewer than two timestamps are given, because a single
            sample cannot be split into three distinct periods.
    """
    cfg = config or SplitConfig()
    if len(timestamps) < 2:
        raise ValueError(
            f"At least 2 timestamps are required for a chronological split (got {len(timestamps)})."
        )

    ordered = sorted(range(len(timestamps)), key=lambda index: (timestamps[index], index))
    ordered_stamps = [timestamps[index] for index in ordered]

    earliest = ordered_stamps[0]
    latest = ordered_stamps[-1]
    span = (latest - earliest).total_seconds()
    if span <= 0.0:
        # Every sample shares one timestamp: it cannot be separated in time.
        raise ValueError(
            "All samples share the same timestamp; a chronological split is impossible."
        )

    def cumulative(fraction: float) -> datetime:
        """Translate a fraction of the time span into an absolute timestamp."""
        return earliest + timedelta(seconds=span * fraction)

    train_end = cumulative(cfg.train_end)
    validation_end = cumulative(cfg.validation_end)

    train: list[int] = []
    validation: list[int] = []
    test: list[int] = []
    for index in ordered:
        stamp = timestamps[index]
        if stamp <= train_end:
            train.append(index)
        elif stamp <= validation_end:
            validation.append(index)
        else:
            test.append(index)

    # Guarantee a non-empty test block on degenerate fixtures: if every sample
    # fell into the first two blocks, move the last training sample over.
    if not test and len(train) > 1:
        moved = train.pop()
        test.append(moved)
    elif not test and train:
        test.append(train.pop())

    return SplitRatios(
        train=train,
        validation=validation,
        test=test,
        train_end_time=max((timestamps[i] for i in train), default=None),
        validation_end_time=max((timestamps[i] for i in validation), default=None),
        earliest_time=earliest,
        latest_time=latest,
    )


def split_dataset(
    batch: GraphBatch,
    config: SplitConfig | None = None,
) -> SplitRatios:
    """Chronological split of the nodes of ``batch`` that carry a target."""
    candidates = batch.target_indices()
    timestamps = [batch.node_timestamps[index] for index in candidates]

    if len(candidates) < 2:
        # Not enough samples to cut into three periods; keep them in training
        # rather than failing, and let the caller report an empty val/test.
        return SplitRatios(
            train=list(candidates),
            validation=[],
            test=[],
            train_end_time=max(timestamps, default=None),
            validation_end_time=None,
            earliest_time=min(timestamps, default=None),
            latest_time=max(timestamps, default=None),
        )

    ratios = split_ratios_from_timestamps(timestamps, config)
    # Map the positions within `candidates` back to real node indices.
    return SplitRatios(
        train=[candidates[i] for i in ratios.train],
        validation=[candidates[i] for i in ratios.validation],
        test=[candidates[i] for i in ratios.test],
        train_end_time=ratios.train_end_time,
        validation_end_time=ratios.validation_end_time,
        earliest_time=ratios.earliest_time,
        latest_time=ratios.latest_time,
    )


@dataclass
class GraphDataset:
    """A graph plus its chronological split - the training input as a whole.

    Attributes:
        batch: The full graph (nodes, edges, targets).
        split: Which nodes belong to train / validation / test.
        graph_config: The configuration the graph was built with.
        feature_config: The scaling constants used for the features.
        source: Path of the tracks file the graph was derived from, if any.
    """

    batch: GraphBatch
    split: SplitRatios
    graph_config: GraphConfig = field(default_factory=GraphConfig)
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)
    source: str | None = None

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return self.batch.node_count

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return self.batch.edge_count

    @property
    def track_count(self) -> int:
        """Number of tracks the graph was built from."""
        return len(self.batch.track_ids)

    @property
    def sample_count(self) -> int:
        """Number of nodes that carry a target."""
        return self.batch.target_count

    @property
    def sizes(self) -> dict[str, int]:
        """Sample count per block (``train``, ``validation``, ``test``)."""
        return self.split.sizes

    def nodes_of(self, block: str) -> list[int]:
        """Node indices of one block.

        Raises:
            ValueError: If ``block`` is not ``train``, ``validation`` or ``test``.
        """
        if block == "train":
            return self.split.train
        if block == "validation":
            return self.split.validation
        if block == "test":
            return self.split.test
        raise ValueError(f"Unknown split block {block!r}; expected train/validation/test.")

    def as_dict(self) -> dict[str, Any]:
        """JSON-serialisable summary (no raw feature matrices)."""
        return {
            "schema_version": "1.0",
            "project": "SIH26078",
            "phase": "Phase 3 - GNN event evolution (prototype)",
            "source": self.source,
            "track_count": self.track_count,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "temporal_edge_count": self.batch.temporal_edge_count(),
            "spatial_edge_count": self.batch.spatial_edge_count(),
            "prediction_samples": self.sample_count,
            "split": {
                "train": len(self.split.train),
                "validation": len(self.split.validation),
                "test": len(self.split.test),
            },
            "graph_config": {
                "max_spatial_distance_km": self.graph_config.max_spatial_distance_km,
                "include_spatial_edges": self.graph_config.include_spatial_edges,
            },
        }


def load_tracks(path: Path | str) -> list[EventTrack]:
    """Read tracks from a Phase 2 ``tracks.json`` file.

    Args:
        path: Location of the tracking file.

    Returns:
        The validated :class:`~extreme_weather.schemas.track.EventTrack` list.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the payload is not a valid tracking document.
    """
    track_path = Path(path)
    if not track_path.is_file():
        raise FileNotFoundError(
            f"No tracking file at '{track_path}'. Run 'python scripts/track_events.py' first."
        )
    try:
        payload = json.loads(track_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Tracking file '{track_path}' is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict) or "tracks" not in payload:
        raise ValueError(f"Tracking file '{track_path}' does not contain a 'tracks' list.")
    try:
        return [EventTrack.model_validate(item) for item in payload["tracks"]]
    except Exception as exc:  # pydantic ValidationError and friends
        raise ValueError(f"Tracking file '{track_path}' contains an invalid track: {exc}") from exc


def build_dataset(
    tracks: Sequence[EventTrack],
    graph_config: GraphConfig | None = None,
    split_config: SplitConfig | None = None,
    source: str | None = None,
) -> GraphDataset:
    """Build the graph of ``tracks`` and split it chronologically."""
    cfg = graph_config or GraphConfig()
    batch = build_graph(list(tracks), cfg)
    split = split_dataset(batch, split_config)
    return GraphDataset(
        batch=batch,
        split=split,
        graph_config=cfg,
        feature_config=cfg.feature_config,
        source=source,
    )


def build_dataset_from_file(
    tracks_path: Path | str,
    graph_config: GraphConfig | None = None,
    split_config: SplitConfig | None = None,
) -> GraphDataset:
    """Load ``tracks.json`` and build the dataset in one step."""
    tracks = load_tracks(tracks_path)
    return build_dataset(tracks, graph_config, split_config, source=str(tracks_path))
