"""Build the GNN graph dataset from Phase 2 tracks (SIH26078, Phase 3).

```text
data/processed/tracks.json
        ↓  graph.build_graph
        ↓  dataset.split_dataset  (chronological train / validation / test)
GraphDataset (in memory; nothing is written)
```

The script only **reads**; it never writes, and in particular never touches
``data/raw``.

Examples::

    python scripts/build_graph_dataset.py
    python scripts/build_graph_dataset.py --max-spatial-distance-km 80
    python scripts/build_graph_dataset.py --no-spatial-edges
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from extreme_weather.config import get_settings  # noqa: E402
from extreme_weather.gnn.dataset import (  # noqa: E402
    SplitConfig,
    build_dataset_from_file,
)

PROTOTYPE_NOTE = (
    "Prototype graph built from synthetic tracks. Not an operational product and not "
    "verified against observed weather."
)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the dataset builder."""
    settings = get_settings()
    default_tracks = settings.processed_dir / settings.tracks_filename

    parser = argparse.ArgumentParser(
        description="DeepWeather AI (SIH26078) - build the GNN graph dataset from tracks."
    )
    parser.add_argument(
        "--tracks",
        default=None,
        help=f"Phase 2 tracks file (default: {default_tracks}).",
    )
    parser.add_argument(
        "--max-spatial-distance-km",
        type=float,
        default=150.0,
        help="Largest distance for a spatial edge between concurrent events (default: 150).",
    )
    parser.add_argument(
        "--no-spatial-edges",
        action="store_true",
        help="Build a purely temporal graph (no cross-track edges).",
    )
    parser.add_argument(
        "--train-fraction", type=float, default=0.6, help="Fraction of the time span for training."
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of the time span for validation.",
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2, help="Fraction of the time span for testing."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Build the graph, split it chronologically and print the counts."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    tracks_path = Path(args.tracks) if args.tracks else (
        settings.processed_dir / settings.tracks_filename
    )

    test_fraction = args.test_fraction
    if abs(args.train_fraction + args.validation_fraction + test_fraction - 1.0) > 1e-9:
        remainder = 1.0 - args.train_fraction - args.validation_fraction
        if remainder <= 0.0:
            print(
                "error: --train-fraction + --validation-fraction must leave a positive "
                "test fraction.",
                file=sys.stderr,
            )
            return 2
        test_fraction = remainder

    from extreme_weather.gnn.features import EDGE_FEATURE_NAMES, NODE_FEATURE_NAMES
    from extreme_weather.gnn.graph import GraphConfig

    graph_config = GraphConfig(
        max_spatial_distance_km=args.max_spatial_distance_km,
        include_spatial_edges=not args.no_spatial_edges,
    )
    split_config = SplitConfig(
        train=args.train_fraction,
        validation=args.validation_fraction,
        test=test_fraction,
    )

    try:
        dataset = build_dataset_from_file(tracks_path, graph_config, split_config)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    summary = dataset.as_dict()
    print("GNN graph dataset")
    print("----------------")
    print(f"  tracks file          : {tracks_path}")
    print(f"  tracks               : {summary['track_count']}")
    print(f"  nodes                : {summary['node_count']}")
    print(f"  edges                : {summary['edge_count']}")
    print(f"    temporal           : {summary['temporal_edge_count']}")
    print(f"    spatial            : {summary['spatial_edge_count']}")
    print(f"  prediction samples   : {summary['prediction_samples']}")
    print(f"  train samples        : {summary['split']['train']}")
    print(f"  validation samples   : {summary['split']['validation']}")
    print(f"  test samples         : {summary['split']['test']}")
    print(f"  node features        : {len(NODE_FEATURE_NAMES)}")
    print(f"  edge features        : {len(EDGE_FEATURE_NAMES)}")
    print(f"  spatial threshold    : {graph_config.max_spatial_distance_km} km")

    if dataset.split.earliest_time and dataset.split.latest_time:
        print(f"  time span            : {dataset.split.earliest_time} .. {dataset.split.latest_time}")
    if dataset.split.train_end_time:
        print(f"  train ends           : {dataset.split.train_end_time}")
    if dataset.split.validation_end_time:
        print(f"  validation ends      : {dataset.split.validation_end_time}")

    print()
    print("  note  : the split is chronological - train precedes validation precedes test.")
    print(f"  note  : {PROTOTYPE_NOTE}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
