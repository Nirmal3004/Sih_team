"""Train the GNN prototype on the Phase 3 graph dataset (SIH26078, Phase 3).

```text
data/processed/tracks.json
        ↓  build_dataset       (graph + chronological split)
        ↓  training.train      (train block only; best on validation)
models/gnn/best.json          (checkpoint)
```

The test block is evaluated **once** after training, purely as a diagnostic. No
model selection ever uses it.

Examples::

    python scripts/train_gnn.py
    python scripts/train_gnn.py --epochs 500 --learning-rate 0.02
    python scripts/train_gnn.py --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from extreme_weather.config import get_settings  # noqa: E402
from extreme_weather.gnn.checkpoint import (  # noqa: E402
    GNNCheckpoint,
    build_training_metadata,
    checkpoint_path,
    save_checkpoint,
)
from extreme_weather.gnn.dataset import SplitConfig, build_dataset_from_file  # noqa: E402
from extreme_weather.gnn.graph import GraphConfig  # noqa: E402
from extreme_weather.gnn.model import ModelConfig  # noqa: E402
from extreme_weather.gnn.training import TrainingConfig, train  # noqa: E402

PROTOTYPE_NOTE = (
    "Prototype model trained on synthetic data. The reported losses describe a "
    "regression on synthetic trajectories; they are NOT a forecasting skill score "
    "and must not be read as real-world accuracy."
)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the training script."""
    settings = get_settings()
    default_tracks = settings.processed_dir / settings.tracks_filename
    default_checkpoint = checkpoint_path()

    parser = argparse.ArgumentParser(
        description="DeepWeather AI (SIH26078) - train the GNN event-evolution prototype."
    )
    parser.add_argument("--tracks", default=None, help=f"Tracks file (default: {default_tracks}).")
    parser.add_argument(
        "--checkpoint", default=None, help=f"Checkpoint output (default: {default_checkpoint})."
    )
    parser.add_argument("--epochs", type=int, default=400, help="Training epochs (default: 400).")
    parser.add_argument(
        "--learning-rate", type=float, default=0.01, help="Adam learning rate (default: 0.01)."
    )
    parser.add_argument(
        "--seed", type=int, default=20260601, help="Random seed (default: 20260601)."
    )
    parser.add_argument("--hidden-dim", type=int, default=32, help="Hidden width (default: 32).")
    parser.add_argument(
        "--message-layers", type=int, default=2, help="Message-passing layers (default: 2)."
    )
    parser.add_argument(
        "--position-weight", type=float, default=1.0, help="Position loss weight (default: 1.0)."
    )
    parser.add_argument(
        "--intensity-weight", type=float, default=0.5, help="Intensity loss weight (default: 0.5)."
    )
    parser.add_argument(
        "--max-spatial-distance-km",
        type=float,
        default=150.0,
        help="Spatial edge threshold (default: 150).",
    )
    parser.add_argument(
        "--no-spatial-edges", action="store_true", help="Train on a purely temporal graph."
    )
    parser.add_argument("--train-fraction", type=float, default=0.6, help="Training time fraction.")
    parser.add_argument(
        "--validation-fraction", type=float, default=0.2, help="Validation time fraction."
    )
    parser.add_argument(
        "--test-fraction", type=float, default=0.2, help="Test time fraction."
    )
    parser.add_argument("--verbose", action="store_true", help="Print every epoch.")
    return parser


def _resolve_split(train: float, validation: float, test: float) -> SplitConfig | None:
    """Build a :class:`SplitConfig`, rebalancing an inconsistent test fraction."""
    if abs(train + validation + test - 1.0) <= 1e-9:
        return SplitConfig(train=train, validation=validation, test=test)
    remainder = 1.0 - train - validation
    if remainder <= 0.0:
        return None
    return SplitConfig(train=train, validation=validation, test=remainder)


def main(argv: list[str] | None = None) -> int:
    """Train, evaluate and write the checkpoint."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    tracks_path = (
        Path(args.tracks) if args.tracks else settings.processed_dir / settings.tracks_filename
    )
    output_path = Path(args.checkpoint) if args.checkpoint else checkpoint_path()

    split_config = _resolve_split(args.train_fraction, args.validation_fraction, args.test_fraction)
    if split_config is None:
        print(
            "error: --train-fraction + --validation-fraction must leave a positive test fraction.",
            file=sys.stderr,
        )
        return 2

    graph_config = GraphConfig(
        max_spatial_distance_km=args.max_spatial_distance_km,
        include_spatial_edges=not args.no_spatial_edges,
    )
    try:
        dataset = build_dataset_from_file(tracks_path, graph_config, split_config)
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    train_indices = dataset.split.train
    if not train_indices:
        print("error: the chronological split produced no training samples.", file=sys.stderr)
        return 1

    model_config = ModelConfig(
        hidden_dim=args.hidden_dim,
        message_layers=args.message_layers,
        position_weight=args.position_weight,
        intensity_weight=args.intensity_weight,
        seed=args.seed,
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        verbose=args.verbose,
    )

    print("Training the GNN prototype")
    print("---------------------------")
    print(f"  tracks file         : {tracks_path}")
    print(f"  tracks              : {dataset.track_count}")
    print(f"  nodes / edges       : {dataset.node_count} / {dataset.edge_count}")
    print(f"  train samples       : {len(train_indices)}")
    print(f"  validation samples  : {len(dataset.split.validation)}")
    print(f"  test samples        : {len(dataset.split.test)}")
    print(f"  epochs              : {args.epochs}")
    print(f"  learning rate       : {args.learning_rate}")
    print(f"  seed                : {args.seed}")
    print()

    result = train(
        batch=dataset.batch,
        train_indices=train_indices,
        validation_indices=dataset.split.validation,
        model_config=model_config,
        training_config=training_config,
        test_indices=dataset.split.test,
    )

    history = result.history
    metadata = build_training_metadata(
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        train_samples=len(train_indices),
        validation_samples=len(dataset.split.validation),
        test_samples=len(dataset.split.test),
        train_loss=history.final_train_loss,
        validation_loss=history.records[-1].validation_loss if history.records else None,
        best_validation_loss=history.best_validation_loss,
        best_epoch=history.best_epoch,
        test_loss=result.test_loss,
        node_count=dataset.node_count,
        edge_count=dataset.edge_count,
        track_count=dataset.track_count,
    )

    checkpoint = GNNCheckpoint(
        model_config=model_config,
        feature_config=dataset.feature_config,
        training=metadata,
        parameters={
            name: tensor.detach().clone() for name, tensor in result.model.state_dict().items()
        },
    )
    written = save_checkpoint(checkpoint, output_path)

    print("Results")
    print("-------")
    print(f"  final train loss     : {history.final_train_loss}")
    print(f"  best validation loss : {history.best_validation_loss}")
    print(f"  best epoch           : {history.best_epoch}")
    if result.test_loss is not None:
        print(f"  test loss (once)     : {result.test_loss}")
    print(f"  checkpoint           : {written}")
    print()
    print(f"  note  : {PROTOTYPE_NOTE}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
