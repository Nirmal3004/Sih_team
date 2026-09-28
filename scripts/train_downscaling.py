"""Train the AI downscaling prototype (SIH26078, Phase 4).

```text
coarse NetCDF
        ↓  synthetic high-resolution reference
        ↓  features + residual targets
        ↓  chronological split (train / validation / test)
        ↓  training on the TRAIN block only, best on validation
models/downscaling/best.pt
```

The test block is evaluated **once** after training, as a diagnostic. No model
selection ever uses it.

Examples::

    python scripts/train_downscaling.py
    python scripts/train_downscaling.py --epochs 40 --learning-rate 0.003
    python scripts/train_downscaling.py --max-frames 24 --verbose
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from extreme_weather.config import get_settings  # noqa: E402
from extreme_weather.downscaling.checkpoint import (  # noqa: E402
    DownscalingCheckpoint,
    build_training_metadata,
    checkpoint_path,
    save_checkpoint,
)
from extreme_weather.downscaling.dataset import (  # noqa: E402
    SplitConfig,
    build_downscaling_dataset,
    load_coarse_dataset,
)
from extreme_weather.downscaling.model import ModelConfig  # noqa: E402
from extreme_weather.downscaling.reference import ReferenceConfig  # noqa: E402
from extreme_weather.downscaling.training import TrainingConfig, train  # noqa: E402

PROTOTYPE_NOTE = (
    "Prototype model trained against a SYNTHETIC high-resolution reference. The "
    "reported losses measure agreement with that generated target; they are NOT "
    "real-world forecast skill and NOT operational accuracy."
)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the training script."""
    settings = get_settings()
    default_input = settings.processed_dir / "sample_history_processed.nc"
    default_checkpoint = checkpoint_path()

    parser = argparse.ArgumentParser(
        description="DeepWeather AI (SIH26078) - train the AI downscaling prototype."
    )
    parser.add_argument("--input", default=None, help=f"Coarse NetCDF (default: {default_input}).")
    parser.add_argument(
        "--checkpoint", default=None, help=f"Checkpoint output (default: {default_checkpoint})."
    )
    parser.add_argument(
        "--variable", default="precipitation", help="Variable to downscale (default: precipitation)."
    )
    parser.add_argument(
        "--target-resolution", type=float, default=None, help="Target resolution in degrees."
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Use only the first N frames.")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs (default: 60).")
    parser.add_argument(
        "--learning-rate", type=float, default=2e-3, help="Adam learning rate (default: 0.002)."
    )
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8).")
    parser.add_argument("--seed", type=int, default=20260601, help="Random seed (default: 20260601).")
    parser.add_argument("--hidden-channels", type=int, default=32, help="Hidden width (default: 32).")
    parser.add_argument("--num-blocks", type=int, default=3, help="Residual blocks (default: 3).")
    parser.add_argument(
        "--patience", type=int, default=0, help="Early-stopping patience; 0 (default) disables it."
    )
    parser.add_argument("--train-fraction", type=float, default=0.6, help="Training time fraction.")
    parser.add_argument(
        "--validation-fraction", type=float, default=0.2, help="Validation time fraction."
    )
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Test time fraction.")
    parser.add_argument(
        "--structure-strength", type=float, default=0.45, help="Synthetic structure amplitude."
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
    """Train the prototype and write the checkpoint."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    input_path = Path(args.input) if args.input else (
        settings.processed_dir / "sample_history_processed.nc"
    )
    output_path = Path(args.checkpoint) if args.checkpoint else checkpoint_path()
    target_resolution = (
        args.target_resolution
        if args.target_resolution is not None
        else settings.downsample_target_resolution_deg
    )

    split_config = _resolve_split(args.train_fraction, args.validation_fraction, args.test_fraction)
    if split_config is None:
        print(
            "error: --train-fraction + --validation-fraction must leave a positive test fraction.",
            file=sys.stderr,
        )
        return 2

    try:
        coarse, latitude, longitude, timestamps = load_coarse_dataset(input_path)
    except (FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    subset = slice(0, args.max_frames) if args.max_frames else None
    try:
        dataset = build_downscaling_dataset(
            coarse,
            latitude,
            longitude,
            timestamps,
            target_resolution,
            variable=args.variable,
            reference_config=ReferenceConfig(structure_strength=args.structure_strength),
            split_config=split_config,
            time_subset=subset,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    model_config = ModelConfig(
        hidden_channels=args.hidden_channels, num_blocks=args.num_blocks
    )
    training_config = TrainingConfig(
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        seed=args.seed,
        patience=args.patience if args.patience > 0 else None,
        verbose=args.verbose,
    )

    print("Training the AI downscaling prototype")
    print("--------------------------------------")
    print(f"  input file        : {input_path}")
    print(f"  variable          : {dataset.variable}")
    print(f"  input resolution  : {dataset.input_resolution_deg:.4f} deg "
          f"({dataset.input_shape[0]} x {dataset.input_shape[1]})")
    print(f"  output resolution : {dataset.output_resolution_deg:.4f} deg "
          f"({dataset.output_shape[0]} x {dataset.output_shape[1]})")
    print(f"  scale factor      : {dataset.scale_factor:.3f}")
    print(f"  train samples     : {len(dataset.split.train)}")
    print(f"  validation samples: {len(dataset.split.validation)}")
    print(f"  test samples      : {len(dataset.split.test)}")
    print(f"  epochs            : {args.epochs}")
    print(f"  learning rate     : {args.learning_rate}")
    print(f"  batch size        : {args.batch_size}")
    print(f"  seed              : {args.seed}")
    print()

    result = train(dataset, model_config, training_config)
    history = result.history

    metadata = build_training_metadata(
        seed=args.seed,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        train_samples=len(dataset.split.train),
        validation_samples=len(dataset.split.validation),
        test_samples=len(dataset.split.test),
        train_loss=history.final_train_loss,
        validation_loss=history.records[-1].validation_loss if history.records else None,
        best_validation_loss=history.best_validation_loss,
        best_epoch=history.best_epoch,
        test_loss=result.test_loss,
        stopped_early=history.stopped_early,
    )

    checkpoint = DownscalingCheckpoint(
        model_config=model_config,
        normalization=dataset.normalization,
        training=metadata,
        model_state={
            name: tensor.detach().clone()
            for name, tensor in result.model.state_dict().items()
        },
        input_resolution_deg=dataset.input_resolution_deg,
        output_resolution_deg=dataset.output_resolution_deg,
        input_shape=dataset.input_shape,
        output_shape=dataset.output_shape,
        variable=dataset.variable,
    )
    written = save_checkpoint(checkpoint, output_path)

    print("Results")
    print("-------")
    print(f"  final train loss     : {history.final_train_loss}")
    print(f"  best validation loss : {history.best_validation_loss}")
    print(f"  best epoch           : {history.best_epoch}")
    print(f"  test loss (once)     : {result.test_loss}")
    print(f"  stopped early        : {history.stopped_early}")
    print(f"  parameters           : {result.model.parameter_count}")
    print(f"  checkpoint           : {written}")
    print()
    print(f"  note  : {PROTOTYPE_NOTE}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())

