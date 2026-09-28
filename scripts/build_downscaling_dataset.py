"""Build the AI downscaling dataset from a coarse dataset (SIH26078, Phase 4).

```text
data/processed/sample_history_processed.nc  (coarse, ~12 km)
        ↓  synthetic high-resolution reference per frame  (~5 km)
        ↓  feature extraction + residual targets
        ↓  chronological train / validation / test split
        ↓  normalisation fitted on the TRAINING block only
data/processed/downscaling/dataset.npz  +  dataset.json
```

The high-resolution reference is **synthetic**: it is generated deterministically
from the coarse field, it is not observational truth, and metrics measured
against it describe agreement with that generated target only. See
``docs/AI_DOWNSCALING.md``. ``data/raw/`` is never written.

Examples::

    python scripts/build_downscaling_dataset.py
    python scripts/build_downscaling_dataset.py --target-resolution 0.05
    python scripts/build_downscaling_dataset.py --max-frames 24
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

from extreme_weather.config import get_settings  # noqa: E402
from extreme_weather.downscaling.dataset import (  # noqa: E402
    SplitConfig,
    build_downscaling_dataset,
    load_coarse_dataset,
)
from extreme_weather.downscaling.features import FEATURE_NAMES  # noqa: E402
from extreme_weather.downscaling.reference import ReferenceConfig  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the dataset builder."""
    settings = get_settings()
    default_input = settings.processed_dir / "sample_history_processed.nc"
    default_output = settings.processed_dir / "downscaling"

    parser = argparse.ArgumentParser(
        description="DeepWeather AI (SIH26078) - build the AI downscaling dataset."
    )
    parser.add_argument(
        "--input", default=None, help=f"Coarse processed NetCDF (default: {default_input})."
    )
    parser.add_argument(
        "--output-dir", default=None, help=f"Output directory (default: {default_output})."
    )
    parser.add_argument(
        "--variable",
        default="precipitation",
        help="Variable to downscale (default: precipitation).",
    )
    parser.add_argument(
        "--target-resolution",
        type=float,
        default=None,
        help="Target resolution in degrees (default: the configured value).",
    )
    parser.add_argument("--max-frames", type=int, default=None, help="Use only the first N frames.")
    parser.add_argument("--train-fraction", type=float, default=0.6, help="Training time fraction.")
    parser.add_argument(
        "--validation-fraction", type=float, default=0.2, help="Validation time fraction."
    )
    parser.add_argument("--test-fraction", type=float, default=0.2, help="Test time fraction.")
    parser.add_argument(
        "--structure-strength",
        type=float,
        default=0.45,
        help="Amplitude of the synthetic fine-scale structure (default: 0.45).",
    )
    parser.add_argument("--no-write", action="store_true", help="Do not write any file.")
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
    """Build the dataset, split it chronologically and print the counts."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    input_path = Path(args.input) if args.input else (
        settings.processed_dir / "sample_history_processed.nc"
    )
    output_dir = Path(args.output_dir) if args.output_dir else (
        settings.processed_dir / "downscaling"
    )
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

    summary = dataset.summary()
    print("AI downscaling dataset")
    print("---------------------")
    print(f"  input file          : {input_path}")
    print(f"  variable            : {dataset.variable}")
    print(f"  input resolution    : {dataset.input_resolution_deg:.4f} deg")
    print(f"  output resolution   : {dataset.output_resolution_deg:.4f} deg")
    print(f"  scale factor        : {dataset.scale_factor:.3f}")
    print(f"  input grid          : {dataset.input_shape[0]} x {dataset.input_shape[1]}")
    print(f"  output grid         : {dataset.output_shape[0]} x {dataset.output_shape[1]}")
    print(f"  samples             : {dataset.sample_count}")
    print(f"  train samples       : {summary['split']['train']}")
    print(f"  validation samples  : {summary['split']['validation']}")
    print(f"  test samples        : {summary['split']['test']}")
    print(f"  feature channels    : {len(FEATURE_NAMES)}")
    print(f"  normalisation       : {dataset.normalization.source}")
    print(
        "  conservation error  : "
        f"{dataset.metadata['conservation_error_max']:.3e} "
        "(max |area-average(reference) - coarse|)"
    )
    print()
    print("  note  : the high-resolution target is SYNTHETIC, not observational truth.")
    print("  note  : metrics against it are not real-world forecast skill.")

    if args.no_write:
        print()
        print("  note  : --no-write, nothing stored.")
        return 0

    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        arrays_path = output_dir / "dataset.npz"
        np.savez_compressed(
            arrays_path,
            features=dataset.features,
            targets=dataset.targets,
            references=dataset.references,
            baselines=dataset.baselines,
            coarse=dataset.coarse,
        )
        metadata_path = output_dir / "dataset.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "schema_version": "1.0",
                    "project": "SIH26078",
                    "phase": "Phase 4 - AI downscaling (research prototype)",
                    **summary,
                    "split_indices": {
                        "train": dataset.split.train,
                        "validation": dataset.split.validation,
                        "test": dataset.split.test,
                    },
                    "timestamps": [str(value) for value in dataset.timestamps],
                    "normalization": dataset.normalization.as_dict(),
                    "metadata": {
                        key: value
                        for key, value in dataset.metadata.items()
                        if key
                        not in ("coarse_latitude", "coarse_longitude", "fine_latitude")
                    },
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    except OSError as exc:
        print(f"error: could not write the dataset: {exc}", file=sys.stderr)
        return 1

    print(f"  arrays              : {arrays_path}")
    print(f"  metadata            : {metadata_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
