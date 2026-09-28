"""Evaluate the Phase 3 GNN on real data against a persistence baseline (Phase 7).

```bash
python scripts/evaluate_gnn_real.py --tracks data/validation/reports/<id>_tracks.json
```

Writes ``data/validation/reports/gnn_validation.json`` containing the dataset,
the split, the sample counts, the checkpoint, the metrics, the baseline metrics,
the comparison and the limitations.

The baseline is persistence: ``predicted_next = current_state``. A GNN that
cannot beat "nothing moves" is adding noise rather than signal, and this script
is built so that outcome is reportable rather than embarrassing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extreme_weather.config import PROJECT_ID, PROJECT_NAME  # noqa: E402
from extreme_weather.validation.metrics import compute_gnn_comparison  # noqa: E402

#: Printed when the split is too small to score, instead of a fabricated score.
NOT_COMPUTED = "Ground truth unavailable - metric not computed."


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} ({PROJECT_ID}) - GNN real-data evaluation."
    )
    parser.add_argument("--tracks", type=Path, required=True, help="Phase 2 tracks file.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "data" / "validation" / "reports" / "gnn_validation.json",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Train on train, select on validation, score the test split once."""
    args = build_parser().parse_args(argv)
    if not args.tracks.is_file():
        print(f"ERROR: no tracks file at '{args.tracks}'.", file=sys.stderr)
        return 2

    import numpy as np
    import torch

    from extreme_weather.gnn.dataset import SplitConfig, build_dataset_from_file
    from extreme_weather.gnn.model import WeatherGNN
    from extreme_weather.gnn.training import TrainingConfig, predict_all, train

    dataset = build_dataset_from_file(args.tracks, split_config=SplitConfig())
    batch = dataset.batch
    test_nodes = [
        index
        for index in dataset.split.test
        if index < len(batch.targets) and batch.targets[index] is not None
    ]
    counts = {
        "train": len(dataset.split.train),
        "validation": len(dataset.split.validation),
        "test": len(test_nodes),
    }

    if not test_nodes:
        payload = {
            "available": False,
            "reason": (
                "The test split contains no node with a known next state, so no "
                "metric was computed."
            ),
            "sample_counts": counts,
        }
    else:
        model = WeatherGNN()
        train(
            batch,
            dataset.split.train,
            dataset.split.validation,
            training_config=TrainingConfig(epochs=args.epochs, seed=7),
        )
        with torch.no_grad():
            predictions = predict_all(model, batch).detach().cpu().numpy()

        targets = np.asarray([batch.targets[i] for i in test_nodes], dtype="float64")
        predicted = np.asarray([predictions[i] for i in test_nodes], dtype="float64")
        zeros = np.zeros_like(targets)
        payload = compute_gnn_comparison(
            predicted_lat=predicted[:, 0],
            predicted_lon=predicted[:, 1],
            predicted_intensity=predicted[:, 2],
            current_lat=zeros[:, 0],
            current_lon=zeros[:, 1],
            current_intensity=zeros[:, 2],
            future_lat=targets[:, 0],
            future_lon=targets[:, 1],
            future_intensity=targets[:, 2],
            dataset=args.tracks.stem,
            split="test",
            model_checkpoint="in-memory (not saved)",
            sample_counts=counts,
            limitations=(
                "Graph targets are per-node deltas, so persistence is the zero "
                "delta ('nothing moves').",
                "The test split was scored exactly once; the checkpoint was "
                "selected on validation loss only.",
                "No confidence interval is claimed for a sample count this small.",
            ),
        ).as_dict()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  report  : {args.output}")
    print(f"  samples : {counts}")
    if payload.get("comparison", {}).get("_summary"):
        print(f"  verdict : {payload['comparison']['_summary']['verdict']}")
    else:
        print(f"  verdict : {payload.get('reason', NOT_COMPUTED)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
