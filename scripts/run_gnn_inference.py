"""Run a GNN next-state prediction for one Phase 2 track (SIH26078, Phase 3).

```text
models/gnn/best.json  +  data/processed/tracks.json
        ↓  GNNService.predict(track_id)
GNNPrediction  (prototype)
```

Examples::

    python scripts/run_gnn_inference.py --track-id TRK-EXTREME_RAINFALL-20240602T0200-001
    python scripts/run_gnn_inference.py --list-tracks
    python scripts/run_gnn_inference.py --track-id <id> --json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from extreme_weather.gnn.checkpoint import checkpoint_path  # noqa: E402
from extreme_weather.gnn.service import (  # noqa: E402
    GNNInferenceError,
    GNNInvalidTrackError,
    GNNModelUnavailableError,
    GNNService,
    GNNUnknownTrackError,
)

PROTOTYPE_BANNER = (
    "GNN prototype prediction - NOT an operational forecast. The model was trained on "
    "synthetic data and has not been validated against observed weather."
)


def build_parser() -> argparse.ArgumentParser:
    """Command-line interface of the inference script."""
    default_checkpoint = checkpoint_path()
    parser = argparse.ArgumentParser(
        description="DeepWeather AI (SIH26078) - GNN prototype next-state prediction."
    )
    parser.add_argument(
        "--track-id", default=None, help="Phase 2 track id to predict the next state of."
    )
    parser.add_argument(
        "--checkpoint", default=None, help=f"Checkpoint file (default: {default_checkpoint})."
    )
    parser.add_argument(
        "--list-tracks", action="store_true", help="List the available track ids and exit."
    )
    parser.add_argument("--json", action="store_true", help="Print the prediction as JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Predict the next state of a track and print it."""
    args = build_parser().parse_args(argv)
    service = GNNService(checkpoint_path=args.checkpoint)

    if args.list_tracks:
        try:
            tracks = service._load_tracks()  # noqa: SLF001 - deliberate CLI introspection
        except GNNInvalidTrackError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        if not tracks:
            print("No tracks available.")
            return 0
        print("Available tracks")
        print("----------------")
        for track in tracks:
            print(
                f"  {track.track_id}  {track.event_type.value}  {track.point_count} points  "
                f"{track.duration_hours:.2f} h"
            )
        return 0

    if not args.track_id:
        print("error: --track-id is required (or use --list-tracks).", file=sys.stderr)
        return 2

    try:
        prediction = service.predict(args.track_id)
    except GNNModelUnavailableError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except GNNUnknownTrackError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except (GNNInvalidTrackError, GNNInferenceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(prediction.model_dump(mode="json"), indent=2))
        return 0

    print(PROTOTYPE_BANNER)
    print()
    print("Track")
    print(prediction.track_id)
    print()
    print("Current")
    print(f"  time     : {prediction.timestamp}")
    print(f"  lat      : {prediction.current_latitude:.6f}")
    print(f"  lon      : {prediction.current_longitude:.6f}")
    print(f"  intensity: {prediction.current_intensity:.4f}")
    print()
    print("Predicted next")
    print(f"  lat      : {prediction.predicted_latitude:.6f}")
    print(f"  lon      : {prediction.predicted_longitude:.6f}")
    print(f"  intensity: {prediction.predicted_intensity:.4f}")
    print()
    print("Prediction (delta)")
    print(f"  delta lat       : {prediction.delta_latitude:+.6f} deg")
    print(f"  delta lon       : {prediction.delta_longitude:+.6f} deg")
    print(f"  delta intensity : {prediction.delta_intensity:+.4f}")
    print(f"  implied speed   : {prediction.predicted_speed_kmh:.4f} km/h")
    print(f"  direction       : {prediction.predicted_direction}")
    print()
    print("Model")
    print(f"  version         : {prediction.model_version}")
    print(f"  node index      : {prediction.node_index}")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
