"""Generate the deterministic synthetic weather dataset (SIH26078, Phase 0).

Usage (from the repository root)::

    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --seed 7 --time-steps 48
    python scripts/generate_sample_data.py --no-event --output data/samples/quiet.nc

The script writes ``data/raw/sample_history.nc`` plus a provenance manifest
(``data/raw/sample_history.manifest.json``). No API keys and no network access
are required - the data is fully synthetic.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make `src/` importable when the package is not installed (repo-local runs).
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from extreme_weather.config import PROJECT_ID, PROJECT_NAME, get_settings  # noqa: E402
from extreme_weather.data.generator import (  # noqa: E402
    DATASET_BASENAME,
    NETCDF_SUFFIX,
    default_config,
    generate_and_save,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for the generator script."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} ({PROJECT_ID}) - synthetic weather data generator.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Destination NetCDF path (default: <DATA_DIR>/raw/sample_history.nc).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility.")
    parser.add_argument("--time-steps", type=int, default=None, help="Number of time steps.")
    parser.add_argument(
        "--resolution",
        type=float,
        default=None,
        help="Grid resolution in degrees (0.125 deg ~= 12 km NWP native).",
    )
    parser.add_argument(
        "--latitude-bounds",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Latitude domain bounds in degrees north.",
    )
    parser.add_argument(
        "--longitude-bounds",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=None,
        help="Longitude domain bounds in degrees east.",
    )
    parser.add_argument(
        "--no-event",
        action="store_true",
        help="Disable the injected synthetic extreme rainfall event.",
    )
    parser.add_argument(
        "--peak-mm-per-hour",
        type=float,
        default=None,
        help="Peak precipitation rate of the injected event.",
    )
    parser.add_argument(
        "--no-manifest",
        action="store_true",
        help="Do not write the JSON provenance manifest.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: build the configuration, generate data, print a summary."""
    args = build_parser().parse_args(argv)
    settings = get_settings()

    event_overrides: dict[str, float] = {}
    if args.peak_mm_per_hour is not None:
        event_overrides["peak_precipitation_mm_per_hour"] = args.peak_mm_per_hour

    config = default_config(
        seed=args.seed,
        time_steps=args.time_steps,
        resolution_deg=args.resolution,
        latitude_bounds=tuple(args.latitude_bounds) if args.latitude_bounds else None,
        longitude_bounds=tuple(args.longitude_bounds) if args.longitude_bounds else None,
        event_enabled=False if args.no_event else None,
        event_overrides=event_overrides or None,
        settings=settings,
    )

    output_path = args.output or settings.raw_dir / f"{DATASET_BASENAME}{NETCDF_SUFFIX}"
    result = generate_and_save(
        output_path,
        config=config,
        write_manifest_file=not args.no_manifest,
    )

    dataset = result.dataset
    print(f"{PROJECT_NAME} ({PROJECT_ID}) - synthetic dataset generated")
    print(f"  file            : {result.dataset_path}")
    print(f"  manifest        : {result.manifest_path}")
    print(f"  seed            : {config.seed}")
    print(f"  dimensions      : {dict(dataset.sizes)}")
    print(f"  variables       : {', '.join(dataset.data_vars)}")
    print(f"  resolution      : {config.resolution_deg} deg (~{config.resolution_deg * 111.32:.2f} km)")
    print(f"  time            : {result.manifest['coordinates']['time']['start']} .. "
          f"{result.manifest['coordinates']['time']['end']}")
    if config.event.enabled:
        event = config.event
        print(
            "  injected event  : "
            f"{event.name} @ ({event.center_latitude}, {event.center_longitude}), "
            f"steps {event.start_index}..{event.start_index + event.duration_steps - 1}, "
            f"peak {event.peak_precipitation_mm_per_hour} mm/h"
        )
        peak = float(dataset["precipitation"].max())
        print(f"  peak precip     : {peak:.2f} mm/h")
    else:
        print("  injected event  : disabled")
    print("  note            : SYNTHETIC data - not observations, not an NWP product")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
