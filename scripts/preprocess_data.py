"""Run the preprocessing pipeline on a NetCDF dataset (SIH26078, Phase 0).

Usage (from the repository root)::

    python scripts/preprocess_data.py
    python scripts/preprocess_data.py --input data/raw/sample_history.nc --max-gap 2

Pipeline: ``load -> validate -> missing-value handling -> derived variables -> save``.
The raw input file is never modified.
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
from extreme_weather.data.generator import DATASET_BASENAME, NETCDF_SUFFIX  # noqa: E402
from extreme_weather.data.preprocessing import (  # noqa: E402
    DEFAULT_MAX_GAP,
    default_processed_path,
    preprocess_file,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for the preprocessing script."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} ({PROJECT_ID}) - preprocessing pipeline runner.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Input NetCDF file (default: <DATA_DIR>/raw/sample_history.nc).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output NetCDF file (default: <DATA_DIR>/processed/<name>_processed.nc).",
    )
    parser.add_argument(
        "--max-gap",
        type=int,
        default=DEFAULT_MAX_GAP,
        help="Maximum interior gap length (in time steps) that may be filled. Use -1 for unlimited.",
    )
    parser.add_argument(
        "--fill-dim",
        type=str,
        default="time",
        help="Dimension along which missing values are interpolated.",
    )
    parser.add_argument(
        "--no-wind-direction",
        action="store_true",
        help="Only derive wind_speed, not wind_direction.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: preprocess the dataset and print a summary report."""
    args = build_parser().parse_args(argv)
    settings = get_settings()

    input_path = args.input or settings.raw_dir / f"{DATASET_BASENAME}{NETCDF_SUFFIX}"
    max_gap = None if args.max_gap is not None and args.max_gap < 0 else args.max_gap

    if not Path(input_path).exists():
        print(f"ERROR: input dataset not found: {input_path}", file=sys.stderr)
        print("       Run 'python scripts/generate_sample_data.py' first.", file=sys.stderr)
        return 2

    result = preprocess_file(
        input_path,
        args.output or default_processed_path(input_path),
        max_gap=max_gap,
        fill_dim=args.fill_dim,
        add_wind_direction=not args.no_wind_direction,
    )
    report = result.report

    print(f"{PROJECT_NAME} ({PROJECT_ID}) - preprocessing complete")
    print(f"  input            : {report.input_path}")
    print(f"  output           : {report.output_path}")
    print(f"  max_gap          : {'unlimited' if report.max_gap is None else report.max_gap}")
    print(f"  variables        : {', '.join(report.variables)}")
    print(f"  derived          : {', '.join(report.derived_variables)}")
    print(f"  missing before   : {report.total_missing_before}")
    print(f"  missing after    : {report.total_missing_after}")
    print(f"  values filled    : {report.values_filled}")
    print(f"  output variables : {', '.join(result.dataset.data_vars)}")
    print(f"  output dims      : {dict(result.dataset.sizes)}")
    for warning in report.warnings:
        print(f"  warning          : {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
