"""Detect extreme weather events in a NetCDF dataset (SIH26078, Phase 1).

Usage (from the repository root)::

    python scripts/detect_anomalies.py
    python scripts/detect_anomalies.py --input data/processed/sample_history_processed.nc
    python scripts/detect_anomalies.py --reference data/raw/sample_history.nc --score efi
    python scripts/detect_anomalies.py --variables precipitation temperature
    python scripts/detect_anomalies.py --figure data/processed/anomaly.png

The analysis is a **prototype** run on synthetic data with prototype thresholds:
it is not a real-time forecast and must not be used for warnings.
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

from extreme_weather.anomaly import (  # noqa: E402
    PROTOTYPE_THRESHOLD_WARNING,
    AnomalyDetectionService,
    EventStore,
    build_climatology,
    compute_score,
    default_thresholds,
    detect_extreme_cells,
)
from extreme_weather.config import PROJECT_ID, PROJECT_NAME, get_settings  # noqa: E402
from extreme_weather.data.generator import DATASET_BASENAME, NETCDF_SUFFIX  # noqa: E402
from extreme_weather.data.loader import load_dataset  # noqa: E402
from extreme_weather.data.preprocessing import (  # noqa: E402
    DEFAULT_PROCESSED_SUFFIX,
    PROCESSED_DATASET_BASENAME,
)
from extreme_weather.schemas.event import ExtremeWeatherEvent  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for the anomaly detection script."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} ({PROJECT_ID}) - extreme weather anomaly detection.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Dataset to analyse (default: <DATA_DIR>/processed/<name>_processed.nc).",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference dataset for the climatology (default: the input dataset).",
    )
    parser.add_argument(
        "--variables",
        nargs="+",
        default=["precipitation"],
        help="Variables to analyse (each needs a threshold ladder).",
    )
    parser.add_argument(
        "--score",
        default="absolute",
        help="Anomaly score: absolute, standardized, percentile, extremeness or efi.",
    )
    parser.add_argument(
        "--grouping",
        default="month",
        choices=("month", "day_of_year", "all"),
        help="Climatology grouping.",
    )
    parser.add_argument(
        "--min-cells", type=int, default=1, help="Minimum cluster size kept as an event."
    )
    parser.add_argument(
        "--events-output",
        type=Path,
        default=None,
        help="JSON output path (default: <DATA_DIR>/processed/events.json).",
    )
    parser.add_argument(
        "--no-anomaly-netcdf",
        action="store_true",
        help="Do not write anomaly.nc next to the events file.",
    )
    parser.add_argument(
        "--figure",
        type=Path,
        default=None,
        help="Optional PNG path for the four-panel overview figure.",
    )
    parser.add_argument(
        "--max-print", type=int, default=20, help="Maximum number of events printed."
    )
    return parser


def format_event(event: ExtremeWeatherEvent) -> str:
    """Return the human readable block for one event."""
    units = f" {event.units}" if event.units else ""
    return "\n".join(
        [
            f"Event ID: {event.event_id}",
            f"Type: {event.event_type.value}",
            f"Severity: {event.severity.value}",
            f"Location: {event.latitude:.3f} N, {event.longitude:.3f} E",
            f"Time: {event.timestamp.isoformat()}",
            f"Intensity: {event.intensity:.2f}{units}",
            f"Anomaly score: {event.anomaly_score:.3f}",
            f"Radius: {event.affected_radius_km:.1f} km",
            f"Cells: {event.cell_count}",
        ]
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the analysis, print the events and store the results."""
    args = build_parser().parse_args(argv)
    settings = get_settings()

    input_path = args.input or (
        settings.processed_dir
        / f"{PROCESSED_DATASET_BASENAME}{DEFAULT_PROCESSED_SUFFIX}{NETCDF_SUFFIX}"
    )
    if not Path(input_path).exists():
        fallback = settings.raw_dir / f"{DATASET_BASENAME}{NETCDF_SUFFIX}"
        print(
            f"ERROR: input dataset not found: {input_path}\n"
            f"       (the raw dataset is at {fallback}; run "
            "'python scripts/generate_sample_data.py' if that is missing too)",
            file=sys.stderr,
        )
        return 2

    store = EventStore(args.events_output) if args.events_output else EventStore()
    service = AnomalyDetectionService(store=store, thresholds=default_thresholds())

    try:
        result = service.run(
            input_path,
            reference_path=args.reference,
            variables=tuple(args.variables),
            score_name=args.score,
            grouping=args.grouping,
            min_cells=args.min_cells,
            write_outputs=True,
            keep_fields=not args.no_anomaly_netcdf,
        )
    except Exception as exc:  # noqa: BLE001 - CLI boundary, reported to the user
        print(f"ERROR: anomaly analysis failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    print(f"{PROJECT_NAME} ({PROJECT_ID}) - extreme weather anomaly detection (prototype)")
    print(f"  input         : {input_path}")
    print(f"  reference     : {args.reference or 'self (demo shortcut)'}")
    print(f"  variables     : {', '.join(result.variables)}")
    print(f"  score         : {result.score_name}   grouping: {result.grouping}")
    print(f"  thresholds    : {PROTOTYPE_THRESHOLD_WARNING}")

    print("\nDetected extreme weather events:")
    print("--------------------------------")
    if result.event_count == 0:
        print("No extreme weather events detected with the current thresholds.")
    else:
        for event in result.events[: max(args.max_print, 0)]:
            print(format_event(event))
            print()
        if result.event_count > args.max_print:
            print(f"... {result.event_count - args.max_print} more event(s) not printed")

    severity_counts = result.metadata.get("severity_counts", {})
    if severity_counts:
        print(
            "\n  by severity   : "
            + ", ".join(f"{name}={count}" for name, count in severity_counts.items())
        )

    print(f"\n  events json   : {store.path}")
    anomaly_path = store.path.parent / "anomaly.nc"
    if not args.no_anomaly_netcdf and anomaly_path.is_file():
        print(f"  anomaly netcdf: {anomaly_path}")

    if args.figure:
        _write_figure(args.figure, input_path, args.reference, result, service)

    print("\n  note          : prototype analysis on synthetic data; not an operational forecast.")
    return 0


def _write_figure(
    figure_path: Path,
    input_path: Path,
    reference_path: Path | None,
    result,
    service: AnomalyDetectionService,
) -> None:
    """Render the optional overview figure (never fatal)."""
    from extreme_weather.anomaly.visualization import VisualizationError, plot_analysis_overview

    variable = result.variables[0]
    dataset = load_dataset(input_path, validate=False)
    reference = load_dataset(reference_path, validate=False) if reference_path else dataset
    climatology = build_climatology(reference, variable, result.grouping)
    score = compute_score(result.score_name, dataset[variable], climatology)
    ladder = service.thresholds.for_variable(variable)
    mask = detect_extreme_cells(score, ladder, event_type=ladder.event_type)
    try:
        written = plot_analysis_overview(
            dataset[variable], score, mask, result.events, output_path=figure_path
        )
    except VisualizationError as exc:
        print(f"  figure        : skipped ({exc})", file=sys.stderr)
    else:
        print(f"  figure        : {written}")


if __name__ == "__main__":
    raise SystemExit(main())