"""Ingest a real or historical weather dataset (SIH26078, Phase 6).

Usage (from the repository root)::

    python scripts/ingest_weather.py --file <path>
    python scripts/ingest_weather.py --file <path> --validate
    python scripts/ingest_weather.py --file <path> --process
    python scripts/ingest_weather.py --file <path> --origin SYNTHETIC --dry-run

No API key is required for local-file ingestion.

The pipeline is::

    file -> fetch -> load (source units) -> quality -> normalize
         -> manifest -> registry -> (optional) Phase 0 preprocessing

**``--validate`` stops after the quality report** and writes nothing, so a file
can be inspected before anything is committed to the data directory.

**``--process`` additionally runs the existing Phase 0 preprocessing pipeline**
(``extreme_weather.data.preprocessing.preprocess_file``) so the ingested
dataset reaches the same processed format the synthetic pipeline produces. That
is what makes the file usable by Phase 1-5 unchanged.

Origins are never guessed: ``--origin`` defaults to ``HISTORICAL`` and the
value decides whether the file is written under ``data/raw/real`` or
``data/raw/synthetic``. A real file presented as synthetic is the exact mistake
this flag exists to prevent.
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
from extreme_weather.ingestion import (  # noqa: E402
    DataOrigin,
    IngestionError,
    IngestionRequest,
    IngestionService,
    OpenMeteoRequest,
    SourceNotFoundError,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for the ingestion script."""
    parser = argparse.ArgumentParser(
        description=(
            f"{PROJECT_NAME} ({PROJECT_ID}) - real / historical weather data ingestion."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--file", type=Path, default=None, help="Local NetCDF or GRIB file to ingest."
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help="Stable id to register the dataset under (default: the file stem).",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Registered source to read with (netcdf, grib, nwp).",
    )
    parser.add_argument(
        "--origin",
        type=str,
        default=DataOrigin.HISTORICAL,
        choices=list(DataOrigin.ALL),
        help=(
            "Provenance class. SYNTHETIC for data this project generated, "
            "HISTORICAL for a real past period, EXTERNAL for a real fetched source."
        ),
    )
    parser.add_argument(
        "--variables",
        type=str,
        default=None,
        help=(
            "Comma-separated canonical variables to keep. Default: the configured "
            "whitelist (all six core variables)."
        ),
    )
    parser.add_argument(
        "--lat-bounds", type=str, default=None, help="Latitude box as 'min,max'."
    )
    parser.add_argument(
        "--lon-bounds", type=str, default=None, help="Longitude box as 'min,max'."
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Report quality and stop. Writes nothing.",
    )
    parser.add_argument(
        "--process",
        action="store_true",
        help="Also run Phase 0 preprocessing so Phase 1-5 can consume the file.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the whole chain but write no file and register nothing.",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Register the dataset even when the quality report has errors.",
    )
    parser.add_argument(
        "--fetch-era5",
        action="store_true",
        help=(
            "Download an ERA5 subset instead of reading a local file. Requires "
            "--era5-lats, --era5-lons, --start and --end."
        ),
    )
    parser.add_argument(
        "--era5-lats", type=str, default=None, help="Comma-separated latitudes for ERA5."
    )
    parser.add_argument(
        "--era5-lons", type=str, default=None, help="Comma-separated longitudes for ERA5."
    )
    parser.add_argument("--start", type=str, default=None, help="ERA5 start date (YYYY-MM-DD).")
    parser.add_argument("--end", type=str, default=None, help="ERA5 end date (YYYY-MM-DD).")
    return parser


def _parse_floats(value: str | None) -> tuple[float, ...] | None:
    """Parse a comma-separated float list, or return ``None``."""
    if not value:
        return None
    return tuple(float(part) for part in str(value).split(",") if part.strip())


def _parse_bounds(value: str | None) -> tuple[float, float] | None:
    """Parse a ``"min,max"`` bounds string, or return ``None``."""
    if not value:
        return None
    parts = [part.strip() for part in str(value).split(",")]
    if len(parts) != 2:
        raise ValueError(f"Expected bounds as 'min,max', got '{value}'.")
    low, high = float(parts[0]), float(parts[1])
    if low > high:
        raise ValueError(f"Bounds minimum {low} exceeds maximum {high}.")
    return (low, high)


#: The ERA5 hourly fields fetched when ``--fetch-era5`` does not name any. The
#: full Phase 0 contract, so the result can enter the pipeline unchanged.
ERA5_DEFAULT_VARIABLES: tuple[str, ...] = (
    "precipitation",
    "temperature_2m",
    "surface_pressure",
    "relative_humidity_2m",
    "wind_speed_10m",
    "wind_direction_10m",
)


def fetch_era5(service: IngestionService, args: argparse.Namespace) -> Path:
    """Download a bounded ERA5 subset and return the cached NetCDF path."""
    from extreme_weather.ingestion import ERA5Source

    missing = [
        name
        for name, value in (
            ("--era5-lats", args.era5_lats),
            ("--era5-lons", args.era5_lons),
            ("--start", args.start),
            ("--end", args.end),
        )
        if not value
    ]
    if missing:
        raise ValueError(
            f"--fetch-era5 also needs {', '.join(missing)}. A bounded request is "
            "required: an unbounded archive request would be a multi-gigabyte "
            "download."
        )

    # The default request asks for the full contract. Asking for less would
    # produce a dataset Phase 0 rejects, and the caller would have no way to
    # know that a *default* was the reason.
    request = OpenMeteoRequest(
        latitudes=_parse_floats(args.era5_lats) or (),
        longitudes=_parse_floats(args.era5_lons) or (),
        start_date=str(args.start),
        end_date=str(args.end),
        variables=(
            tuple(
                part.strip()
                for part in (args.variables or "").split(",")
                if part.strip()
            )
            or ERA5_DEFAULT_VARIABLES
        ),
    )
    source = service.get_source("era5")
    dataset_id = args.dataset_id or "era5_subset"
    print(
        f"Fetching ERA5: {request.point_count()} point(s), "
        f"{request.start_date} .. {request.end_date}"
    )
    return source.fetch(request, dataset_id=dataset_id)  # type: ignore[attr-defined]


def report_quality(result) -> None:
    """Print the quality report in a readable block."""
    report = result.quality
    print("\nQuality report")
    print("--------------")
    print(f"  valid            : {report.valid}")
    print(f"  errors           : {report.error_count}")
    print(f"  warnings         : {report.warning_count}")
    print(f"  missing fraction : {report.missing_fraction:.6f}")
    print(f"  time steps       : {report.time_steps}")
    print(f"  spatial points   : {report.spatial_points}")
    print(f"  variables        : {', '.join(report.variables)}")
    if report.duplicate_timestamps:
        print(f"  duplicate stamps : {report.duplicate_timestamps}")
    if report.duplicate_coordinates:
        print(f"  duplicate coords : {report.duplicate_coordinates}")
    if report.missing_standard_variables:
        print(
            "  not present      : "
            + ", ".join(report.missing_standard_variables)
            + " (reported, not invented)"
        )
    for error in report.errors:
        print(f"  ERROR: {error}")
    for warning in report.warnings:
        print(f"  WARN : {warning}")


def report_manifest(result) -> None:
    """Print the manifest summary."""
    manifest = result.manifest
    print("\nManifest")
    print("--------")
    print(f"  dataset id   : {manifest.dataset_id}")
    print(f"  source       : {manifest.source}")
    print(f"  source url   : {manifest.source_url or 'none (local file)'}")
    print(f"  origin       : {manifest.origin}")
    print(f"  is synthetic : {manifest.is_synthetic}")
    print(f"  is live      : {manifest.is_live}")
    print(f"  format       : {manifest.format}")
    print(f"  file         : {manifest.file_name or 'n/a'}")
    print(f"  file hash    : {manifest.file_hash[:16]}... (sha256)")
    print(f"  content hash : {(manifest.content_hash or '')[:16]}... (sha256)")
    print(f"  variables    : {', '.join(manifest.variables)}")
    if manifest.variables_missing:
        print(f"  not present  : {', '.join(manifest.variables_missing)}")
    print(f"  resolution   : {manifest.resolution_deg} deg (~{manifest.resolution_km} km)")
    print(f"  time range   : {manifest.time_start} .. {manifest.time_end}")
    print(f"  units        : {manifest.units}")
    for variable, record in sorted(manifest.unit_conversions.items()):
        print(
            f"    {variable}: {record.get('original_units')} -> "
            f"{record.get('normalized_units')} ({record.get('conversion_method')})"
        )
    for action in manifest.coordinate_actions:
        print(f"    coordinate: {action}")
    print("  provenance   :")
    for step in manifest.processing_steps:
        print(
            f"    {step.stage:<11} {step.tool} "
            f"(in={str(step.input_hash)[:12]} out={str(step.output_hash)[:12]})"
        )
    quality = manifest.quality_summary
    if quality:
        print(
            f"  quality      : valid={quality.get('valid')} "
            f"errors={quality.get('error_count')} warnings={quality.get('warning_count')}"
        )



def run_preprocessing(dataset_path: Path) -> Path:
    """Run the existing Phase 0 preprocessing over an ingested dataset.

    This is what makes an ingested file indistinguishable from a generated one
    for the Phase 1-5 pipeline: same validation, same missing-value handling,
    same derived variables, same processed output format.
    """
    from extreme_weather.data.preprocessing import preprocess_file

    settings = get_settings()
    target = (
        settings.processed_dir
        / f"{dataset_path.stem}{settings.processed_real_suffix}"
    )
    return preprocess_file(dataset_path, target)


def main(argv: list[str] | None = None) -> int:
    """Entry point: ingest, validate, normalize, manifest, register."""
    args = build_parser().parse_args(argv)
    settings = get_settings()
    service = IngestionService()

    if args.file is None and not args.fetch_era5:
        print(
            "ERROR: supply --file <path> (local ingestion, no API key needed) or "
            "--fetch-era5 with a bounded box.",
            file=sys.stderr,
        )
        return 2

    source_path: Path
    if args.fetch_era5:
        try:
            source_path = fetch_era5(service, args)
        except (IngestionError, ValueError) as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 2
    else:
        source_path = args.file

    dataset_id = args.dataset_id or source_path.stem
    source_name = args.source or (
        "grib" if source_path.suffix.lower() in (".grib", ".grib2", ".grb", ".grb2") else "netcdf"
    )

    try:
        variables = tuple(
            part.strip() for part in (args.variables or "").split(",") if part.strip()
        )
        request = IngestionRequest(
            dataset_id=dataset_id,
            origin=args.origin,
            source_name=source_name,
            path=source_path,
            variables=variables,
            latitude=_parse_bounds(args.lat_bounds),
            longitude=_parse_bounds(args.lon_bounds),
            strict=not args.no_strict,
            # --validate is a read-only preview: nothing is written.
            dry_run=args.dry_run or args.validate,
        )
        result = service.ingest(request)
    except (IngestionError, SourceNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print(f"{PROJECT_NAME} ({PROJECT_ID}) - weather data ingestion (Phase 6)")
    print(f"  source file : {source_path}")
    print(f"  dataset id  : {dataset_id}")
    print(f"  source      : {source_name}")
    print(f"  origin      : {args.origin}")
    print(f"  validate    : {args.validate}")
    print(f"  dry run     : {args.dry_run or args.validate}")
    print(f"  process     : {args.process}")
    print(f"  manifests   : {settings.manifests_dir}")

    report_quality(result)
    report_manifest(result)

    print("\nSteps")
    print("-----")
    for step in result.steps:
        print(f"  {step}")
    for warning in result.warnings:
        print(f"  NOTE: {warning}")
    for note in result.degraded:
        print(f"  NOTE: {note}")

    if args.validate:
        print("\n  --validate: nothing was written.")
        return 0 if result.quality.valid else 1

    if result.output_path is not None and args.process:
        try:
            processed = run_preprocessing(result.output_path)
            print(f"\n  processed   : {processed}")
            print("  Phase 1-5 can now read the processed file unchanged.")
        except Exception as exc:  # noqa: BLE001 - reported, never fatal to the ingest
            print(f"\n  preprocessing failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            return 1

    print(
        "\n  Next: python scripts/detect_anomalies.py (Phase 1) then "
        "python scripts/track_events.py (Phase 2)."
    )
    print(
        "  Note: this is a research prototype on real historical data. It is not an "
        "operational forecast and the results are not verified against warnings."
    )
    return 0 if result.quality.valid else 1


if __name__ == "__main__":
    raise SystemExit(main())

