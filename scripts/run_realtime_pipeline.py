#!/usr/bin/env python3
"""Run the near-real-time weather pipeline (SIH26078, Phase 8).

This is the command-line entry point for the Phase 8 ingestion and
orchestration layer. It drives the same
:class:`~extreme_weather.realtime.orchestrator.PipelineOrchestrator` the API
uses, so a CLI run and an API run produce identical results.

Examples
--------
Run once with the deterministic fixture provider (no network, no API key)::

    python scripts/run_realtime_pipeline.py --once

Plan a run without writing anything::

    python scripts/run_realtime_pipeline.py --dry-run

Run every hour against the ERA5 archive::

    python scripts/run_realtime_pipeline.py --interval-minutes 60 --provider open-meteo

Ingest a file already on disk::

    python scripts/run_realtime_pipeline.py --once --provider local --file data/raw/x.nc

This is a **research prototype**. It is not an operational emergency-warning
system, it carries no SLA, and it demonstrates no forecast skill.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if SRC.is_dir() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from extreme_weather.config import PROJECT_ID, PROJECT_NAME  # noqa: E402
from extreme_weather.realtime.exceptions import (  # noqa: E402
    PipelineConfigurationError,
    ProviderError,
)
from extreme_weather.realtime.freshness import (  # noqa: E402
    DEFAULT_FRESHNESS_THRESHOLDS,
    FreshnessThresholds,
)
from extreme_weather.realtime.logging import get_logger  # noqa: E402
from extreme_weather.realtime.orchestrator import (  # noqa: E402
    STAGE_ORDER,
    PipelineOrchestrator,
)
from extreme_weather.realtime.providers import (  # noqa: E402
    DEFAULT_TIMEOUT_SECONDS,
    WeatherRequest,
    build_provider,
    default_providers,
)
from extreme_weather.realtime.retry import RetryPolicy  # noqa: E402
from extreme_weather.realtime.scheduler import (  # noqa: E402
    RealtimeScheduler,
    SchedulerConfig,
)

UTC = timezone.utc

DEFAULT_VARIABLES = ("precipitation", "temperature")
DEFAULT_LATITUDE = (12.0, 12.5, 13.0)
DEFAULT_LONGITUDE = (76.0, 76.5, 77.0)


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    # The remote provider needs a URL, so it is not in default_providers(),
    # but it must still be advertised here or --provider would look unsupported.
    known = sorted([*default_providers(), "remote-http"])
    parser = argparse.ArgumentParser(
        prog="run_realtime_pipeline.py",
        description=(
            f"{PROJECT_NAME} ({PROJECT_ID}) - near-real-time weather ingestion "
            "and pipeline orchestration (Phase 8). Research prototype: not an "
            "operational emergency-warning system."
        ),
        epilog=(
            "Examples:\n"
            "  %(prog)s --once\n"
            "  %(prog)s --dry-run\n"
            "  %(prog)s --interval-minutes 60 --provider open-meteo\n"
            "  %(prog)s --once --provider local --file data/raw/sample.nc\n"
            "\n"
            "Providers:\n"
            "  local        read a NetCDF/GRIB file from disk (needs --file)\n"
            "  open-meteo   ERA5 reanalysis via the Open-Meteo archive (historical)\n"
            "  remote-http  a configurable HTTPS endpoint (needs --url)\n"
            "  fixture      deterministic synthetic data, no network required\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = parser.add_argument_group("execution mode")
    mode.add_argument(
        "--once", action="store_true", help="Run the pipeline a single time and exit."
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate the configuration and show the planned stages without "
        "writing any artifact.",
    )
    mode.add_argument(
        "--interval-minutes",
        type=int,
        default=0,
        metavar="N",
        help="Run every N minutes instead of once. 0 (the default) runs once.",
    )
    mode.add_argument(
        "--max-runs",
        type=int,
        default=None,
        metavar="N",
        help="Stop after N runs. Only meaningful with --interval-minutes.",
    )

    source = parser.add_argument_group("data source")
    source.add_argument(
        "--provider",
        default="fixture",
        help=f"Provider to use. One of: {', '.join(known)} (default: fixture).",
    )
    source.add_argument(
        "--start",
        default=None,
        help="Window start (ISO 8601). Overrides --hours.",
    )
    source.add_argument(
        "--end",
        default=None,
        help="Window end (ISO 8601). Defaults to now.",
    )
    source.add_argument(
        "--hours",
        type=int,
        default=6,
        help="Window length ending at --end, in hours (default: 6).",
    )
    source.add_argument(
        "--variables",
        default=",".join(DEFAULT_VARIABLES),
        help=f"Comma-separated variables (default: {','.join(DEFAULT_VARIABLES)}).",
    )
    source.add_argument(
        "--lat", default="12.0,12.5,13.0", help="Comma-separated latitudes."
    )
    source.add_argument(
        "--lon", default="76.0,76.5,77.0", help="Comma-separated longitudes."
    )
    source.add_argument("--file", default=None, help="Local file for --provider local.")
    source.add_argument(
        "--url", default=None, help="HTTPS endpoint for --provider remote-http."
    )

    tuning = parser.add_argument_group("reliability and freshness")
    tuning.add_argument(
        "--retries",
        type=int,
        default=3,
        help="Total ingestion attempts, including the first (default: 3).",
    )
    tuning.add_argument(
        "--retry-delay",
        type=float,
        default=1.0,
        help="Base backoff delay in seconds (default: 1.0).",
    )
    tuning.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Network timeout in seconds (default: {DEFAULT_TIMEOUT_SECONDS}).",
    )
    tuning.add_argument(
        "--fresh-hours",
        type=float,
        default=None,
        help="Data older than this is STALE instead of FRESH "
        f"(default: {DEFAULT_FRESHNESS_THRESHOLDS.fresh_within.total_seconds() / 3600:g}).",
    )
    tuning.add_argument(
        "--stale-hours",
        type=float,
        default=None,
        help="Data older than this is EXPIRED "
        f"(default: {DEFAULT_FRESHNESS_THRESHOLDS.stale_within.total_seconds() / 3600:g}).",
    )
    tuning.add_argument(
        "--context-steps",
        type=int,
        default=24,
        help="Hours of history retained for climatology and tracking (default: 24).",
    )
    tuning.add_argument(
        "--force",
        action="store_true",
        help="Reprocess even when this exact ingestion has been seen before.",
    )
    tuning.add_argument(
        "--no-stale",
        action="store_true",
        help="Stop the run when the data is EXPIRED instead of processing it.",
    )

    output = parser.add_argument_group("output")
    output.add_argument(
        "--json", action="store_true", help="Print the run as JSON on stdout."
    )
    output.add_argument(
        "--quiet", action="store_true", help="Only log warnings and errors."
    )
    output.add_argument(
        "--verbose", "-v", action="store_true", help="Log debug detail as well."
    )
    return parser


def _parse_window(args: argparse.Namespace) -> tuple[datetime, datetime]:
    """Resolve the requested time window.

    Returns:
        The ``(start, end)`` pair.

    Raises:
        PipelineConfigurationError: The window is not a valid range.
    """
    end = _parse_moment(args.end) or datetime.now(UTC)
    if args.start:
        start = _parse_moment(args.start)
        if start is None:
            raise PipelineConfigurationError(f"Could not parse --start '{args.start}'.")
    else:
        start = end - timedelta(hours=int(args.hours))
    if end < start:
        raise PipelineConfigurationError(
            f"The window ends ({end.isoformat()}) before it starts ({start.isoformat()})."
        )
    return start, end


def _parse_moment(text: str | None) -> datetime | None:
    """Parse an ISO 8601 timestamp, returning ``None`` when unusable."""
    if not text:
        return None
    candidate = str(text).strip()
    if candidate.endswith("Z"):
        candidate = candidate[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        return None
    return parsed.replace(tzinfo=UTC) if parsed.tzinfo is None else parsed.astimezone(UTC)


def _parse_floats(text: str, label: str) -> tuple[float, ...]:
    """Parse a comma-separated float list."""
    values = tuple(float(p.strip()) for p in str(text).split(",") if p.strip())
    if not values:
        raise PipelineConfigurationError(f"--{label} needs at least one value.")
    return values


def main(argv: list[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        A process exit code: ``0`` on success, ``1`` when the run failed, and
        ``2`` for a configuration error.
    """
    args = build_parser().parse_args(argv)
    level = logging.DEBUG if args.verbose else (logging.WARNING if args.quiet else logging.INFO)
    logging.getLogger("extreme_weather.realtime").setLevel(level)
    logger = get_logger()

    try:
        start, end = _parse_window(args)
        variables = tuple(v.strip() for v in args.variables.split(",") if v.strip())
        provider = build_provider(
            args.provider,
            local_path=args.file,
            remote_url=args.url,
            timeout_seconds=args.timeout,
        )
        request = WeatherRequest(
            start=start,
            end=end,
            variables=variables,
            latitude=_parse_floats(args.lat, "lat"),
            longitude=_parse_floats(args.lon, "lon"),
            timeout_seconds=args.timeout,
        )
        thresholds = FreshnessThresholds(
            fresh_within=timedelta(
                hours=args.fresh_hours
                if args.fresh_hours is not None
                else DEFAULT_FRESHNESS_THRESHOLDS.fresh_within.total_seconds() / 3600
            ),
            stale_within=timedelta(
                hours=args.stale_hours
                if args.stale_hours is not None
                else DEFAULT_FRESHNESS_THRESHOLDS.stale_within.total_seconds() / 3600
            ),
        )
        orchestrator = PipelineOrchestrator(
            provider,
            retry_policy=RetryPolicy(
                max_attempts=int(args.retries), base_delay=float(args.retry_delay)
            ),
            freshness_thresholds=thresholds,
            context_steps=int(args.context_steps),
        )
        orchestrator.load_state()
    except (PipelineConfigurationError, ProviderError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    config = SchedulerConfig(
        interval_minutes=int(args.interval_minutes),
        max_runs=args.max_runs,
        dry_run=bool(args.dry_run),
    )
    scheduler = RealtimeScheduler(orchestrator, lambda: request, config)

    # --- dry run: plan only -------------------------------------------
    if args.dry_run:
        plan = scheduler.dry_run_plan()
        if args.json:
            print(json.dumps(plan, indent=2, default=str))
        else:
            _print_plan(plan)
        return 0

    if not provider.available():
        print(
            f"error: provider '{args.provider}' is not available "
            "(check --file or --url).",
            file=sys.stderr,
        )
        return 2

    # --- run -----------------------------------------------------------
    try:
        ticks = scheduler.run(force=bool(args.force), allow_stale=not args.no_stale)
    except KeyboardInterrupt:
        print("\ninterrupted: the current run finished before stopping.", file=sys.stderr)
        return 130

    if not ticks:
        print("no runs were executed.", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps([t.as_dict() for t in ticks], indent=2, default=str))
    else:
        for tick in ticks:
            _print_tick(tick)

    failed = [t for t in ticks if not t.succeeded]
    if failed:
        print(
            f"\n{len(failed)} of {len(ticks)} run(s) failed. See the stage output above.",
            file=sys.stderr,
        )
        return 1
    return 0


def _print_plan(plan: dict[str, Any]) -> None:
    """Print a dry-run plan in a readable form."""
    provider = plan["provider"]
    request = plan["request"]
    print(f"{PROJECT_NAME} ({PROJECT_ID}) - Phase 8 dry run")
    print("=" * 66)
    print(f"provider      : {provider['name']} ({provider['description']})")
    print(f"available     : {provider['available']}")
    print(f"window        : {request['start']} -> {request['end']} ({request['hours']} h)")
    print(f"variables     : {', '.join(request['variables'])}")
    print(f"grid          : {len(request['latitude'])}x{len(request['longitude'])} points")
    print(f"retry policy  : {plan['retry_policy']['max_attempts']} attempts, "
          f"base delay {plan['retry_policy']['base_delay']}s")
    print(f"freshness     : fresh <= {plan['freshness_thresholds']['fresh_within_hours']}h, "
          f"stale <= {plan['freshness_thresholds']['stale_within_hours']}h")
    print(f"context steps : {plan['context_steps']}")
    print("planned stages:")
    for index, name in enumerate(plan["planned_stages"], start=1):
        note = " (optional)" if name in ("gnn", "downscale") else ""
        print(f"  {index}. {name}{note}")
    print(f"\nartifacts written: {plan['would_write']}")
    print(plan["note"])
    print("Research prototype. Not an operational emergency-warning system.")


def _print_tick(tick: Any) -> None:
    """Print one scheduled run."""
    if tick.run is None:
        print(f"run failed to start: {tick.error}", file=sys.stderr)
        return
    run = tick.run
    print()
    print(f"run {run.run_id}  status={run.status}  freshness={run.freshness}")
    if run.source_timestamp:
        age = f" ({run.age_minutes:.0f} min old)" if run.age_minutes is not None else ""
        print(f"  latest data: {run.source_timestamp}{age}")
    for stage in run.stages:
        duration = f"{stage.duration:.2f}s" if stage.duration is not None else "-"
        print(f"  [{stage.status:<7}] {stage.name:<10} {duration:>8}  {stage.message}")
    for warning in run.warnings:
        print(f"  warning: {warning}")
    for error in run.errors:
        print(f"  error:   {error}")
    if run.dry_run:
        print("  (dry run: no artifacts were written)")


if __name__ == "__main__":
    raise SystemExit(main())
