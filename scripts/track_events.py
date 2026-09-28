"""Track Phase 1 extreme weather events into continuous trajectories (SIH26078).

Usage (from the repository root)::

    python scripts/track_events.py
    python scripts/track_events.py --events data/processed/events.json
    python scripts/track_events.py --max-distance-km 150 --max-speed-kmh 80
    python scripts/track_events.py --tracks-output data/processed/tracks.json

Pipeline: ``events.json -> TrackingService -> tracks.json``.

The tracking step is a **prototype**: it links synthetic detections with a simple,
explainable cost function. It is not an operational forecast and never touches
``data/raw``.
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
from extreme_weather.schemas.track import EventTrack, TrackStatus  # noqa: E402
from extreme_weather.tracking import (  # noqa: E402
    DATA_DISCLAIMER,
    MatchingConfig,
    TrackingError,
    TrackingService,
)
from extreme_weather.tracking.models import TrackResult  # noqa: E402

STATUS_ORDER: tuple[TrackStatus, ...] = (
    TrackStatus.ACTIVE,
    TrackStatus.NEW,
    TrackStatus.ENDED,
)


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for the tracking script."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} ({PROJECT_ID}) - extreme weather event tracking.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help="Phase 1 events file (default: <DATA_DIR>/processed/events.json).",
    )
    parser.add_argument(
        "--tracks-output",
        type=Path,
        default=None,
        help="Tracks JSON output (default: <DATA_DIR>/processed/tracks.json).",
    )
    parser.add_argument(
        "--max-distance-km",
        type=float,
        default=MatchingConfig().max_distance_km,
        help="Reject matches further apart than this.",
    )
    parser.add_argument(
        "--max-speed-kmh",
        type=float,
        default=MatchingConfig().max_speed_kmh,
        help="Reject matches implying a faster movement than this.",
    )
    parser.add_argument(
        "--max-time-gap-hours",
        type=float,
        default=MatchingConfig().max_time_gap_hours,
        help="Reject matches separated by more time than this.",
    )
    parser.add_argument(
        "--distance-weight",
        type=float,
        default=MatchingConfig().distance_weight,
        help="Weight of the normalised distance term in the matching cost.",
    )
    parser.add_argument(
        "--intensity-weight",
        type=float,
        default=MatchingConfig().intensity_weight,
        help="Weight of the intensity-similarity term in the matching cost.",
    )
    parser.add_argument(
        "--no-write", action="store_true", help="Do not write the tracks JSON file."
    )
    parser.add_argument(
        "--max-print", type=int, default=10, help="Maximum number of tracks printed."
    )
    return parser


def format_track(track: EventTrack) -> str:
    """Return the human readable block for one track."""
    start = track.start_time.isoformat()
    end = track.end_time.isoformat()
    latest = track.points[-1]
    return "\n".join(
        [
            f"Track ID: {track.track_id}",
            f"Type: {track.event_type.value}",
            f"Status: {track.status.value}",
            f"Start: {start}",
            f"End: {end}",
            f"Duration: {track.duration_hours:.2f} h ({track.point_count} points)",
            f"Peak intensity: {track.peak_intensity:.2f} {track.units or ''}".rstrip(),
            f"Max speed: {track.max_speed_kmh:.2f} km/h",
            f"Mean speed: {track.mean_speed_kmh:.2f} km/h",
            f"Distance: {track.distance_travelled_km:.2f} km",
            f"Direction: {track.direction}"
            + (
                f" ({track.bearing_degrees:.1f} deg)" if track.bearing_degrees is not None else ""
            ),
            f"Intensity trend: {track.intensity_trend}",
            f"Latest location: {latest.latitude:.3f} N, {latest.longitude:.3f} E",
        ]
    )


def summarise(result: TrackResult) -> None:
    """Print the aggregate summary requested by the CLI contract."""
    longest = result.longest()
    fastest = result.fastest()

    print("\nSummary")
    print("-------")
    print(f"  Events                : {result.event_count}")
    print(f"  Tracks                : {result.track_count}")
    print(f"  Events in tracks      : {result.matched_event_count}")
    print(f"  Total tracked time    : {result.duration_hours:.2f} h")
    counts = {status.value: len(result.by_status(status)) for status in STATUS_ORDER}
    print(
        "  By status             : "
        + ", ".join(f"{name}={count}" for name, count in counts.items())
    )

    if longest is None:
        print("  Longest track         : none (no events to track)")
    else:
        print(
            f"  Longest track         : {longest.track_id} "
            f"({longest.duration_hours:.2f} h, {longest.point_count} points, "
            f"peak {longest.peak_intensity:.2f} {longest.units or ''})".rstrip()
        )
        print(f"    direction           : {longest.direction}")
        print(f"    intensity trend     : {longest.intensity_trend}")

    if fastest is None:
        print("  Maximum speed         : none")
    else:
        print(
            f"  Maximum speed         : {fastest.max_speed_kmh:.2f} km/h "
            f"({fastest.track_id}, direction {fastest.direction})"
        )


def main(argv: list[str] | None = None) -> int:
    """Entry point: load Phase 1 events, track them, print and store results."""
    args = build_parser().parse_args(argv)
    settings = get_settings()

    config = MatchingConfig(
        max_distance_km=args.max_distance_km,
        max_speed_kmh=args.max_speed_kmh,
        max_time_gap_hours=args.max_time_gap_hours,
        distance_weight=args.distance_weight,
        intensity_weight=args.intensity_weight,
    )
    service = TrackingService(config=config)
    if args.tracks_output is not None:
        from extreme_weather.tracking.store import TrackStore

        service.track_store = TrackStore(args.tracks_output)

    print(f"{PROJECT_NAME} ({PROJECT_ID}) - extreme weather event tracking (prototype)")
    print(f"  events file : {args.events or service.event_store.path}")
    print(
        "  matching    : "
        f"max_distance_km={config.max_distance_km}, max_speed_kmh={config.max_speed_kmh}, "
        f"max_time_gap_hours={config.max_time_gap_hours}, "
        f"distance_weight={config.distance_weight}, intensity_weight={config.intensity_weight}"
    )

    try:
        result = service.run(events_path=args.events, write_output=not args.no_write)
    except TrackingError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\nTracked events:")
    print("---------------")
    if result.track_count == 0:
        print("No events available to track.")
    for track in result.tracks[: max(args.max_print, 0)]:
        print(format_track(track))
        print()

    summarise(result)

    if not args.no_write:
        print(f"\n  tracks json  : {service.track_store.path}")
    print(f"\n  note         : {DATA_DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
