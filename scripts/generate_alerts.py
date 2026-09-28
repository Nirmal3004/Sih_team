"""Generate extreme weather alerts from the Phase 0-4 pipeline (SIH26078).

Usage (from the repository root)::

    python scripts/generate_alerts.py
    python scripts/generate_alerts.py --dry-run
    python scripts/generate_alerts.py --use-gnn
    python scripts/generate_alerts.py --no-notify

Pipeline::

    events.json
        + tracks.json
        + GNN prediction (optional, when a checkpoint exists)
        + AI-downscaled threat (optional, when available)
      -> risk engine (deterministic, explainable)
      -> alert rules (R1-R7)
      -> deduplication (event + level + time bucket)
      -> data/processed/alerts.json
      -> notification adapters (local mock only)

**``--dry-run`` writes nothing**: neither the alert store nor the notification
log is touched, so it is safe to run against a real store.

Every optional stage degrades instead of failing. If the GNN checkpoint or the
downscaling model is missing, alerts are still produced and simply report a
narrower ``source_components`` list - see the ``Degraded inputs`` section.

The severities below are **project prototype categories**, not IMD or WMO
warnings, and the pipeline runs on synthetic data. This is not operational.
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

from extreme_weather.alerts.engine import RISK_DISCLAIMER  # noqa: E402
from extreme_weather.alerts.models import Alert  # noqa: E402
from extreme_weather.alerts.severity import (  # noqa: E402
    SEVERITY_DISCLAIMER,
    AlertSeverity,
)
from extreme_weather.alerts.service import (  # noqa: E402
    AlertError,
    AlertGenerationResult,
    AlertService,
)
from extreme_weather.alerts.store import AlertStore, AlertStoreError  # noqa: E402
from extreme_weather.anomaly.service import EventStore  # noqa: E402
from extreme_weather.config import PROJECT_ID, PROJECT_NAME, get_settings  # noqa: E402
from extreme_weather.tracking.store import TrackStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for the alert generation script."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} ({PROJECT_ID}) - extreme weather alerting (prototype).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alerts-output",
        type=Path,
        default=None,
        help="Alerts JSON output (default: <DATA_DIR>/processed/alerts.json).",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=None,
        help="Phase 1 events file (default: <DATA_DIR>/processed/events.json).",
    )
    parser.add_argument(
        "--tracks",
        type=Path,
        default=None,
        help="Phase 2 tracks file (default: <DATA_DIR>/processed/tracks.json).",
    )
    parser.add_argument(
        "--use-gnn",
        action="store_true",
        help=(
            "Enrich alerts with the Phase 3 GNN prediction. Off by default so a "
            "missing checkpoint degrades silently instead of erroring."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would happen WITHOUT writing the alert or notification store.",
    )
    parser.add_argument(
        "--no-notify", action="store_true", help="Skip the notification adapters."
    )
    parser.add_argument(
        "--max-print", type=int, default=10, help="Maximum number of alerts printed."
    )
    return parser


def format_alert(alert: Alert) -> str:
    """Render one alert as a human-readable block."""
    lines = [
        f"{alert.alert_id}  [{alert.severity.value} / priority {alert.priority.value}]",
        f"  status           : {alert.status.value} "
        f"({alert.confidence_label.value} input completeness)",
        f"  event            : {alert.event_id} ({alert.event_type.value})",
        f"  track            : {alert.track_id or 'not tracked'}",
        f"  location         : {alert.current_lat:.3f} N, {alert.current_lon:.3f} E",
    ]
    if alert.has_prediction:
        lines.append(
            f"  predicted        : {alert.predicted_lat:.3f} N, "
            f"{alert.predicted_lon:.3f} E (intensity {alert.predicted_intensity:.2f})"
        )
    else:
        lines.append("  predicted        : no GNN prediction available")
    lines.extend(
        [
            f"  intensity        : {alert.intensity:.2f} "
            f"(affected radius {alert.affected_radius_km:.1f} km)",
            f"  movement         : {alert.movement_speed_kmh:.2f} km/h "
            f"{alert.movement_direction}",
            f"  risk score       : {alert.risk.score:.2f}/100 -> {alert.risk.level.value}",
            f"  components used  : {', '.join(alert.source_components) or 'none'}",
            f"  valid until      : {alert.valid_until.isoformat()}",
            "  why this alert   :",
        ]
    )
    for line in alert.risk.reason_lines():
        lines.append(f"    {line}")
    return "\n".join(lines)


def summarise(result: AlertGenerationResult, store: AlertStore) -> None:
    """Print the aggregate summary requested by the CLI contract."""
    stored = result.stored
    alerts = stored.alerts
    print("\nSummary")
    print("-------")
    print(f"  Alerts in store   : {stored.alert_count}")
    print(f"  Created this run  : {stored.created_count}")
    print(f"  Updated this run  : {stored.updated_count}")
    print(f"  Suppressed        : {stored.suppressed_count}")
    if alerts:
        counts = ", ".join(
            f"{status.value}="
            f"{sum(1 for a in alerts if a.status is status)}"
            for status in type(alerts[0].status)
        )
        print(f"  By status         : {counts}")
        severities = ", ".join(
            f"{severity.value}="
            f"{sum(1 for a in alerts if a.severity is severity)}"
            for severity in AlertSeverity
        )
        print(f"  By severity       : {severities}")
    print(f"  Alert file        : {store.path}")


def main(argv: list[str] | None = None) -> int:
    """Entry point: load the pipeline outputs, score, apply rules and store."""
    args = build_parser().parse_args(argv)

    alert_store = AlertStore(args.alerts_output)
    service = AlertService(
        alert_store=alert_store,
        event_store=EventStore(args.events),
        track_store=TrackStore(args.tracks),
    )
    if args.use_gnn:
        from extreme_weather.gnn.service import GNNService

        service.gnn_service = GNNService(track_store=TrackStore(args.tracks))

    print(f"{PROJECT_NAME} ({PROJECT_ID}) - extreme weather alerting (prototype)")
    print(f"  events file : {args.events or service.event_store.path}")
    print(f"  tracks file : {args.tracks or service.track_store.path}")
    print(f"  alerts file : {alert_store.path}")
    print(f"  dry run     : {args.dry_run}")
    print(f"  notify      : {not args.no_notify}")
    print(f"  use GNN     : {args.use_gnn}")

    try:
        result = service.generate(dry_run=args.dry_run, notify=not args.no_notify)
    except (AlertError, AlertStoreError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("\nAlerts:")
    print("-------")
    if result.alert_count == 0:
        print(
            "No alerts were generated. Either there are no Phase 1 events, or none of\n"
            "them reached the configured trigger severity. Run "
            "'python scripts/detect_anomalies.py' first."
        )
    for alert in result.stored.alerts[: max(args.max_print, 0)]:
        print(format_alert(alert))
        print()

    summarise(result, alert_store)

    if result.degraded:
        print("\nDegraded inputs (the run continued without these)")
        print("---------------------------------------------------")
        for note in result.degraded:
            print(f"  - {note}")

    if result.notifications is not None:
        print(
            f"\n  notifications  : {len(result.notifications.records)} recorded locally"
        )
        for status in result.notifications.providers:
            print(
                f"    provider {status.provider}: delivered={status.delivered} "
                f"- {status.detail}"
            )

    print(f"\n  severity note : {SEVERITY_DISCLAIMER}")
    print(f"  risk note     : {RISK_DISCLAIMER}")
    if args.dry_run:
        print("\n  DRY RUN: no alert or notification store was written.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

