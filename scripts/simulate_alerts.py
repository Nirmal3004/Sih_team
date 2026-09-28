"""Demonstrate the Phase 5 alert lifecycle on synthetic data (SIH26078).

Usage (from the repository root)::

    python scripts/simulate_alerts.py
    python scripts/simulate_alerts.py --dry-run

    SIMULATION - SYNTHETIC DATA

The script walks a **synthetic** event through the whole alerting pipeline::

    normal event -> anomaly -> tracked event -> predicted movement
                -> alert created -> acknowledged -> resolution

It uses the same service, rules, store and notification adapters as the real
pipeline, so a demonstration shows what the pipeline actually does. It is fully
deterministic: the same stored events and tracks always yield the same steps,
the same alert ids and the same scores.

This is a **demonstration, not a forecast**. The events come from the project's
deterministic synthetic generator; no real weather observation is read, invented
or implied, and the result is not an official warning.
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

from extreme_weather.alerts.severity import SEVERITY_DISCLAIMER  # noqa: E402
from extreme_weather.alerts.service import (  # noqa: E402
    AlertError,
    AlertService,
)
from extreme_weather.alerts.simulation import (  # noqa: E402
    SIMULATION_LABEL,
    run_simulation,
)
from extreme_weather.alerts.store import AlertStore, AlertStoreError  # noqa: E402
from extreme_weather.anomaly.service import EventStore  # noqa: E402
from extreme_weather.config import PROJECT_ID, PROJECT_NAME  # noqa: E402
from extreme_weather.tracking.store import TrackStore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    """Return the command line parser for the simulation script."""
    parser = argparse.ArgumentParser(
        description=f"{PROJECT_NAME} ({PROJECT_ID}) - alert lifecycle simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--alerts-output",
        type=Path,
        default=None,
        help="Alerts JSON output (default: <DATA_DIR>/processed/alerts.json).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run the walk-through WITHOUT writing the alert or notification store.",
    )
    parser.add_argument(
        "--steps", type=int, default=6, help="Number of lifecycle steps to display."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: run the deterministic demonstration and print its steps."""
    args = build_parser().parse_args(argv)

    print("=" * 72)
    print(SIMULATION_LABEL)
    print("=" * 72)
    print(
        "This walk-through uses the project's deterministic SYNTHETIC event data.\n"
        "It is not a real weather event, not a forecast, and not an official warning."
    )
    print()

    alert_store = AlertStore(args.alerts_output)
    service = AlertService(
        alert_store=alert_store,
        event_store=EventStore(),
        track_store=TrackStore(),
    )

    # Attach a simulated ~5 km downscaled threat value to every stored synthetic
    # event so the demonstration exercises the "downscaling contributed" path.
    threat_lookup: dict[str, tuple[float, float | None]] = {}
    try:
        events = service.event_store.load()
    except Exception as exc:  # a missing/corrupt event file means no threats
        print(f"note: no usable event file ({exc}); continuing without a threat field.")
        events = None
    if events is not None:
        for event in events.events:
            threat_lookup[event.event_id] = (event.intensity, 5.0)

    if not threat_lookup:
        print(
            "No Phase 1 events are available. Run 'python scripts/detect_anomalies.py'\n"
            "and 'python scripts/track_events.py' first."
        )
        return 1

    try:
        result = run_simulation(
            service,
            steps=args.steps,
            dry_run=args.dry_run,
            threat_lookup=threat_lookup,
        )
    except (AlertError, AlertStoreError, ValueError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    print("Lifecycle")
    print("---------")
    for step in result.steps:
        stamp = step.timestamp.isoformat() if step.timestamp else "n/a"
        detail = ""
        if step.status is not None:
            detail = f" [status={step.status.value}"
            if step.severity is not None:
                detail += f", severity={step.severity.value}"
            if step.score is not None:
                detail += f", score={step.score:.2f}"
            detail += "]"
        print(f"{step.index + 1}. {step.stage:<15} {stamp:<21}{detail}")
        print(f"   {step.description}")

    print("\nAlerts after the simulation")
    print("----------------------------")
    for alert in result.alerts:
        print(
            f"  {alert.alert_id}  severity={alert.severity.value} "
            f"status={alert.status.value} score={alert.risk.score:.2f}"
        )
        print(f"    components used: {', '.join(alert.source_components) or 'none'}")
        if alert.acknowledged_at is not None:
            print(f"    acknowledged at: {alert.acknowledged_at.isoformat()}")
        if alert.resolved_at is not None:
            print(f"    resolved at    : {alert.resolved_at.isoformat()}")

    print(f"\n  notifications recorded: {result.notification_count} (local mock only)")
    if args.dry_run:
        print("  DRY RUN: no alert or notification store was written.")
    else:
        print(f"  alert store           : {alert_store.path}")

    print(f"\n  severity note: {SEVERITY_DISCLAIMER}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

