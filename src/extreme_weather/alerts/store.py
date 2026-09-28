"""File-backed alert store (SIH26078, Phase 5).

``alerts.json`` is written by ``scripts/generate_alerts.py`` (or by
:meth:`AlertService.generate`) and read by the alerts API and the dashboard.
There is deliberately **no database**: a versioned JSON file keeps the demo
reproducible, the infrastructure minimal and the state inspectable with ``cat``.

Robustness
----------
The store is written to survive the failure modes a file store actually hits in
a demo, and each is covered by a test:

* **missing file** - :meth:`AlertStore.load` returns ``None`` (an empty store),
  which the API turns into an empty list with an explanatory message, not a 500.
* **empty file** - a zero-byte file is treated as an empty store, not an error.
* **malformed JSON** - raises :class:`AlertStoreError`; the router maps that to
  HTTP 422 so the operator is told the file is broken.
* **malformed records** - individual alerts that fail validation, and duplicate
  ``alert_id`` values, are skipped and counted in ``metadata["skipped_records"]``
  so one bad record cannot hide every good one.
* **write failures** (read-only directory, disk full) - raise
  :class:`AlertStoreError` instead of propagating an ``OSError``.

Only ``data/processed`` is ever written; ``data/raw`` is never touched.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from extreme_weather.alerts.models import Alert, AlertStatus, AlertStoreResult
from extreme_weather.config import get_settings

__all__ = ["AlertStore", "AlertStoreError"]


class AlertStoreError(RuntimeError):
    """Raised when an alert file cannot be read or written."""


class AlertStore:
    """Read/write access to the stored alerts.

    Args:
        path: JSON file holding the alerts. Defaults to
            ``<DATA_DIR>/processed/<settings.alerts_filename>``.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        settings = get_settings()
        self._path = (
            Path(path) if path is not None else settings.processed_dir / settings.alerts_filename
        )

    @property
    def path(self) -> Path:
        """Location of the alert file."""
        return self._path

    def exists(self) -> bool:
        """True when an alert file is present."""
        return self._path.is_file()

    def save(self, result: AlertStoreResult) -> Path:
        """Write ``result`` as JSON, creating parent directories as needed.

        Raises:
            AlertStoreError: If the file cannot be written.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as handle:
                json.dump(result.as_dict(), handle, indent=2, ensure_ascii=False, default=str)
                handle.write("\n")
        except OSError as exc:
            raise AlertStoreError(f"Could not write alerts to '{self._path}': {exc}") from exc
        return self._path

    def load(self) -> AlertStoreResult | None:
        """Read the stored alerts, or ``None`` when nothing is stored.

        A zero-byte file is treated as an empty store: a truncated write should
        not make the dashboard unreachable.

        Raises:
            AlertStoreError: If the file exists, is non-empty and cannot be parsed.
        """
        if not self._path.is_file():
            return None
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise AlertStoreError(f"Could not read alert file '{self._path}': {exc}") from exc

        if not raw.strip():
            return AlertStoreResult(alerts=[], metadata={"empty_file": True})

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AlertStoreError(
                f"Could not read alert file '{self._path}': {exc}"
            ) from exc

        if not isinstance(payload, Mapping):
            raise AlertStoreError(
                f"Alert file '{self._path}' must contain a JSON object, got "
                f"{type(payload).__name__}."
            )
        return AlertStoreResult.from_dict(payload)

    def get(self, alert_id: str) -> Alert | None:
        """Return one alert by id, or ``None`` when it is not present."""
        result = self.load()
        if result is None:
            return None
        for alert in result.alerts:
            if alert.alert_id == alert_id:
                return alert
        return None

    def list(
        self,
        *,
        status: AlertStatus | str | None = None,
        severity: Any | None = None,
        event_type: Any | None = None,
        start_time: Any | None = None,
        end_time: Any | None = None,
        priority: Any | None = None,
    ) -> list[Alert]:
        """Return the stored alerts matching the given filters.

        * ``status``/``severity``/``event_type``/``priority`` accept enum members
          or strings;
        * ``start_time`` keeps alerts **created** at or after the given time;
        * ``end_time`` keeps alerts **created** at or before the given time.

        Raises:
            ValueError: If a filter value does not name a known member.
        """
        result = self.load()
        if result is None:
            return []

        alerts = list(result.alerts)
        if status is not None:
            wanted = (
                status if isinstance(status, AlertStatus) else AlertStatus.from_value(str(status))
            )
            alerts = [alert for alert in alerts if alert.status is wanted]
        if severity is not None:
            from extreme_weather.alerts.severity import AlertSeverity

            wanted = (
                severity
                if isinstance(severity, AlertSeverity)
                else AlertSeverity.from_value(str(severity))
            )
            alerts = [alert for alert in alerts if alert.severity is wanted]
        if priority is not None:
            from extreme_weather.alerts.severity import AlertSeverity

            wanted = (
                priority
                if isinstance(priority, AlertSeverity)
                else AlertSeverity.from_value(str(priority))
            )
            alerts = [alert for alert in alerts if alert.priority is wanted]
        if event_type is not None:
            from extreme_weather.schemas.event import EventType

            wanted = (
                event_type
                if isinstance(event_type, EventType)
                else EventType.from_value(str(event_type))
            )
            alerts = [alert for alert in alerts if alert.event_type is wanted]
        if start_time is not None:
            alerts = [alert for alert in alerts if alert.created_at >= start_time]
        if end_time is not None:
            alerts = [alert for alert in alerts if alert.created_at <= end_time]
        return alerts

    def open_alerts(self) -> list[Alert]:
        """Return the alerts that still need operator attention."""
        result = self.load()
        if result is None:
            return []
        return [alert for alert in result.alerts if alert.status.is_open]

    def clear(self) -> None:
        """Delete the stored alert file if it exists."""
        self._path.unlink(missing_ok=True)

