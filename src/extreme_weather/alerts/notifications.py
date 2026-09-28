"""Notification adapters (SIH26078, Phase 5).

Design
------
Notifications are **optional by construction**. The pipeline never depends on a
notification succeeding: :meth:`NotificationProvider.send` is best-effort, and a
failing provider is recorded as a failure rather than allowed to break alert
generation. The only provider that runs by default is the **local mock**, which
appends to ``data/processed/notifications.json``.

Why this matters for a prototype
--------------------------------
Real channels (SMS, email, push) would need credentials, an outbound network
call and a delivery guarantee - none of which belongs in a research demo, and
none of which is implemented here. :class:`EmailNotificationProvider` and
:class:`WebhookNotificationProvider` therefore exist as **stubs**: they are
registered, they report ``supports() -> True`` only when explicitly configured,
and their ``send`` raises a clear "not implemented" error rather than pretending
to deliver a message. Nothing here sends a real external notification, and no
API key, token or webhook secret is read, stored or logged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, field_validator

from extreme_weather.alerts.models import Alert, AlertStatus
from extreme_weather.config import get_settings

__all__ = [
    "NOTIFICATION_DISCLAIMER",
    "EmailNotificationProvider",
    "MockNotificationProvider",
    "NotificationError",
    "NotificationProvider",
    "NotificationRecord",
    "NotificationResult",
    "NotificationStatus",
    "NotificationStore",
    "WebhookNotificationProvider",
    "build_notification_providers",
]


NOTIFICATION_DISCLAIMER = (
    "Prototype notification log. Messages are written locally and are never "
    "delivered to any external service. No real email, SMS or webhook is sent."
)


class NotificationError(RuntimeError):
    """Raised when a notification cannot be produced or delivered."""


class NotificationStatus(BaseModel):
    """Result of one delivery attempt to one provider."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    provider: str = Field(min_length=1, description="Provider name.")
    delivered: bool = Field(description="True only for a real (local) delivery.")
    detail: str = Field(default="", description="What the provider did.")


class NotificationRecord(BaseModel):
    """One notification, as stored in ``notifications.json``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    notification_id: str = Field(min_length=1)
    alert_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    channel: str = Field(default="local", min_length=1)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)
    severity: str = Field(min_length=1)
    alert_status: AlertStatus
    created_at: datetime
    delivered: bool = Field(default=False)
    detail: str = Field(default="")
    disclaimer: str = Field(default=NOTIFICATION_DISCLAIMER)

    @field_validator("created_at")
    @classmethod
    def _created_to_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)


class NotificationResult(BaseModel):
    """Everything one notification run produced."""

    model_config = ConfigDict(extra="forbid")

    records: list[NotificationRecord] = Field(default_factory=list)
    providers: list[NotificationStatus] = Field(default_factory=list)
    source: str | None = Field(default=None, description="Notification log file.")
    message: str | None = Field(default=None, description="Explains an empty result.")
    notice: str = Field(default=NOTIFICATION_DISCLAIMER)

    @property
    def delivered_count(self) -> int:
        """Number of notifications a provider actually accepted."""
        return sum(1 for record in self.records if record.delivered)

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dictionary with a stable key order."""
        return {
            "schema_version": "1.0",
            "generated_at": (
                datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            ),
            "record_count": len(self.records),
            "delivered_count": self.delivered_count,
            "providers": [status.model_dump(mode="json") for status in self.providers],
            "records": [record.model_dump(mode="json") for record in self.records],
            "notice": NOTIFICATION_DISCLAIMER,
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "NotificationResult":
        """Rebuild a result, skipping malformed records instead of failing."""
        if not isinstance(payload, dict):
            return cls()
        records: list[NotificationRecord] = []
        for item in payload.get("records", []) or []:
            try:
                records.append(NotificationRecord.model_validate(item))
            except Exception:  # a malformed record must not hide the good ones
                continue
        providers: list[NotificationStatus] = []
        for item in payload.get("providers", []) or []:
            try:
                providers.append(NotificationStatus.model_validate(item))
            except Exception:
                continue
        return cls(records=records, providers=providers)


@runtime_checkable
class NotificationProvider(Protocol):
    """The adapter interface every notification channel implements."""

    def send(self, alert: Alert) -> NotificationRecord:
        """Deliver ``alert`` and return the record describing what happened.

        Implementations must not raise for an ordinary delivery failure; they
        return a record with ``delivered=False`` and a ``detail`` string.
        """
        ...

    def supports(self) -> bool:
        """True when this provider is configured and ready to be used."""
        ...

    def name(self) -> str:
        """Stable provider name, recorded on every notification."""
        ...


def _format_subject(alert: Alert) -> str:
    return f"[{alert.severity.value}] {alert.title}"


def _format_body(alert: Alert) -> str:
    """Render an alert as plain text, including its explanation."""
    lines = [
        alert.message,
        "",
        f"Event       : {alert.event_id} ({alert.event_type.value})",
        f"Track       : {alert.track_id or 'not tracked'}",
        f"Status      : {alert.status.value}",
        f"Priority    : {alert.priority.value}",
        f"Risk score  : {alert.risk.score:.2f}/100",
        f"Location    : {alert.current_lat:.3f} N, {alert.current_lon:.3f} E",
    ]
    if alert.has_prediction:
        lines.append(
            f"Predicted   : {alert.predicted_lat:.3f} N, {alert.predicted_lon:.3f} E"
        )
    else:
        lines.append("Predicted   : no GNN prediction available")
    lines.append(
        f"Components  : {', '.join(alert.source_components) or 'none recorded'}"
    )
    lines.append("")
    lines.append("Why this alert exists:")
    lines.extend(alert.risk.reason_lines())
    lines.append("")
    lines.append(alert.notice)
    return "\n".join(lines)


class MockNotificationProvider:
    """Local provider that records notifications instead of sending them.

    It is the default and the only provider the pipeline uses out of the box.
    ``send`` appends one record to the configured log file and also keeps it in
    memory so tests and the API can read it without touching disk twice.
    """

    def __init__(self, store: "NotificationStore | None" = None, *, persist: bool = True) -> None:
        self._store = store or NotificationStore()
        self._persist = persist
        self.records: list[NotificationRecord] = []

    def name(self) -> str:
        """Return the provider name."""
        return "mock"

    def supports(self) -> bool:
        """The mock provider is always available - it needs no credentials."""
        return True

    def send(self, alert: Alert) -> NotificationRecord:
        """Record the alert locally. Never raises for an ordinary failure."""
        record = NotificationRecord(
            notification_id=f"NTF-{alert.alert_id}-{len(self.records) + 1:03d}",
            alert_id=alert.alert_id,
            provider=self.name(),
            channel="local",
            subject=_format_subject(alert),
            body=_format_body(alert),
            severity=alert.severity.value,
            alert_status=alert.status,
            created_at=alert.updated_at,
            delivered=True,
            detail="recorded locally; no external delivery attempted",
        )
        self.records.append(record)
        if self._persist:
            try:
                self._store.append(record)
            except NotificationError as exc:
                # A failed write degrades the channel, never the alert itself.
                return record.model_copy(
                    update={"delivered": False, "detail": f"local log write failed: {exc}"}
                )
        return record


class _StubProvider:
    """Base class for the deliberately unimplemented external channels.

    A stub is *registered* so the adapter architecture is real, but it never
    claims to have delivered anything and never reads a credential.
    """

    provider_name: str = "stub"
    reason: str = "external delivery is not implemented in this prototype"

    def name(self) -> str:
        """Return the provider name."""
        return self.provider_name

    def supports(self) -> bool:
        """Stubs are never enabled automatically."""
        return False

    def send(self, alert: Alert) -> NotificationRecord:
        """Always report a non-delivery; never contact an external service."""
        return NotificationRecord(
            notification_id=f"NTF-{alert.alert_id}-{self.provider_name}",
            alert_id=alert.alert_id,
            provider=self.provider_name,
            channel=self.provider_name,
            subject=_format_subject(alert),
            body=_format_body(alert),
            severity=alert.severity.value,
            alert_status=alert.status,
            created_at=alert.updated_at,
            delivered=False,
            detail=f"{self.reason}; no message was sent and no credential is used",
        )


class EmailNotificationProvider(_StubProvider):
    """Placeholder for a future email channel.

    Intentionally not implemented: sending email would require an SMTP server
    and credentials, and this project makes no external calls. The class exists
    so a real adapter can be added later without changing the pipeline.
    """

    provider_name = "email"
    reason = (
        "email delivery is not implemented; this project sends no external "
        "messages and stores no SMTP credentials"
    )


class WebhookNotificationProvider(_StubProvider):
    """Placeholder for a future webhook channel.

    Intentionally not implemented: no webhook URL or secret is configured, read
    or contacted anywhere in this project.
    """

    provider_name = "webhook"
    reason = (
        "webhook delivery is not implemented; no webhook URL or secret is "
        "configured, read or contacted"
    )


class NotificationStore:
    """Append-only JSON log of notifications (``notifications.json``).

    Like the alert store it tolerates a missing or empty file, and it skips
    malformed records on read so one bad entry cannot hide the history.
    """

    def __init__(self, path: Path | str | None = None) -> None:
        settings = get_settings()
        self._path = (
            Path(path)
            if path is not None
            else settings.processed_dir / settings.notifications_filename
        )

    @property
    def path(self) -> Path:
        """Location of the notification log."""
        return self._path

    def exists(self) -> bool:
        """True when a notification log is present."""
        return self._path.is_file()

    def append(self, record: NotificationRecord) -> Path:
        """Append one record, creating the file and its parent if needed.

        Raises:
            NotificationError: If the log cannot be written.
        """
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            existing = self.load().records
            payload = NotificationResult(records=[*existing, record]).as_dict()
            with self._path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False, default=str)
                handle.write("\n")
        except (OSError, NotificationError) as exc:
            raise NotificationError(
                f"Could not write notifications to '{self._path}': {exc}"
            ) from exc
        return self._path

    def load(self) -> NotificationResult:
        """Read the log, returning an empty result when nothing is stored.

        A malformed file is reported as an error rather than silently emptied -
        the caller decides how loudly to complain.
        """
        if not self._path.is_file():
            return NotificationResult(source=str(self._path), message="No notifications recorded yet.")
        try:
            raw = self._path.read_text(encoding="utf-8")
        except OSError as exc:
            raise NotificationError(
                f"Could not read notification file '{self._path}': {exc}"
            ) from exc
        if not raw.strip():
            return NotificationResult(source=str(self._path), message="The notification log is empty.")
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise NotificationError(
                f"Could not read notification file '{self._path}': {exc}"
            ) from exc
        result = NotificationResult.from_dict(payload)
        return result.model_copy(update={"source": str(self._path)})

    def clear(self) -> None:
        """Delete the notification log if it exists."""
        self._path.unlink(missing_ok=True)


def build_notification_providers(
    store: NotificationStore | None = None,
) -> list[NotificationProvider]:
    """Return the notification providers enabled by default.

    Only the local mock is enabled. The email and webhook adapters exist in the
    registry so the architecture is real, but they report ``supports() -> False``
    and are excluded here - enabling them is an explicit future decision that
    would require credentials this project deliberately does not hold.
    """
    providers: list[NotificationProvider] = [MockNotificationProvider(store)]
    for provider in (EmailNotificationProvider(), WebhookNotificationProvider()):
        if provider.supports():
            providers.append(provider)
    return providers

