"""Alert service and lifecycle (SIH26078, Phase 5).

The service is the business layer: it assembles a
:class:`~extreme_weather.alerts.engine.RiskContext` from the Phase 1-4 outputs,
scores it, runs the rules, applies deduplication, writes the store and hands the
result to the notification adapters. The CLI and the FastAPI router both call
it, so neither contains decision logic.

Graceful degradation
--------------------
Every optional input is genuinely optional. The service loads each source
independently and records which ones actually contributed, so a run with no
GNN checkpoint and no downscaling model still produces honest alerts with
``source_components = ["events"]`` or ``["events", "tracking"]`` instead of
failing. :class:`AlertGenerationResult` carries the same information in
``metadata`` so the CLI and the API can explain the degraded run.

Lifecycle
---------
Transitions are explicit and enforced, not implied by field updates:

.. code-block:: text

    NEW --> ACTIVE --> ACKNOWLEDGED --> RESOLVED
     |         |             |
     +---------+-------------+--> EXPIRED   (event disappeared, R6)

``RESOLVED`` and ``EXPIRED`` are terminal. An attempt at an illegal transition
raises :class:`AlertTransitionError` rather than silently succeeding, because a
dashboard that quietly accepts "resolve an expired alert" is worse than one
that refuses.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from extreme_weather.alerts.engine import RiskConfig, RiskContext, assess_risk
from extreme_weather.alerts.models import (
    Alert,
    AlertStatus,
    AlertStoreResult,
    AlertSummary,
    ConfidenceLabel,
    SeverityCount,
    StatusCount,
    build_alert_id,
    deduplication_key,
)
from extreme_weather.alerts.notifications import (
    NotificationProvider,
    NotificationResult,
    build_notification_providers,
)
from extreme_weather.alerts.rules import RuleConfig, RuleContext, apply_rules
from extreme_weather.alerts.severity import AlertSeverity, max_severity
from extreme_weather.alerts.store import AlertStore, AlertStoreError
from extreme_weather.anomaly.service import AnalysisError, EventStore
from extreme_weather.config import get_settings
from extreme_weather.schemas.event import EventType
from extreme_weather.schemas.track import EventTrack, TrackStatus
from extreme_weather.tracking.store import TrackStore, TrackingStoreError

__all__ = [
    "AlertError",
    "AlertGenerationResult",
    "AlertService",
    "AlertTransitionError",
    "AlertUnknownError",
    "SOURCE_COMPONENT_DOWNSCALING",
    "SOURCE_COMPONENT_EVENTS",
    "SOURCE_COMPONENT_GNN",
    "SOURCE_COMPONENT_TRACKING",
]


#: Names recorded in ``Alert.source_components``. The value tells an operator
#: exactly which pipeline stages produced the alert.
SOURCE_COMPONENT_EVENTS = "events"
SOURCE_COMPONENT_TRACKING = "tracking"
SOURCE_COMPONENT_GNN = "gnn"
SOURCE_COMPONENT_DOWNSCALING = "downscaling"

#: Legal lifecycle transitions. Everything not listed raises.
ALLOWED_TRANSITIONS: dict[AlertStatus, tuple[AlertStatus, ...]] = {
    AlertStatus.NEW: (AlertStatus.ACTIVE, AlertStatus.ACKNOWLEDGED, AlertStatus.RESOLVED, AlertStatus.EXPIRED),
    AlertStatus.ACTIVE: (AlertStatus.ACKNOWLEDGED, AlertStatus.RESOLVED, AlertStatus.EXPIRED),
    AlertStatus.ACKNOWLEDGED: (AlertStatus.RESOLVED, AlertStatus.EXPIRED),
    AlertStatus.RESOLVED: (),
    AlertStatus.EXPIRED: (),
}


class AlertError(RuntimeError):
    """Base error of the alerting layer."""


class AlertUnknownError(LookupError):
    """The requested alert id does not exist (maps to HTTP 404)."""


class AlertTransitionError(AlertError):
    """An illegal lifecycle transition was requested (maps to HTTP 422)."""


@dataclass(frozen=True)
class AlertGenerationResult:
    """Outcome of one :meth:`AlertService.generate` call.

    Attributes:
        stored: The alert store contents after the run.
        notifications: What the providers recorded, or ``None`` when the run
            skipped notification (a dry run).
        dry_run: True when nothing was written to disk.
        degraded: Human-readable notes about which optional inputs were missing.
    """

    stored: AlertStoreResult
    notifications: NotificationResult | None = None
    dry_run: bool = False
    degraded: tuple[str, ...] = ()

    @property
    def alert_count(self) -> int:
        """Number of alerts held after the run."""
        return self.stored.alert_count

    @property
    def created_count(self) -> int:
        """Number of newly created alerts."""
        return self.stored.created_count

    @property
    def updated_count(self) -> int:
        """Number of existing alerts refreshed in place."""
        return self.stored.updated_count

    @property
    def suppressed_count(self) -> int:
        """Number of candidates dropped by deduplication or the trigger rule."""
        return self.stored.suppressed_count


class AlertService:
    """Generate, query and transition alerts.

    Args:
        alert_store: Where alerts are read and written. Defaults to
            ``<DATA_DIR>/processed/<settings.alerts_filename>``.
        event_store: Source of Phase 1 detections.
        track_store: Source of Phase 2 trajectories.
        gnn_service: Optional Phase 3 predictor. ``None`` disables prediction
            enrichment rather than failing.
        notification_providers: Adapters to notify. Defaults to the local mock.
        risk_config: Reference values of the scoring ladder.
        rule_config: Thresholds of the rule engine.
    """

    def __init__(
        self,
        alert_store: AlertStore | None = None,
        *,
        event_store: EventStore | None = None,
        track_store: TrackStore | None = None,
        gnn_service: Any | None = None,
        notification_providers: Sequence[NotificationProvider] | None = None,
        risk_config: RiskConfig | None = None,
        rule_config: RuleConfig | None = None,
    ) -> None:
        settings = get_settings()
        self.alert_store = alert_store or AlertStore()
        self.event_store = event_store or EventStore()
        self.track_store = track_store or TrackStore()
        self.gnn_service = gnn_service
        self.notification_providers: list[NotificationProvider] = list(
            notification_providers
            if notification_providers is not None
            else build_notification_providers()
        )
        self.risk_config = risk_config or RiskConfig()
        self.rule_config = rule_config or RuleConfig(
            default_validity_minutes=settings.alert_default_validity_minutes,
            deduplication_window_minutes=settings.alert_deduplication_window_minutes,
        )

    # ------------------------------------------------------------------ #
    # Loading the optional pipeline outputs
    # ------------------------------------------------------------------ #
    def _load_events(self) -> tuple[dict[str, Any], list[str]]:
        """Load the Phase 1 detections; an absent file yields an empty list."""
        notes: list[str] = []
        try:
            result = self.event_store.load()
        except AnalysisError as exc:
            raise AlertError(f"Could not read the event analysis: {exc}") from exc
        if result is None:
            notes.append("No Phase 1 event analysis found; run scripts/detect_anomalies.py.")
            return {}, notes
        return {event.event_id: event for event in result.events}, notes

    def _load_tracks(self) -> tuple[dict[str, EventTrack], list[str]]:
        """Load the Phase 2 tracks; an absent file yields an empty mapping."""
        notes: list[str] = []
        try:
            result = self.track_store.load()
        except TrackingStoreError as exc:
            raise AlertError(f"Could not read the tracking file: {exc}") from exc
        if result is None:
            notes.append("No Phase 2 tracking file found; run scripts/track_events.py.")
            return {}, notes
        return {track.track_id: track for track in result.tracks}, notes

    def _latest_event_of_track(self, track: EventTrack) -> Any | None:
        """Return the newest event of a track, or ``None`` when it is unknown."""
        events = getattr(self, "_events_by_id", {})
        for point in reversed(track.points):
            event = events.get(point.event_id)
            if event is not None:
                return event
        return None

    def _predict(self, track: EventTrack) -> tuple[Any | None, str | None]:
        """Run the Phase 3 model for one track, degrading to ``None``.

        Returns the prediction and a note describing why it is missing. A missing
        checkpoint is the normal case before training, so it must not raise.
        """
        if self.gnn_service is None:
            return None, "No GNN service configured; prediction not used."
        try:
            return self.gnn_service.predict(track.track_id), None
        except Exception as exc:  # the GNN is optional in every failure mode
            return None, f"GNN prediction unavailable for {track.track_id}: {exc}"


    @property
    def alert_count(self) -> int:
        """Number of alerts held after the run."""
        return self.stored.alert_count

    @property
    def created_count(self) -> int:
        """Number of newly created alerts."""
        return self.stored.created_count

    @property
    def updated_count(self) -> int:
        """Number of existing alerts refreshed in place."""
        return self.stored.updated_count

    @property
    def suppressed_count(self) -> int:
        """Number of candidates dropped by deduplication or by the trigger rule."""
        return self.stored.suppressed_count

    def _downscaled_threat(self, event: Any) -> tuple[float | None, float | None, str | None]:
        """Read the Phase 4 downscaled threat near an event, if one is available.

        The prototype has no persisted downscaled field per event, so this
        returns ``(None, None, note)`` unless a caller supplies a threat value
        (the simulation does). That is the honest behaviour: the alert reports
        that no downscaled information was used instead of inventing one.
        """
        threat = self._downscaled_threat_lookup.get(event.event_id) if hasattr(
            self, "_downscaled_threat_lookup"
        ) else None
        if threat is None:
            return None, None, "No AI-downscaled threat field available for this event."
        value, resolution = threat
        return float(value), float(resolution or 5.0), None

    def _build_context(
        self,
        event: Any,
        track: EventTrack | None,
        prediction: Any | None,
        threat: tuple[float | None, float | None, str | None],
    ) -> RiskContext:
        """Assemble the scoring input from whatever is available."""
        threat_value, threat_resolution, _ = threat
        components = [SOURCE_COMPONENT_EVENTS]
        if track is not None:
            components.append(SOURCE_COMPONENT_TRACKING)
        if prediction is not None:
            components.append(SOURCE_COMPONENT_GNN)
        if threat_value is not None:
            components.append(SOURCE_COMPONENT_DOWNSCALING)

        return RiskContext(
            event_id=event.event_id,
            event_type=event.event_type,
            intensity=float(event.intensity),
            event_severity=event.severity,
            affected_radius_km=float(event.affected_radius_km),
            track_id=track.track_id if track else None,
            duration_hours=float(track.duration_hours) if track else 0.0,
            movement_speed_kmh=float(track.mean_speed_kmh) if track else 0.0,
            intensity_trend=track.intensity_trend if track else None,
            peak_intensity=float(track.peak_intensity) if track else None,
            predicted_intensity=(
                float(prediction.predicted_intensity) if prediction else None
            ),
            predicted_speed_kmh=(
                float(prediction.predicted_speed_kmh)
                if prediction and prediction.predicted_speed_kmh is not None
                else None
            ),
            downscaled_threat=threat_value,
            downscaled_resolution_km=threat_resolution,
            source_components=tuple(components),
        )

    @staticmethod
    def _latest_point(track: EventTrack) -> Any:
        """Return the newest point of a track (tracks always have >= 1 point)."""
        return track.points[-1]

    def _track_for_event(
        self, tracks: Sequence[EventTrack], event: Any
    ) -> EventTrack | None:
        """Return the track whose *newest* point is this event.

        Only the newest detection of a track is alerted on: the older points are
        already superseded, and alerting on them would re-issue the same threat
        every hour.
        """
        for track in tracks:
            if not track.points:
                continue
            if track.points[-1].event_id == event.event_id:
                return track
        return None

    def _title(self, event: Any, level: AlertSeverity) -> str:
        return f"{level.value} {event.event_type.value.replace('_', ' ').title()} alert"

    def _message(
        self,
        event: Any,
        track: EventTrack | None,
        prediction: Any | None,
        context: RiskContext,
    ) -> str:
        """Compose the one-paragraph alert message, using only real facts."""
        parts = [
            f"A {event.event_type.value.replace('_', ' ').lower()} event was detected at "
            f"{event.latitude:.2f} N, {event.longitude:.2f} E with an anomaly score of "
            f"{event.anomaly_score:.2f} (Phase 1 severity {event.severity.value})."
        ]
        if track is not None:
            parts.append(
                f"It has been tracked for {track.duration_hours:.1f} h, moving "
                f"{track.direction} at up to {track.max_speed_kmh:.1f} km/h."
            )
        if prediction is not None:
            parts.append(
                f"A prototype GNN prediction places it near "
                f"{prediction.predicted_latitude:.2f} N, {prediction.predicted_longitude:.2f} E "
                f"with intensity {prediction.predicted_intensity:.2f}."
            )
        else:
            parts.append("No GNN prediction is available for this event.")
        if context.downscaled_threat is None:
            parts.append("No AI-downscaled threat field is available for this event.")
        parts.append(
            "This is prototype output on synthetic data, not an official warning and not "
            "a validated forecast."
        )
        return " ".join(parts)


    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def _merge(self, candidates: list[Alert], existing: list[Alert]) -> list[Alert]:
        """Merge refreshed candidates into the stored alerts.

        An alert that the rules resolved or expired stays in the store with its
        terminal status: history is preserved so the dashboard can show what
        happened, and only ``create``/``update`` ever add or change entries.
        """
        merged: dict[str, Alert] = {alert.alert_id: alert for alert in existing}
        for alert in candidates:
            merged[alert.alert_id] = alert
        return sorted(
            merged.values(),
            key=lambda a: (-a.priority.rank, -a.severity.rank, a.created_at, a.alert_id),
        )

    def generate(
        self,
        *,
        dry_run: bool = False,
        now: datetime | None = None,
        notify: bool = True,
    ) -> AlertGenerationResult:
        """Run the whole alerting pipeline over the stored pipeline outputs.

        ```text
        load events -> load tracks -> load GNN prediction (optional)
            -> load downscaled threat (optional) -> risk engine -> alert rules
            -> deduplication -> alert store -> notification adapters
        ```

        Args:
            dry_run: When True, **nothing is written**: neither the alert store
                nor the notification log is touched. The returned result still
                reports what *would* have happened.
            now: Timestamp to stamp the run. Defaults to the current UTC time;
                pass an explicit value to keep a run reproducible.
            notify: When False, the notification adapters are skipped.

        Returns:
            An :class:`AlertGenerationResult`.

        Raises:
            AlertError: If a required source file exists but cannot be parsed.
        """
        stamp = now or datetime.now(timezone.utc).replace(microsecond=0)
        events_by_id, notes = self._load_events()
        self._events_by_id = events_by_id
        tracks_by_id, track_notes = self._load_tracks()
        notes.extend(track_notes)

        existing_result = self.alert_store.load()
        existing = list(existing_result.alerts) if existing_result else []
        existing_by_key = {alert.deduplication_key: alert for alert in existing}

        created = 0
        updated = 0
        suppressed = 0
        candidates: list[Alert] = []
        window = self.rule_config.deduplication_window_minutes

        for event in events_by_id.values():
            track = self._track_for_event(list(tracks_by_id.values()), event)
            prediction = None
            if track is not None:
                prediction, note = self._predict(track)
                if note:
                    notes.append(note)
            threat = self._downscaled_threat(event)

            context = self._build_context(event, track, prediction, threat)
            risk = assess_risk(context, self.risk_config, generated_at=stamp)

            # Untracked events are only alertable on their own Phase 1 severity.
            existing_alert = existing_by_key.get(
                deduplication_key(event.event_id, risk.level, stamp, window)
            )
            rule_context = RuleContext(
                event_id=event.event_id,
                event_type=event.event_type,
                event_severity=event.severity,
                intensity=context.intensity,
                observed_at=event.timestamp,
                risk_level=risk.level,
                risk_score=risk.score,
                track_id=track.track_id if track else None,
                duration_hours=context.duration_hours,
                intensity_trend=context.intensity_trend,
                predicted_intensity=context.predicted_intensity,
                has_downscaled_threat=context.downscaled_threat is not None,
                existing_status=existing_alert.status if existing_alert else None,
                existing_level=existing_alert.severity if existing_alert else None,
                hours_since_observed=(
                    max(0.0, (stamp - event.timestamp).total_seconds() / 3600.0)
                ),
            )
            outcomes = apply_rules(rule_context, self.rule_config)
            if outcomes and outcomes[0].suppress:
                suppressed += 1
                continue

            # Combine the rules that changed a level, in registry order.
            severity = risk.level
            priority = risk.level
            status = AlertStatus.ACTIVE if track is not None else AlertStatus.NEW
            for outcome in outcomes:
                if outcome.severity is not None:
                    severity = outcome.severity
                if outcome.priority is not None:
                    priority = max_severity(priority, outcome.priority)
                if outcome.status is not None and outcome.status is not status:
                    status = outcome.status
            severity = max_severity(severity, risk.level)
            priority = max_severity(priority, severity)


            key = deduplication_key(event.event_id, severity, stamp, window)
            alert_id = build_alert_id(event.event_id, severity, stamp, window)
            point = self._latest_point(track) if track is not None else None
            components = list(context.source_components)

            if existing_alert is not None:
                # Deduplication: refresh the open alert instead of adding a twin.
                candidates.append(
                    existing_alert.model_copy(
                        update={
                            "severity": severity,
                            "priority": priority,
                            "risk": risk,
                            "updated_at": stamp,
                            "valid_until": stamp
                            + timedelta(minutes=self.rule_config.default_validity_minutes),
                            "source_components": components,
                        }
                    )
                )
                updated += 1
                continue

            candidates.append(
                Alert(
                    alert_id=alert_id,
                    event_id=event.event_id,
                    track_id=track.track_id if track else None,
                    event_type=event.event_type,
                    severity=severity,
                    priority=priority,
                    title=self._title(event, severity),
                    message=self._message(event, track, prediction, context),
                    created_at=stamp,
                    updated_at=stamp,
                    valid_from=stamp,
                    valid_until=stamp
                    + timedelta(minutes=self.rule_config.default_validity_minutes),
                    status=status,
                    confidence_label=ConfidenceLabel.for_components(components),
                    risk=risk,
                    current_lat=float(point.latitude if point else event.latitude),
                    current_lon=float(point.longitude if point else event.longitude),
                    predicted_lat=(
                        float(prediction.predicted_latitude) if prediction else None
                    ),
                    predicted_lon=(
                        float(prediction.predicted_longitude) if prediction else None
                    ),
                    intensity=context.intensity,
                    predicted_intensity=(
                        float(prediction.predicted_intensity) if prediction else None
                    ),
                    movement_speed_kmh=context.movement_speed_kmh,
                    movement_direction=track.direction if track else "CALM",
                    affected_radius_km=float(event.affected_radius_km),
                    source_components=components,
                    deduplication_key=key,
                )
            )
            created += 1

        stored = AlertStoreResult(
            alerts=self._merge(candidates, existing),
            generated_at=stamp,
            created_count=created,
            updated_count=updated,
            suppressed_count=suppressed,
            metadata={
                "dry_run": dry_run,
                "event_count": len(events_by_id),
                "track_count": len(tracks_by_id),
                "risk_config": self.risk_config.as_dict(),
                "rule_config": self.rule_config.as_dict(),
                "notes": list(dict.fromkeys(notes)),
            },
        )

        notifications: NotificationResult | None = None
        if notify and not dry_run:
            notifications = self.notify(candidates)
            stored = stored.model_copy(
                update={
                    "metadata": {
                        **stored.metadata,
                        "notification_count": len(notifications.records),
                    }
                }
            )

        if not dry_run:
            self.alert_store.save(stored)

        return AlertGenerationResult(
            stored=stored,
            notifications=notifications,
            dry_run=dry_run,
            degraded=tuple(dict.fromkeys(notes)),
        )

    # ------------------------------------------------------------------ #
    # Notification
    # ------------------------------------------------------------------ #
    def notify(
        self,
        alerts: Sequence[Alert],
        notification_store: Any | None = None,
    ) -> NotificationResult:
        """Send every alert to every enabled provider, best-effort.

        A provider that fails is recorded as a non-delivery; it never prevents
        the alert itself from being stored. That is the whole point of keeping
        notifications optional.
        """
        from extreme_weather.alerts.notifications import NotificationResult, NotificationStatus

        records = []
        providers: list[NotificationStatus] = []
        for provider in self.notification_providers:
            name = provider.name()
            for alert in alerts:
                try:
                    records.append(provider.send(alert))
                except Exception as exc:  # a broken channel must not break alerts
                    providers.append(
                        NotificationStatus(
                            provider=name,
                            delivered=False,
                            detail=f"provider raised {type(exc).__name__}: {exc}",
                        )
                    )
            providers.append(
                NotificationStatus(
                    provider=name,
                    delivered=True,
                    detail=f"{name} is enabled",
                )
            )
        return NotificationResult(
            records=records,
            providers=providers,
            source=str(notification_store) if notification_store is not None else None,
        )


    # ------------------------------------------------------------------ #
    # CRUD
    # ------------------------------------------------------------------ #
    def _load_or_raise(self) -> AlertStoreResult:
        """Load the store, treating an absent file as an empty store."""
        result = self.alert_store.load()
        return result if result is not None else AlertStoreResult()

    def _persist(self, alerts: list[Alert], *, updated_at: datetime | None = None) -> list[Alert]:
        """Write the alert list back to the store, newest priority first."""
        stamp = updated_at or datetime.now(timezone.utc).replace(microsecond=0)
        ordered = sorted(
            alerts,
            key=lambda a: (-a.priority.rank, -a.severity.rank, a.created_at, a.alert_id),
        )
        previous = self._load_or_raise()
        self.alert_store.save(
            AlertStoreResult(
                alerts=ordered,
                generated_at=stamp,
                created_count=previous.created_count,
                updated_count=previous.updated_count,
                suppressed_count=previous.suppressed_count,
                metadata=previous.metadata,
            )
        )
        return ordered

    def _find(self, alerts: Sequence[Alert], alert_id: str) -> Alert:
        """Return one alert or raise :class:`AlertUnknownError` (HTTP 404)."""
        for alert in alerts:
            if alert.alert_id == alert_id:
                return alert
        raise AlertUnknownError(f"Alert '{alert_id}' not found.")

    @staticmethod
    def _check_transition(current: AlertStatus, target: AlertStatus) -> None:
        """Reject an illegal lifecycle transition (HTTP 422).

        Raises:
            AlertTransitionError: If the transition is not allowed.
        """
        if target not in ALLOWED_TRANSITIONS[current]:
            allowed = ", ".join(s.value for s in ALLOWED_TRANSITIONS[current]) or "none"
            raise AlertTransitionError(
                f"Cannot move an alert from {current.value} to {target.value}. "
                f"Allowed from {current.value}: {allowed}."
            )

    def create_alert(
        self,
        alert: Alert,
        *,
        now: datetime | None = None,
    ) -> Alert:
        """Store a new alert.

        Raises:
            AlertError: If an alert with the same id is already stored, which
                would break the deterministic-id guarantee.
        """
        alerts = list(self._load_or_raise().alerts)
        if any(existing.alert_id == alert.alert_id for existing in alerts):
            raise AlertError(
                f"Alert '{alert.alert_id}' already exists; use update_alert instead."
            )
        stamp = now or alert.updated_at
        alerts.append(alert.model_copy(update={"updated_at": stamp}))
        self._persist(alerts, updated_at=stamp)
        return alert

    def update_alert(
        self,
        alert_id: str,
        changes: Mapping[str, Any],
        *,
        now: datetime | None = None,
    ) -> Alert:
        """Apply a validated field change to an existing alert.

        Status is **not** updatable this way: lifecycle changes go through
        :meth:`acknowledge_alert`, :meth:`resolve_alert` and :meth:`expire_alerts`
        so the legal-transition check is never bypassed.

        Raises:
            AlertUnknownError: If the alert does not exist.
            AlertError: If a field cannot be changed this way, or the change is
                rejected by validation.
        """
        forbidden = {"alert_id", "status", "created_at", "acknowledged_at", "resolved_at"}
        illegal = sorted(forbidden & set(changes))
        if illegal:
            raise AlertError(
                f"Field(s) {illegal} cannot be changed with update_alert; use the "
                "dedicated lifecycle methods."
            )

        alerts = list(self._load_or_raise().alerts)
        current = self._find(alerts, alert_id)
        stamp = now or datetime.now(timezone.utc).replace(microsecond=0)
        try:
            updated = current.model_copy(update={**dict(changes), "updated_at": stamp})
            # model_copy does not validate, so validate explicitly.
            updated = Alert.model_validate(updated.model_dump())
        except Exception as exc:
            raise AlertError(f"Invalid change to alert '{alert_id}': {exc}") from exc

        alerts = [updated if a.alert_id == alert_id else a for a in alerts]
        self._persist(alerts, updated_at=stamp)
        return updated


    # ------------------------------------------------------------------ #
    # Lifecycle transitions
    # ------------------------------------------------------------------ #
    def acknowledge_alert(
        self,
        alert_id: str,
        *,
        note: str | None = None,
        now: datetime | None = None,
    ) -> Alert:
        """Acknowledge an alert: the operator has seen it and taken it on.

        ``NEW``, ``ACTIVE`` and ``ACKNOWLEDGED`` (re-acknowledging is a no-op
        that keeps the original timestamp) are accepted.

        Raises:
            AlertUnknownError: If the alert does not exist.
            AlertTransitionError: If the alert is already resolved or expired.
        """
        alerts = list(self._load_or_raise().alerts)
        current = self._find(alerts, alert_id)
        if current.status is AlertStatus.ACKNOWLEDGED:
            return current
        self._check_transition(current.status, AlertStatus.ACKNOWLEDGED)

        stamp = now or datetime.now(timezone.utc).replace(microsecond=0)
        changes: dict[str, Any] = {
            "status": AlertStatus.ACKNOWLEDGED,
            "acknowledged_at": stamp,
            "updated_at": stamp,
        }
        if note:
            changes["resolved_reason"] = f"acknowledged: {note}"
        updated = current.model_copy(update=changes)
        alerts = [updated if a.alert_id == alert_id else a for a in alerts]
        self._persist(alerts, updated_at=stamp)
        return updated

    def resolve_alert(
        self,
        alert_id: str,
        *,
        reason: str | None = None,
        now: datetime | None = None,
    ) -> Alert:
        """Resolve an alert: the threat has been dealt with.

        Raises:
            AlertUnknownError: If the alert does not exist.
            AlertTransitionError: If the alert is already resolved or expired.
        """
        alerts = list(self._load_or_raise().alerts)
        current = self._find(alerts, alert_id)
        if current.status is AlertStatus.RESOLVED:
            return current
        self._check_transition(current.status, AlertStatus.RESOLVED)

        stamp = now or datetime.now(timezone.utc).replace(microsecond=0)
        updated = current.model_copy(
            update={
                "status": AlertStatus.RESOLVED,
                "resolved_at": stamp,
                "updated_at": stamp,
                "resolved_reason": reason or "resolved by operator",
            }
        )
        alerts = [updated if a.alert_id == alert_id else a for a in alerts]
        self._persist(alerts, updated_at=stamp)
        return updated

    def promote_alert(self, alert_id: str, *, now: datetime | None = None) -> Alert:
        """Move a ``NEW`` or ``ACTIVE`` alert to ``ACTIVE``.

        Promoting an alert that is already ``ACTIVE`` is a no-op, matching
        :meth:`acknowledge_alert` and :meth:`resolve_alert`.

        Raises:
            AlertUnknownError: If the alert does not exist.
            AlertTransitionError: If the transition is not allowed.
        """
        alerts = list(self._load_or_raise().alerts)
        current = self._find(alerts, alert_id)
        if current.status is AlertStatus.ACTIVE:
            return current
        self._check_transition(current.status, AlertStatus.ACTIVE)
        stamp = now or datetime.now(timezone.utc).replace(microsecond=0)
        updated = current.model_copy(update={"status": AlertStatus.ACTIVE, "updated_at": stamp})
        alerts = [updated if a.alert_id == alert_id else a for a in alerts]
        self._persist(alerts, updated_at=stamp)
        return updated


    def expire_alerts(
        self,
        *,
        now: datetime | None = None,
        hours: float | None = None,
    ) -> list[Alert]:
        """Expire open alerts that are past their validity or too old (R6).

        An alert is expired when either its ``valid_until`` has passed, or it has
        not been refreshed for ``hours`` (defaulting to the rule configuration's
        ``expire_after_hours``). Acknowledged alerts are skipped because
        acknowledgement is an operator decision, not a stale-record artefact.

        Returns:
            The alerts expired by this call, in store order.
        """
        alerts = list(self._load_or_raise().alerts)
        stamp = now or datetime.now(timezone.utc).replace(microsecond=0)
        window = (
            float(hours) if hours is not None else float(self.rule_config.expire_after_hours)
        )

        expired: list[Alert] = []
        updated_alerts: list[Alert] = []
        for alert in alerts:
            if alert.status is not AlertStatus.ACKNOWLEDGED and alert.status.is_open:
                past_validity = alert.valid_until <= stamp
                too_old = (stamp - alert.updated_at).total_seconds() / 3600.0 >= window
                if past_validity or too_old:
                    reason = (
                        "validity window ended"
                        if past_validity
                        else f"not refreshed for more than {window:.2f} h"
                    )
                    updated = alert.model_copy(
                        update={
                            "status": AlertStatus.EXPIRED,
                            "resolved_at": stamp,
                            "updated_at": stamp,
                            "resolved_reason": f"expired: {reason}",
                        }
                    )
                    expired.append(updated)
                    updated_alerts.append(updated)
                    continue
            updated_alerts.append(alert)

        if expired:
            self._persist(updated_alerts, updated_at=stamp)
        return expired

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def get_alert(self, alert_id: str) -> Alert:
        """Return one alert.

        Raises:
            AlertUnknownError: If the alert does not exist.
        """
        return self._find(self._load_or_raise().alerts, alert_id)

    def get_active_alerts(self) -> list[Alert]:
        """Return the alerts that still need operator attention.

        "Active" means ``NEW``, ``ACTIVE`` or ``ACKNOWLEDGED`` - i.e. anything
        not yet resolved or expired. An operator who has acknowledged an alert
        still needs to see it until they resolve it.
        """
        return [alert for alert in self._load_or_raise().alerts if alert.status.is_open]

    def get_alert_history(self, *, include_open: bool = True) -> list[Alert]:
        """Return the alert history, newest first.

        Args:
            include_open: When False, only terminal alerts are returned.
        """
        alerts = [
            alert
            for alert in self._load_or_raise().alerts
            if include_open or alert.status.is_terminal
        ]
        return sorted(alerts, key=lambda a: (a.created_at, a.alert_id), reverse=True)

    def summary(self) -> AlertSummary:
        """Return the dashboard headline counters.

        Every number is a count of stored alerts - no estimate, no prediction.
        """
        result = self.alert_store.load()
        if result is None:
            return AlertSummary(
                total_alerts=0,
                active_alerts=0,
                severe_alerts=0,
                acknowledged_alerts=0,
                resolved_alerts=0,
                expired_alerts=0,
                tracked_events=0,
                predicted_threats=0,
                highest_severity=None,
                source=str(self.alert_store.path),
                message="No alerts available yet. Run 'python scripts/generate_alerts.py'.",
            )

        alerts = result.alerts
        open_alerts = [alert for alert in alerts if alert.status.is_open]
        severe = [
            alert
            for alert in open_alerts
            if alert.severity.rank >= AlertSeverity.SEVERE_WARNING.rank
        ]
        return AlertSummary(
            total_alerts=len(alerts),
            active_alerts=len(open_alerts),
            severe_alerts=len(severe),
            acknowledged_alerts=sum(1 for a in alerts if a.status is AlertStatus.ACKNOWLEDGED),
            resolved_alerts=sum(1 for a in alerts if a.status is AlertStatus.RESOLVED),
            expired_alerts=sum(1 for a in alerts if a.status is AlertStatus.EXPIRED),
            tracked_events=len({a.track_id for a in open_alerts if a.track_id}),
            predicted_threats=sum(1 for a in open_alerts if a.has_prediction),
            highest_severity=(
                max((a.severity for a in open_alerts), key=lambda s: s.rank)
                if open_alerts
                else None
            ),
            by_status=[
                StatusCount(status=status, count=sum(1 for a in alerts if a.status is status))
                for status in AlertStatus
            ],
            by_severity=[
                SeverityCount(
                    severity=severity,
                    count=sum(1 for a in alerts if a.severity is severity),
                )
                for severity in AlertSeverity
            ],
            source=str(self.alert_store.path),
            generated_at=result.generated_at,
            message=None if alerts else "The alert store exists but holds no alerts.",
        )

    # ------------------------------------------------------------------ #
    # Simulation support
    # ------------------------------------------------------------------ #
    def with_downscaled_threat(
        self, lookup: Mapping[str, tuple[float, float | None]]
    ) -> "AlertService":
        """Return a copy of this service carrying a downscaled-threat lookup.

        The Phase 4 prototype does not persist a per-event downscaled field, so
        a caller (the simulation) may supply one. It is a plain mapping, and the
        service still records ``downscaling`` in ``source_components`` only for
        events the mapping actually covers.
        """
        clone = AlertService(
            alert_store=self.alert_store,
            event_store=self.event_store,
            track_store=self.track_store,
            gnn_service=self.gnn_service,
            notification_providers=list(self.notification_providers),
            risk_config=self.risk_config,
            rule_config=self.rule_config,
        )
        clone._downscaled_threat_lookup = dict(lookup)
        return clone

        """Return the alerts that still need operator attention.

        "Active" means ``NEW``, ``ACTIVE`` or ``ACKNOWLEDGED`` - i.e. anything
        not yet resolved or expired. An operator who has acknowledged an alert
        still needs to see it until they resolve it.
        """
        return [alert for alert in self._load_or_raise().alerts if alert.status.is_open]

    def get_alert_history(self, *, include_open: bool = True) -> list[Alert]:
        """Return the alert history, newest first.

        Args:
            include_open: When False, only terminal alerts are returned.
        """
        alerts = [
            alert
            for alert in self._load_or_raise().alerts
            if include_open or alert.status.is_terminal
        ]
        return sorted(alerts, key=lambda a: (a.created_at, a.alert_id), reverse=True)

