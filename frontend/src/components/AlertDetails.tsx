/**
 * Alert details view (SIH26078, Phase 5).
 *
 * Shows one alert in full: location, predicted location, movement, intensity,
 * affected zone, which pipeline components contributed, and the acknowledge /
 * resolve workflow.
 *
 * The two action buttons are the only writes this dashboard performs, and they
 * are always explicit. `RESOLVED` and `EXPIRED` are terminal, so the buttons
 * are disabled in those states rather than letting the backend reject the call.
 */
import type { Alert } from "../types/api";

import AlertTimeline from "./AlertTimeline";

function fmt(value: string | null): string {
  return value ? new Date(value).toLocaleString() : "n/a";
}

const COMPONENT_LABELS: Record<string, string> = {
  events: "Phase 1 - anomaly detection",
  tracking: "Phase 2 - event tracking",
  gnn: "Phase 3 - GNN next-state prediction",
  downscaling: "Phase 4 - AI downscaling (~5 km)",
};

const CONFIDENCE_LABELS: Record<string, string> = {
  HIGH: "all available components contributed",
  MEDIUM: "one optional component contributed",
  LOW: "detection and tracking only - no forward-looking input was used",
};

export default function AlertDetails({
  alert,
  busy,
  onAcknowledge,
  onResolve,
}: {
  alert: Alert;
  busy: boolean;
  onAcknowledge: () => void;
  onResolve: () => void;
}): React.JSX.Element {
  const isOpen = alert.status === "NEW" || alert.status === "ACTIVE" || alert.status === "ACKNOWLEDGED";
  const canAcknowledge = alert.status === "NEW" || alert.status === "ACTIVE";
  const canResolve = isOpen;

  return (
    <article className="alert-details" aria-labelledby="alert-details-heading">
      <header className="alert-details-header">
        <h3 id="alert-details-heading">{alert.title}</h3>
        <span className={`badge badge-alert-${alert.status.toLowerCase()}`}>{alert.status}</span>
      </header>

      <p className="alert-message">{alert.message}</p>

      <dl className="kv">
        <div>
          <dt>Alert id</dt>
          <dd>{alert.alert_id}</dd>
        </div>
        <div>
          <dt>Event</dt>
          <dd>
            {alert.event_id} ({alert.event_type.replace(/_/g, " ")})
          </dd>
        </div>
        <div>
          <dt>Track</dt>
          <dd>{alert.track_id ?? "not tracked"}</dd>
        </div>
        <div>
          <dt>Severity / priority</dt>
          <dd>
            {alert.severity} / {alert.priority}
          </dd>
        </div>
        <div>
          <dt>Risk score</dt>
          <dd>
            {alert.risk.score.toFixed(2)} / 100 &rarr; {alert.risk.level}
          </dd>
        </div>
        <div>
          <dt>Current location</dt>
          <dd>
            {alert.current_lat.toFixed(3)} N, {alert.current_lon.toFixed(3)} E
          </dd>
        </div>
        <div>
          <dt>Predicted location</dt>
          <dd>
            {alert.predicted_lat === null || alert.predicted_lon === null
              ? "no GNN prediction available"
              : `${alert.predicted_lat.toFixed(3)} N, ${alert.predicted_lon.toFixed(3)} E`}
          </dd>
        </div>
        <div>
          <dt>Intensity</dt>
          <dd>
            {alert.intensity.toFixed(2)}
            {alert.predicted_intensity !== null
              ? ` (predicted ${alert.predicted_intensity.toFixed(2)})`
              : ""}
          </dd>
        </div>
        <div>
          <dt>Movement</dt>
          <dd>
            {alert.movement_speed_kmh.toFixed(2)} km/h {alert.movement_direction}
          </dd>
        </div>
        <div>
          <dt>Affected zone</dt>
          <dd>radius {alert.affected_radius_km.toFixed(1)} km around the event centre</dd>
        </div>
        <div>
          <dt>Input completeness</dt>
          <dd>
            {alert.confidence_label} - {CONFIDENCE_LABELS[alert.confidence_label]}
          </dd>
        </div>
        <div>
          <dt>Valid until</dt>
          <dd>{fmt(alert.valid_until)}</dd>
        </div>
        <div>
          <dt>Acknowledged</dt>
          <dd>{fmt(alert.acknowledged_at)}</dd>
        </div>
        <div>
          <dt>Resolved</dt>
          <dd>{alert.resolved_at ? fmt(alert.resolved_at) : "not resolved"}</dd>
        </div>
      </dl>

      <div className="alert-components">
        <h4>Pipeline components used</h4>
        {alert.source_components.length === 0 ? (
          <p className="hint">No component provenance was recorded.</p>
        ) : (
          <ul>
            {alert.source_components.map((component) => (
              <li key={component}>
                {COMPONENT_LABELS[component] ?? component}
              </li>
            ))}
          </ul>
        )}
        {["gnn", "downscaling"].some(
          (component) => !alert.source_components.includes(component as never),
        ) ? (
          <p className="hint">
            Missing optional components contributed nothing to the score - they were not
            silently assumed.
          </p>
        ) : null}
      </div>

      <AlertTimeline alert={alert} />

      <div className="alert-actions">
        <button
          type="button"
          className="button"
          disabled={!canAcknowledge || busy}
          onClick={onAcknowledge}
        >
          Acknowledge
        </button>
        <button type="button" className="button" disabled={!canResolve || busy} onClick={onResolve}>
          Resolve
        </button>
        {busy ? <span className="muted">Working ...</span> : null}
      </div>

      <p className="hint">
        Acknowledging and resolving record operator actions on a prototype alert. They are not
        a public warning acknowledgement and carry no legal or operational status.
      </p>
    </article>
  );
}
