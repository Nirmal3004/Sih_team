/**
 * Alert dashboard panel (SIH26078, Phase 5).
 *
 * Renders what the backend alerting layer produced: the headline counters, the
 * alert list with location / movement / intensity / trend / status, and the
 * acknowledge & resolve workflow.
 *
 * Three things this panel deliberately does **not** do:
 *  - it never shows an "accuracy" figure - the risk score is a rule-based
 *    triage aid, so any percentage would be fabricated;
 *  - it never calls an alert severity an official warning;
 *  - it never auto-acknowledges or auto-resolves anything - those are explicit
 *    operator actions.
 */
import { useCallback, useEffect, useMemo, useState } from "react";

import {
  acknowledgeAlert,
  describeApiError,
  fetchAlerts,
  fetchAlertSummary,
  generateAlerts,
  resolveAlert,
  simulateAlerts,
} from "../services/alertsApi";
import type {
  Alert,
  AlertSeverity,
  AlertSummary,
  SimulationResponse,
} from "../types/api";

import AlertDetails from "./AlertDetails";
import RiskSummary from "./RiskSummary";

/** Most severe first - the list is sorted by this order. */
const SEVERITY_ORDER: AlertSeverity[] = [
  "EXTREME",
  "SEVERE_WARNING",
  "WARNING",
  "ADVISORY",
  "WATCH",
  "INFO",
];

type PanelState =
  | { status: "loading" }
  | {
      status: "ready";
      alerts: Alert[];
      summary: AlertSummary | null;
      message: string | null;
      generatedAt: string | null;
      notice: string;
      selectedId: string | null;
      busy: string | null;
      actionError: string | null;
      note: string | null;
      simulation: SimulationResponse | null;
    }
  | { status: "error"; message: string };

function fmtTime(value: string | null): string {
  return value ? new Date(value).toLocaleString() : "n/a";
}

function trendLabel(alert: Alert): string {
  if (alert.predicted_intensity === null) {
    return "no prediction";
  }
  const delta = alert.predicted_intensity - alert.intensity;
  if (Math.abs(delta) < 0.01) {
    return "steady (predicted)";
  }
  return `${delta >= 0 ? "+" : ""}${delta.toFixed(2)} predicted`;
}

function Counter({
  label,
  value,
  emphasis,
}: {
  label: string;
  value: number;
  emphasis?: boolean;
}): React.JSX.Element {
  return (
    <div className={emphasis ? "counter counter-emphasis" : "counter"}>
      <span className="counter-value">{value}</span>
      <span className="counter-label">{label}</span>
    </div>
  );
}


function AlertRow({
  alert,
  selected,
  onSelect,
}: {
  alert: Alert;
  selected: boolean;
  onSelect: (alertId: string) => void;
}): React.JSX.Element {
  return (
    <tr
      className={selected ? "alert-row alert-row-selected" : "alert-row"}
      onClick={() => onSelect(alert.alert_id)}
    >
      <td>
        <span className={`badge badge-severity-${alert.severity.toLowerCase()}`}>
          {alert.severity}
        </span>
      </td>
      <td>{alert.event_type.replace(/_/g, " ")}</td>
      <td>
        {alert.current_lat.toFixed(2)} N, {alert.current_lon.toFixed(2)} E
      </td>
      <td>
        {alert.predicted_lat === null || alert.predicted_lon === null
          ? "n/a"
          : `${alert.predicted_lat.toFixed(2)} N, ${alert.predicted_lon.toFixed(2)} E`}
      </td>
      <td>{alert.intensity.toFixed(2)}</td>
      <td>{trendLabel(alert)}</td>
      <td>{alert.movement_speed_kmh.toFixed(1)} km/h</td>
      <td>{alert.movement_direction}</td>
      <td>
        <span className={`badge badge-alert-${alert.status.toLowerCase()}`}>{alert.status}</span>
      </td>
      <td>{fmtTime(alert.created_at)}</td>
    </tr>
  );
}

export default function AlertsPanel(): React.JSX.Element {
  const [state, setState] = useState<PanelState>({ status: "loading" });

  /** Merge a partial update into the ready state, ignoring other states. */
  const patch = useCallback((update: Partial<Extract<PanelState, { status: "ready" }>>) => {
    setState((previous) =>
      previous.status === "ready" ? { ...previous, ...update } : previous,
    );
  }, []);

  const load = useCallback(async (): Promise<void> => {
    setState({ status: "loading" });
    try {
      const [list, summary] = await Promise.all([fetchAlerts(), fetchAlertSummary()]);
      setState((previous) => ({
        status: "ready",
        alerts: list.alerts,
        summary: summary.summary,
        message: list.message,
        generatedAt: list.generated_at,
        notice: list.notice,
        selectedId: previous.status === "ready" ? previous.selectedId : null,
        busy: null,
        actionError: null,
        note: null,
        simulation: previous.status === "ready" ? previous.simulation : null,
      }));
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const sorted = useMemo<Alert[]>(() => {
    if (state.status !== "ready") {
      return [];
    }
    return [...state.alerts].sort((a, b) => {
      const bySeverity =
        SEVERITY_ORDER.indexOf(a.severity) - SEVERITY_ORDER.indexOf(b.severity);
      return bySeverity !== 0 ? bySeverity : b.created_at.localeCompare(a.created_at);
    });
  }, [state]);

  const selectedId = state.status === "ready" ? state.selectedId : null;
  const selected = useMemo(
    () => sorted.find((alert) => alert.alert_id === selectedId) ?? null,
    [sorted, selectedId],
  );

  const runPipeline = useCallback(
    async (dryRun: boolean): Promise<void> => {
      patch({ busy: "generate", actionError: null, note: null });
      try {
        const result = await generateAlerts({ dry_run: dryRun });
        const note = dryRun
          ? `Dry run: ${result.created_count} alert(s) would be created, ${result.updated_count} updated, ${result.suppressed_count} suppressed. Nothing was written.`
          : `Created ${result.created_count}, updated ${result.updated_count}, suppressed ${result.suppressed_count}, ${result.notification_count} notification(s) recorded locally.`;
        await load();
        patch({ note });
      } catch (error) {
        patch({ busy: null, actionError: describeApiError(error) });
      }
    },
    [load, patch],
  );

  const runSimulation = useCallback(async (): Promise<void> => {
    patch({ busy: "simulate", actionError: null });
    try {
      // Dry run: a demo must never mutate the real store from a button click.
      const result = await simulateAlerts({ dry_run: true });
      patch({ busy: null, simulation: result });
    } catch (error) {
      patch({ busy: null, actionError: describeApiError(error) });
    }
  }, [patch]);

  const act = useCallback(
    async (alertId: string, action: "acknowledge" | "resolve"): Promise<void> => {
      patch({ busy: alertId, actionError: null, note: null });
      try {
        const response =
          action === "acknowledge"
            ? await acknowledgeAlert(alertId, "acknowledged from the dashboard")
            : await resolveAlert(alertId, "resolved from the dashboard");
        await load();
        patch({ note: response.message });
      } catch (error) {
        patch({ busy: null, actionError: describeApiError(error) });
      }
    },
    [load, patch],
  );



  if (state.status === "loading") {
    return (
      <section className="panel" aria-labelledby="alerts-heading">
        <header className="panel-header">
          <h2 id="alerts-heading">Alerts</h2>
        </header>
        <p className="status status-loading" role="status">
          Loading alerts ...
        </p>
      </section>
    );
  }

  if (state.status === "error") {
    return (
      <section className="panel" aria-labelledby="alerts-heading">
        <header className="panel-header">
          <h2 id="alerts-heading">Alerts</h2>
          <button type="button" className="button" onClick={() => void load()}>
            Retry
          </button>
        </header>
        <p className="status status-error" role="alert">
          {state.message}
        </p>
      </section>
    );
  }

  const { summary } = state;
  const busy = state.busy !== null;

  return (
    <section className="panel" aria-labelledby="alerts-heading">
      <header className="panel-header">
        <h2 id="alerts-heading">Extreme weather alerts</h2>
        <div className="panel-actions">
          <button
            type="button"
            className="button"
            disabled={busy}
            onClick={() => void runPipeline(true)}
          >
            Dry run
          </button>
          <button
            type="button"
            className="button"
            disabled={busy}
            onClick={() => void runPipeline(false)}
          >
            Generate alerts
          </button>
          <button
            type="button"
            className="button"
            disabled={busy}
            onClick={() => void runSimulation()}
          >
            Simulate lifecycle
          </button>
          <button type="button" className="button" onClick={() => void load()}>
            Refresh
          </button>
        </div>
      </header>

      {summary ? (
        <div className="counters">
          <Counter label="Active alerts" value={summary.active_alerts} emphasis />
          <Counter
            label="Severe alerts"
            value={summary.severe_alerts}
            emphasis={summary.severe_alerts > 0}
          />
          <Counter label="Events tracked" value={summary.tracked_events} />
          <Counter label="Predicted threats" value={summary.predicted_threats} />
          <Counter label="Acknowledged" value={summary.acknowledged_alerts} />
          <Counter label="Resolved" value={summary.resolved_alerts} />
          <Counter label="Expired" value={summary.expired_alerts} />
        </div>
      ) : null}

      {summary?.highest_severity ? (
        <p className="hint">
          Highest open severity: <strong>{summary.highest_severity}</strong> - a project
          prototype category, not an official IMD or WMO warning.
        </p>
      ) : null}

      {state.message ? <p className="hint">{state.message}</p> : null}
      {state.generatedAt ? (
        <p className="hint">Alert run: {fmtTime(state.generatedAt)}</p>
      ) : null}
      {state.note ? <p className="hint">{state.note}</p> : null}
      {state.actionError ? (
        <p className="status status-error" role="alert">
          {state.actionError}
        </p>
      ) : null}

      {sorted.length === 0 ? (
        <p className="hint">
          No alerts stored. Run <code>python scripts/generate_alerts.py</code> after the
          anomaly detection and tracking steps.
        </p>
      ) : (
        <div className="table-scroll">
          <table className="alert-table">
            <caption>
              Stored alerts. Severity is a project prototype category, not an official
              warning. Select a row for its details, risk explanation and timeline.
            </caption>
            <thead>
              <tr>
                <th scope="col">Severity</th>
                <th scope="col">Event type</th>
                <th scope="col">Current location</th>
                <th scope="col">Predicted location</th>
                <th scope="col">Intensity</th>
                <th scope="col">Trend</th>
                <th scope="col">Speed</th>
                <th scope="col">Direction</th>
                <th scope="col">Status</th>
                <th scope="col">Created</th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((alert) => (
                <AlertRow
                  key={alert.alert_id}
                  alert={alert}
                  selected={alert.alert_id === selectedId}
                  onSelect={(id) => patch({ selectedId: id === selectedId ? null : id })}
                />
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selected ? (
        <>
          <AlertDetails
            alert={selected}
            busy={state.busy === selected.alert_id}
            onAcknowledge={() => void act(selected.alert_id, "acknowledge")}
            onResolve={() => void act(selected.alert_id, "resolve")}
          />
          <RiskSummary alert={selected} />
        </>
      ) : null}

      {state.simulation ? (
        <div className="simulation">
          <h3>{state.simulation.label}</h3>
          <p className="hint">{state.simulation.disclaimer}</p>
          <ol className="simulation-steps">
            {state.simulation.steps.map((step) => (
              <li key={`${step.index}-${step.stage}`}>
                <strong>{step.stage}</strong>{" "}
                <span className="muted">{fmtTime(step.timestamp)}</span>
                <br />
                {step.description}
              </li>
            ))}
          </ol>
        </div>
      ) : null}

      <p className="hint">{state.notice}</p>
    </section>
  );
}

