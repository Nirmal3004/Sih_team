/**
 * Operations panel (SIH26078, Phase 10).
 *
 * Shows the measured state of the *system*: liveness, readiness, component
 * availability, artifact integrity and request metrics.
 *
 * The display rules follow the project's existing discipline:
 *
 *  - every state is one of AVAILABLE, UNAVAILABLE, STALE, DEGRADED or
 *    PROTOTYPE, and each is styled distinctly, so a missing model cannot be
 *    mistaken for a working one;
 *  - a component that has not been checked reports UNKNOWN, not AVAILABLE.
 *    "We did not look" and "it works" are different facts;
 *  - the metrics table contains only system counters. There is no accuracy
 *    figure anywhere in this panel, because none has been measured;
 *  - the prototype disclaimer is always rendered, not only in the page footer.
 */
import { useCallback, useEffect, useState } from "react";

import {
  describeApiError,
  fetchOperationsMetrics,
  fetchOperationsStatus,
  fetchIntegrity,
  fetchHealth,
} from "../services/operationsApi";
import type {
  ComponentStatus,
  IntegrityCheck,
  OperationsIntegrity,
  OperationsMetrics,
  OperationsStatus,
} from "../types/api";

type State =
  | { status: "loading" }
  | {
      status: "ready";
      health: Record<string, unknown> | null;
      readiness: OperationsStatus | null;
      metrics: OperationsMetrics | null;
      integrity: OperationsIntegrity | null;
    }
  | { status: "error"; message: string };

/** CSS class for a component state. */
function stateClass(state: string): string {
  switch (state) {
    case "AVAILABLE":
    case "READY":
    case "VALID":
      return "badge badge-ok";
    case "DEGRADED":
    case "INCOMPATIBLE":
    case "STALE":
      return "badge badge-warning";
    case "NOT_READY":
    case "UNAVAILABLE":
    case "CORRUPT":
      return "badge badge-error";
    default:
      return "badge";
  }
}

/** Render a duration in seconds as a human-readable string. */
function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) {
    return "-";
  }
  if (seconds < 1) {
    return `${Math.round(seconds * 1000)} ms`;
  }
  if (seconds < 60) {
    return `${seconds.toFixed(1)} s`;
  }
  if (seconds < 3600) {
    return `${(seconds / 60).toFixed(1)} min`;
  }
  return `${(seconds / 3600).toFixed(1)} h`;
}

/** Sum every labelled series of a counter into one number. */
function total(counter: Record<string, number> | undefined): number {
  if (!counter) {
    return 0;
  }
  return Object.values(counter).reduce((sum, value) => sum + value, 0);
}

export default function OperationsPanel() {
  const [state, setState] = useState<State>({ status: "loading" });
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    setBusy(true);
    try {
      // Each resource is fetched independently: a 404 from one of them is an
      // absent measurement, not a reason to hide the others.
      const [health, readiness, metrics, integrity] = await Promise.all([
        fetchHealth().catch(() => null),
        fetchOperationsStatus().catch(() => null),
        fetchOperationsMetrics().catch(() => null),
        fetchIntegrity().catch(() => null),
      ]);
      setState({ status: "ready", health, readiness, metrics, integrity });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    } finally {
      setBusy(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  if (state.status === "loading") {
    return (
      <section className="panel" aria-labelledby="operations-heading">
        <header className="panel-header">
          <h2 id="operations-heading">System operations</h2>
        </header>
        <p className="hint">Loading operational status...</p>
      </section>
    );
  }

  if (state.status === "error") {
    return (
      <section className="panel" aria-labelledby="operations-heading">
        <header className="panel-header">
          <h2 id="operations-heading">System operations</h2>
        </header>
        <p className="error-text">
          Could not load the operational status: {state.message}
        </p>
      </section>
    );
  }

  const { health, readiness, metrics, integrity } = state;

  return (
    <section className="panel" aria-labelledby="operations-heading">
      <header className="panel-header">
        <h2 id="operations-heading">System operations</h2>
        <div className="panel-actions">
          <span className="badge badge-synthetic">Research prototype</span>
          <span className={stateClass(readiness?.state ?? "UNKNOWN")}>
            {readiness?.state ?? "UNKNOWN"}
          </span>
        </div>
      </header>

      <dl className="kv">
        <dt>Liveness</dt>
        <dd>
          {health ? (
            <span className="badge badge-ok">AVAILABLE</span>
          ) : (
            <span className="badge badge-error">UNAVAILABLE</span>
          )}
          {health ? ` (${String(health.status ?? "ok")})` : ""}
        </dd>
        <dt>Readiness</dt>
        <dd>
          {readiness ? (
            <span className={stateClass(readiness.state)}>{readiness.state}</span>
          ) : (
            <span className="badge">UNKNOWN</span>
          )}
        </dd>
        <dt>Service</dt>
        <dd>{readiness ? `${readiness.service} ${readiness.version}` : "Unknown"}</dd>
        <dt>Environment</dt>
        <dd>{readiness?.environment ?? "Unknown"}</dd>
        <dt>Uptime</dt>
        <dd>{formatDuration(metrics?.uptime_seconds)}</dd>
      </dl>

      <h3>Components</h3>
      {readiness && readiness.components.length > 0 ? (
        <table className="table">
          <thead>
            <tr>
              <th>Component</th>
              <th>State</th>
              <th>Required</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {readiness.components.map((component: ComponentStatus) => (
              <tr key={component.name}>
                <td>{component.name}</td>
                <td>
                  <span className={stateClass(component.state)}>
                    {component.state}
                  </span>
                </td>
                <td>{component.required ? "yes" : "no"}</td>
                <td>{component.detail || "-"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : (
        <p className="hint">Component state was not reported.</p>
      )}

      {readiness && readiness.notes.length > 0 ? (
        <>
          <h3>Runtime warnings</h3>
          <ul>
            {readiness.notes.map((note) => (
              <li key={note}>{note}</li>
            ))}
          </ul>
        </>
      ) : null}

      {integrity ? (
        <>
          <h3>Artifact integrity</h3>
          <table className="table">
            <thead>
              <tr>
                <th>Artifact</th>
                <th>State</th>
                <th>Reason</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(integrity.checks).map(
                ([name, check]: [string, IntegrityCheck]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>
                      <span className={stateClass(check.state)}>{check.state}</span>
                    </td>
                    <td>{check.reason || "Passed every check."}</td>
                  </tr>
                ),
              )}
            </tbody>
          </table>
          <p className="hint">{integrity.note}</p>
        </>
      ) : null}

      {metrics ? (
        <>
          <h3>Request metrics</h3>
          <table className="table">
            <thead>
              <tr>
                <th>Counter</th>
                <th>Value</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(metrics.counters)
                .filter(([, series]) => Object.keys(series).length > 0)
                .map(([name, series]) => (
                  <tr key={name}>
                    <td>{name}</td>
                    <td>{total(series)}</td>
                  </tr>
                ))}
            </tbody>
          </table>
          <p className="hint">{metrics.note}</p>
        </>
      ) : null}

      <div className="panel-actions">
        <button type="button" onClick={refresh} disabled={busy}>
          {busy ? "Refreshing..." : "Refresh"}
        </button>
      </div>

      <p className="hint">
        {readiness?.disclaimer ??
          "Research prototype. Not an operational disaster-warning system."}{" "}
        READY means this process can serve requests; it makes no claim about
        forecast skill, and this panel shows no accuracy figure because none has
        been measured.
      </p>
    </section>
  );
}
