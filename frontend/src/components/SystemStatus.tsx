/**
 * System status panel (Phase 11) - the dashboard's first card.
 *
 * Shows what the system is, what data it is looking at, whether each optional
 * model is actually loadable, and how much it has produced.
 *
 * The rendering rule that matters: a count the backend reported as `null` is
 * shown as "unavailable", never as `0`. "Nothing stored" and "zero found" are
 * different facts, and rendering both as 0 is how a dashboard starts lying.
 */
import { useCallback, useEffect, useState } from "react";

import { describeApiError, fetchSystemStatus, API_BASE_URL } from "../services/api";
import type { SystemCounts, SystemStatusResponse } from "../types/api";

type StatusState =
  | { status: "loading" }
  | { status: "success"; data: SystemStatusResponse; checkedAt: Date }
  | { status: "error"; message: string };

/** Human wording for each data-source kind. Never shown bare. */
const SOURCE_LABELS: Record<string, string> = {
  SYNTHETIC: "Synthetic (generated locally, not observed)",
  HISTORICAL: "Historical / reanalysis (not a live feed)",
  REALTIME_FIXTURE: "Realtime fixture (simulated, not observed)",
};

const COUNT_LABELS: Array<[keyof SystemCounts, string]> = [
  ["events", "Detected events"],
  ["tracks", "Event tracks"],
  ["alerts", "Alerts"],
  ["datasets", "Datasets"],
  ["pipeline_runs", "Pipeline runs"],
  ["evaluation_runs", "Evaluation runs"],
];

/** Render a count, spelling out "unavailable" rather than showing 0. */
function formatCount(value: number | null): string {
  return value === null ? "unavailable" : String(value);
}

function readinessClass(readiness: string): string {
  if (readiness === "READY") return "badge-ok";
  if (readiness === "DEGRADED") return "badge-warn";
  return "badge-error";
}

/** One row in the component list: available, or unavailable with its reason. */
function ComponentRow({
  name,
  component,
  unavailableLabel,
}: {
  name: string;
  component: { available: boolean; state: string; detail: string; reason: string | null };
  unavailableLabel?: string;
}): React.JSX.Element {
  const badge = component.available
    ? component.state
    : (unavailableLabel ?? "UNAVAILABLE");
  return (
    <li>
      <span className={`badge ${component.available ? "badge-ok" : "badge-warn"}`}>
        {badge}
      </span>{" "}
      <strong>{name}</strong>{" "}
      <span className="muted">
        {component.available ? component.detail : (component.reason ?? "not available")}
      </span>
    </li>
  );
}

export default function SystemStatus(): React.JSX.Element {
  const [state, setState] = useState<StatusState>({ status: "loading" });

  const refresh = useCallback(async (): Promise<void> => {
    setState({ status: "loading" });
    try {
      const data = await fetchSystemStatus();
      setState({ status: "success", data, checkedAt: new Date() });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <section className="panel" aria-labelledby="system-status-heading">
      <header className="panel-header">
        <h2 id="system-status-heading">System status</h2>
        <button type="button" className="button" onClick={() => void refresh()}>
          Refresh
        </button>
      </header>

      {state.status === "loading" && (
        <p className="status status-loading" role="status">
          Checking {API_BASE_URL}/api/v1/system/status ...
        </p>
      )}

      {state.status === "error" && (
        <div className="status status-error" role="alert">
          <p className="status-title">Backend unreachable</p>
          <p>{state.message}</p>
          <p className="hint">
            Start it with <code>uvicorn extreme_weather.api.main:app --reload</code>.
          </p>
        </div>
      )}

      {state.status === "success" && (
        <div className="status status-ok" role="status">
          <p className="status-title">
            <span className={`badge ${readinessClass(state.data.readiness)}`}>
              {state.data.readiness}
            </span>{" "}
            {state.data.service} <span className="muted">({state.data.project})</span>
          </p>

          <dl className="kv">
            <div>
              <dt>Version</dt>
              <dd>
                {state.data.version}{" "}
                <span className="muted">phase {state.data.phase}</span>
              </dd>
            </div>
            <div>
              <dt>Environment</dt>
              <dd>{state.data.environment}</dd>
            </div>
            <div>
              <dt>Data source</dt>
              <dd>
                {SOURCE_LABELS[state.data.data_source.kind] ?? state.data.data_source.kind}
              </dd>
            </div>
            <div>
              <dt>Latest data</dt>
              <dd>
                {state.data.data_source.latest_timestamp ??
                  "unavailable (no run on record)"}
                {state.data.data_source.freshness
                  ? ` · ${state.data.data_source.freshness}`
                  : ""}
              </dd>
            </div>
            <div>
              <dt>Ingestion</dt>
              <dd>
                {state.data.ingestion.state}
                {state.data.ingestion.reason ? (
                  <span className="muted"> — {state.data.ingestion.reason}</span>
                ) : null}
              </dd>
            </div>
          </dl>

          <h3 className="subheading">Components</h3>
          <ul className="kv-list">
            {Object.entries(state.data.models).map(([name, component]) => (
              <ComponentRow key={name} name={name} component={component} />
            ))}
            <ComponentRow name="alerting" component={state.data.alerting} />
            <ComponentRow
              name="evaluation"
              component={state.data.evaluation}
              unavailableLabel="NOT EVALUATED"
            />
          </ul>

          <h3 className="subheading">Stored output</h3>
          <dl className="kv">
            {COUNT_LABELS.map(([key, label]) => (
              <div key={key}>
                <dt>{label}</dt>
                <dd>{formatCount(state.data.counts[key])}</dd>
              </div>
            ))}
            <div>
              <dt>Last check</dt>
              <dd>{state.checkedAt.toLocaleTimeString()}</dd>
            </div>
          </dl>

          <p className="hint">{state.data.note}</p>
          <p className="hint disclaimer">{state.data.disclaimer}</p>
        </div>
      )}
    </section>
  );
}
