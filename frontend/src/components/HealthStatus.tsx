/**
 * Backend health panel.
 *
 * Demonstrates the three states every API-backed panel in this dashboard must
 * handle: loading, success and error (including "backend not running").
 */
import { useCallback, useEffect, useState } from "react";

import { describeApiError, fetchHealth, API_BASE_URL } from "../services/api";
import type { HealthResponse } from "../types/api";

type HealthState =
  | { status: "loading" }
  | { status: "success"; data: HealthResponse; checkedAt: Date }
  | { status: "error"; message: string };

export default function HealthStatus(): React.JSX.Element {
  const [state, setState] = useState<HealthState>({ status: "loading" });

  const checkHealth = useCallback(async (): Promise<void> => {
    setState({ status: "loading" });
    try {
      const data = await fetchHealth();
      setState({ status: "success", data, checkedAt: new Date() });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, []);

  useEffect(() => {
    void checkHealth();
  }, [checkHealth]);

  return (
    <section className="panel" aria-labelledby="health-heading">
      <header className="panel-header">
        <h2 id="health-heading">Backend health</h2>
        <button type="button" className="button" onClick={() => void checkHealth()}>
          Refresh
        </button>
      </header>

      {state.status === "loading" && (
        <p className="status status-loading" role="status">
          Checking {API_BASE_URL}/health ...
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
            <span className="badge badge-ok">{state.data.status}</span> {state.data.service}{" "}
            <span className="muted">({state.data.project})</span>
          </p>
          <dl className="kv">
            <div>
              <dt>Version</dt>
              <dd>{state.data.version}</dd>
            </div>
            <div>
              <dt>Environment</dt>
              <dd>{state.data.environment}</dd>
            </div>
            <div>
              <dt>Last check</dt>
              <dd>{state.checkedAt.toLocaleTimeString()}</dd>
            </div>
            <div>
              <dt>API</dt>
              <dd>{API_BASE_URL}</dd>
            </div>
          </dl>
        </div>
      )}
    </section>
  );
}
