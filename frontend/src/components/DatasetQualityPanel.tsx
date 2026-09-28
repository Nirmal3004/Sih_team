/**
 * Dataset quality panel (SIH26078, Phase 6).
 *
 * Shows the quality report of one dataset: what passed, what warned, and what
 * blocked. Errors and warnings are rendered separately and verbatim, because
 * the whole point of a quality report is that the operator reads the actual
 * message rather than a red dot.
 */
import { useCallback, useEffect, useState } from "react";

import { describeApiError, fetchDatasetQuality } from "../services/dataApi";
import type { QualityReport } from "../types/api";

type State =
  | { status: "loading" }
  | { status: "ready"; report: QualityReport }
  | { status: "error"; message: string };

function pct(value: number | undefined): string {
  return value === undefined ? "n/a" : `${(value * 100).toFixed(2)}%`;
}

export default function DatasetQualityPanel({
  datasetId,
}: {
  datasetId: string;
}): React.JSX.Element {
  const [state, setState] = useState<State>({ status: "loading" });

  const load = useCallback(async (): Promise<void> => {
    setState({ status: "loading" });
    try {
      const report = await fetchDatasetQuality(datasetId);
      setState({ status: "ready", report });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, [datasetId]);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <section className="panel" aria-labelledby="quality-heading">
      <header className="panel-header">
        <h2 id="quality-heading">Data quality</h2>
        <div className="panel-actions">
          {state.status === "ready" ? (
            <span
              className={
                state.report.valid ? "badge badge-source-ok" : "badge badge-source-missing"
              }
            >
              {state.report.valid ? "passed" : "blocked"}
            </span>
          ) : null}
          <button type="button" className="button" onClick={() => void load()}>
            Re-check
          </button>
        </div>
      </header>

      {state.status === "loading" ? (
        <p className="status status-loading" role="status">
          Re-running the quality check ...
        </p>
      ) : null}
      {state.status === "error" ? (
        <p className="status status-error" role="alert">
          {state.message}
        </p>
      ) : null}

      {state.status === "ready" ? (
        <>
          <dl className="kv">
            <div>
              <dt>Missing fraction</dt>
              <dd>{pct(state.report.missing_fraction)}</dd>
            </div>
            <div>
              <dt>Time steps</dt>
              <dd>{state.report.time_steps}</dd>
            </div>
            <div>
              <dt>Spatial points</dt>
              <dd>{state.report.spatial_points}</dd>
            </div>
            <div>
              <dt>Errors</dt>
              <dd>{state.report.error_count ?? state.report.errors.length}</dd>
            </div>
            <div>
              <dt>Warnings</dt>
              <dd>{state.report.warning_count ?? state.report.warnings.length}</dd>
            </div>
            <div>
              <dt>Duplicate stamps</dt>
              <dd>{state.report.duplicate_timestamps ?? 0}</dd>
            </div>
          </dl>

          {state.report.errors.length > 0 ? (
            <div className="quality-block">
              <h3>Blocking errors</h3>
              <ul>
                {state.report.errors.map((error, index) => (
                  <li key={`${index}-${error.slice(0, 24)}`}>{error}</li>
                ))}
              </ul>
              <p className="hint">
                A dataset with errors must not be analysed. Fix the source file and ingest
                it again.
              </p>
            </div>
          ) : null}

          {state.report.warnings.length > 0 ? (
            <div className="quality-block">
              <h3>Warnings</h3>
              <ul>
                {state.report.warnings.map((warning, index) => (
                  <li key={`${index}-${warning.slice(0, 24)}`}>{warning}</li>
                ))}
              </ul>
              <p className="hint">
                Warnings do not block processing, but they were recorded rather than fixed:
                the checks report, they never silently alter the data.
              </p>
            </div>
          ) : null}

          {state.report.missing_standard_variables &&
          state.report.missing_standard_variables.length > 0 ? (
            <p className="hint">
              Not present in this dataset (reported, not invented):{" "}
              {state.report.missing_standard_variables.join(", ")}.
            </p>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
