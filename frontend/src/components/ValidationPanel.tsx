/**
 * Validation panel (SIH26078, Phase 7).
 *
 * This panel is where the project states what it can and cannot demonstrate, so
 * its rules are stricter than the other panels:
 *
 *  - a metric the backend did not compute is rendered as "Not computed", never
 *    as a zero, a dash, or an em dash that could be misread as a value;
 *  - the ground-truth status is shown **above** the numbers, so a count is never
 *    read as an accuracy;
 *  - a comparison verdict is shown as the backend stated it - "worse than
 *    persistence" is displayed, not smoothed away;
 *  - the limitations are always rendered, never collapsed.
 */
import { useCallback, useEffect, useState } from "react";

import {
  describeApiError,
  fetchValidationStatus,
  fetchValidationSummary,
} from "../services/validationApi";
import type { Phase7Summary, ValidationStatus } from "../types/api";

type State =
  | { status: "loading" }
  | { status: "ready"; summary: Phase7Summary }
  | { status: "absent"; message: string }
  | { status: "error"; message: string };

/** Render a value the backend may legitimately have refused to compute. */
function value(text: number | string | string[] | null | undefined): string {
  if (text === null || text === undefined) {
    return "Not computed";
  }
  if (Array.isArray(text)) {
    return text.length > 0 ? text.join(", ") : "none";
  }
  if (typeof text === "number") {
    return Number.isFinite(text) ? String(Number(text.toFixed(4))) : "Not computed";
  }
  return text;
}

function fraction(text: number | null | undefined): string {
  return text === null || text === undefined ? "Not computed" : `${(text * 100).toFixed(1)}%`;
}

export default function ValidationPanel(): React.JSX.Element {
  const [state, setState] = useState<State>({ status: "loading" });
  const [runStatus, setRunStatus] = useState<ValidationStatus | null>(null);

  const load = useCallback(async (): Promise<void> => {
    setState({ status: "loading" });
    try {
      setRunStatus(await fetchValidationStatus());
      setState({ status: "ready", summary: await fetchValidationSummary() });
    } catch (error) {
      const message = describeApiError(error);
      // A missing report is information, not a failure: the phase has simply not
      // been run yet.
      if (message.toLowerCase().includes("404") || message.includes("No Phase 7")) {
        setState({ status: "absent", message });
        return;
      }
      setState({ status: "error", message });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (state.status === "loading") {
    return (
      <section className="panel" aria-labelledby="validation-heading">
        <header className="panel-header">
          <h2 id="validation-heading">Validation</h2>
        </header>
        <p className="status status-loading" role="status">
          Loading the Phase 7 validation report ...
        </p>
      </section>
    );
  }

  if (state.status === "absent") {
    return (
      <section className="panel" aria-labelledby="validation-heading">
        <header className="panel-header">
          <h2 id="validation-heading">Validation</h2>
          <button type="button" className="button" onClick={() => void load()}>
            Check again
          </button>
        </header>
        <p className="status" role="status">
          No Phase 7 validation run exists yet. Run{" "}
          <code>python scripts/run_phase7_validation.py</code> to produce one.
        </p>
        <p className="hint">
          Until it has run, this project makes no claim about how well any component
          performs.
        </p>
      </section>
    );
  }

  if (state.status === "error") {
    return (
      <section className="panel" aria-labelledby="validation-heading">
        <header className="panel-header">
          <h2 id="validation-heading">Validation</h2>
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
  const truth = summary.ground_truth_availability;
  const anomaly = summary.anomaly_diagnostics;
  const tracking = summary.tracking_diagnostics;
  const gnn = summary.gnn_metrics;
  const downscaling = summary.downscaling_metrics;
  const alerts = summary.alert_diagnostics;
  const climatology = summary.climatology_diagnostics;
  const splits = summary.temporal_coverage?.splits ?? [];
  const variables: string[] = Array.isArray(climatology?.variables)
    ? [...climatology.variables]
    : [];

  return (
    <section className="panel" aria-labelledby="validation-heading">
      <header className="panel-header">
        <h2 id="validation-heading">Validation</h2>
        <div className="panel-actions">
          <span className="badge badge-synthetic">Research prototype</span>
          <span className={truth?.available ? "badge badge-source-ok" : "badge badge-source-missing"}>
            {truth?.available ? "Ground truth available" : "Ground truth unavailable"}
          </span>
          <button type="button" className="button" onClick={() => void load()}>
            Refresh
          </button>
        </div>
      </header>

      {/* The ground-truth status leads, so a count is never read as accuracy. */}
      <p className="validation-verdict" role="note">
        <strong>Ground truth: {value(truth?.status)}.</strong>{" "}
        {truth?.available
          ? "Detection skill is computed against the supplied reference catalogue."
          : "Detection skill (POD/FAR/CSI/precision/recall) is NOT computed. The counts below describe what the system produced; they are not accuracy."}
      </p>

      <h3>Dataset and coverage</h3>
      <dl className="kv">
        <div>
          <dt>Dataset</dt>
          <dd>{value(String(summary.dataset_provenance?.dataset_id ?? null))}</dd>
        </div>
        <div>
          <dt>Period</dt>
          <dd>
            {value(String(summary.dataset_provenance?.time_start ?? null))} ..{" "}
            {value(String(summary.dataset_provenance?.time_end ?? null))}
          </dd>
        </div>
        <div>
          <dt>Origin</dt>
          <dd>{value(String(summary.dataset_provenance?.origin ?? null))}</dd>
        </div>
        <div>
          <dt>Live data</dt>
          <dd>
            {summary.dataset_provenance?.is_live === true
              ? "yes"
              : "no - this project ingests files, never a live feed"}
          </dd>
        </div>
        <div>
          <dt>Samples (time)</dt>
          <dd>{value(splits.reduce((sum, s) => sum + (s.size ?? 0), 0) || null)}</dd>
        </div>
        <div>
          <dt>Variables</dt>
          <dd>
            {value(variables)}
          </dd>
        </div>
        <div>
          <dt>Missing data</dt>
          <dd>{fraction(climatology?.missing_fraction as number | undefined)}</dd>
        </div>
      </dl>

      <h3>Chronological split</h3>
      <ul className="split-list">
        {splits.map((split) => (
          <li key={split.name}>
            <strong>{split.name}</strong>: {value(split.size)} steps (
            {value(split.start)} .. {value(split.end)})
          </li>
        ))}
      </ul>
      <p className="hint">
        Strictly chronological: <code>max(train) &lt; min(validation) &lt; min(test)</code>,
        with no overlap and no random shuffling.
      </p>

      <h3>Climatology</h3>
      {climatology?.available && climatology.coverages ? (
        Object.entries(climatology.coverages).map(([name, coverage]) => (
          <p key={name} className="hint">
            <strong>{name}</strong>:{" "}
            {coverage.usable ? "usable" : "NOT USABLE"} (
            {value(coverage.buckets_sufficient)}/{value(coverage.buckets_expected)}{" "}
            buckets) {coverage.message}
          </p>
        ))
      ) : (
        <p className="hint">Not produced.</p>
      )}

      <h3>Anomaly detection and tracking</h3>
      <dl className="kv">
        <div>
          <dt>Events (validation split)</dt>
          <dd>{value(anomaly?.event_count as number | undefined)}</dd>
        </div>
        <div>
          <dt>Tracks</dt>
          <dd>{value(tracking?.track_count as number | undefined)}</dd>
        </div>
        <div>
          <dt>Unmatched events</dt>
          <dd>{value(tracking?.unmatched_events as number | undefined)}</dd>
        </div>
        <div>
          <dt>Detection skill</dt>
          <dd>Not computed - no reference catalogue</dd>
        </div>
      </dl>
      {anomaly?.available ? (
        <p className="hint">{String(anomaly.caveat ?? "")}</p>
      ) : null}
      {tracking?.available ? (
        <p className="hint">{String(tracking.caveat ?? "")}</p>
      ) : null}

      <h3>GNN versus persistence</h3>
      {gnn?.available ? (
        <>
          <p className="validation-verdict">
            <strong>
              Measured on the test split:{" "}
              {String(
                (gnn.comparison as Record<string, { verdict?: string }> | undefined)?._summary
                  ?.verdict ?? "no verdict",
              )}
              .
            </strong>
          </p>
          <dl className="kv">
            {(Object.entries(gnn.metrics ?? {}) as [string, number][])
              .filter(([key]) => key !== "sample_count")
              .map(([key, value_]) => (
                <div key={key}>
                  <dt>{key.replace(/_/g, " ")}</dt>
                  <dd>
                    {value(value_)} (persistence:{" "}
                    {value(gnn.baseline_metrics?.[key] as number | undefined)})
                  </dd>
                </div>
              ))}
          </dl>
          {Array.isArray(gnn.limitations) ? (
            <ul className="caveat-list">
              {gnn.limitations.map((item, index) => (
                <li key={index}>{String(item)}</li>
              ))}
            </ul>
          ) : null}
        </>
      ) : (
        <p className="hint">
          Not evaluated. {String(gnn?.reason ?? "This report has not been produced.")}
        </p>
      )}

      <h3>Downscaling: nearest vs cubic vs AI</h3>
      {downscaling?.available ? (
        <>
          <p className="validation-verdict">
            <strong>Measured on the test split.</strong> Best by MAE:{" "}
            {value(downscaling.winner_by_mae ?? null)}
          </p>
          <dl className="kv">
            {Object.entries(downscaling.methods ?? {}).map(([name, stats]) => (
              <div key={name}>
                <dt>{name}</dt>
                <dd>
                  MAE {value(stats?.mae)} � RMSE {value(stats?.rmse)} � corr{" "}
                  {value(stats?.correlation)}
                </dd>
              </div>
            ))}
          </dl>
          <p className="hint">
            POD/FAR:{" "}
            {downscaling.classification?.available
              ? "computed against the stated threshold"
              : "not computed - no defensible event threshold"}
          </p>
        </>
      ) : (
        <p className="hint">
          Not evaluated. {String(downscaling?.reason ?? "This report has not been produced.")}
        </p>
      )}

      <h3>Alerts</h3>
      <dl className="kv">
        <div>
          <dt>Alerts (validation split)</dt>
          <dd>{value(alerts?.alert_count as number | undefined)}</dd>
        </div>
        <div>
          <dt>Severities</dt>
          <dd>{JSON.stringify(alerts?.severity_distribution ?? {})}</dd>
        </div>
        <div>
          <dt>Trigger components</dt>
          <dd>{JSON.stringify(alerts?.trigger_components ?? {})}</dd>
        </div>
      </dl>
      {alerts?.available ? (
        <p className="hint">{String(alerts.caveat ?? "")}</p>
      ) : null}

      {runStatus && runStatus.reports_missing.length > 0 ? (
        <p className="validation-warning" role="note">
          Reports not produced: {runStatus.reports_missing.join(", ")}. A stage that
          did not run makes no claim, favourable or unfavourable.
        </p>
      ) : null}

      <h3>Limitations</h3>
      <ul className="caveat-list">
        {(summary.limitations ?? []).map((item, index) => (
          <li key={index}>{item}</li>
        ))}
      </ul>
      <p className="hint">
        Generated {value(summary.generated_at)} � package {value(summary.package_version)}
        {runStatus ? ` � run ${runStatus.reports_present.length} report(s) present` : ""}
      </p>
    </section>
  );
}

