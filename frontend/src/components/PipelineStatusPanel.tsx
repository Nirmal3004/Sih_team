/**
 * Pipeline status panel (SIH26078, Phase 8).
 *
 * Displays the near-real-time pipeline's state. Its display rules follow the
 * Phase 7 validation panel, because the failure modes are the same:
 *
 *  - freshness is never rounded up to "fresh". A `null` age renders as
 *    "Unknown", never as "0 min old", which would assert something about the
 *    data that the backend did not assert;
 *  - `STALE` and `EXPIRED` are visually distinct, because expired data must not
 *    be read as current at a glance;
 *  - a `SKIPPED` stage is shown as skipped with its reason, so an absent model
 *    is visible rather than looking like a stage that did nothing;
 *  - the manual "Run pipeline" control offers a dry run by default, so the
 *    default interaction writes nothing.
 */
import { useCallback, useEffect, useState } from "react";

import {
  describeApiError,
  fetchPipelineStatus,
  runPipeline,
} from "../services/pipelineApi";
import type {
  FreshnessStatus,
  PipelineRun,
  PipelineStage,
  PipelineStatus,
} from "../types/api";

type State =
  | { status: "loading" }
  | { status: "ready"; data: PipelineStatus }
  | { status: "error"; message: string };

/** Human labels for each stage. */
const STAGE_LABELS: Record<string, string> = {
  ingest: "Ingestion",
  validate: "Quality gate",
  preprocess: "Preprocessing",
  anomaly: "Anomaly detection",
  track: "Event tracking",
  gnn: "GNN prediction",
  downscale: "AI downscaling",
  alert: "Alerts",
};

/** Render an age the backend reported, or say it is unknown. */
function formatAge(minutes: number | null | undefined): string {
  if (minutes === null || minutes === undefined) {
    return "Unknown";
  }
  if (minutes < 1) {
    return "Less than a minute";
  }
  if (minutes < 60) {
    return `${Math.round(minutes)} min`;
  }
  if (minutes < 1440) {
    return `${(minutes / 60).toFixed(1)} h`;
  }
  return `${(minutes / 1440).toFixed(1)} days`;
}

/** Render a timestamp, or say there is none. */
function formatTimestamp(value: string | null | undefined): string {
  return value ? new Date(value).toISOString().replace("T", " ").slice(0, 19) : "None";
}

/** Freshness badge class, so stale and expired never look alike. */
function freshnessClass(value: FreshnessStatus): string {
  switch (value) {
    case "FRESH":
      return "badge badge-ok";
    case "STALE":
      return "badge badge-warning";
    case "EXPIRED":
      return "badge badge-error";
    default:
      return "badge";
  }
}

export default function PipelineStatusPanel() {
  const [state, setState] = useState<State>({ status: "loading" });
  const [lastRun, setLastRun] = useState<PipelineRun | null>(null);
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);
  const [dryRun, setDryRun] = useState(true);

  const refresh = useCallback(async () => {
    try {
      const data = await fetchPipelineStatus();
      setState({ status: "ready", data });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const triggerRun = useCallback(async () => {
    setBusy(true);
    setActionError(null);
    try {
      const run = await runPipeline({ hours: 6 }, dryRun);
      setLastRun(run);
      await refresh();
    } catch (error) {
      setActionError(describeApiError(error));
    } finally {
      setBusy(false);
    }
  }, [dryRun, refresh]);

  if (state.status === "loading") {
    return (
      <section className="panel" aria-labelledby="pipeline-heading">
        <header className="panel-header">
          <h2 id="pipeline-heading">Near-real-time pipeline</h2>
        </header>
        <p className="hint">Loading pipeline status...</p>
      </section>
    );
  }

  if (state.status === "error") {
    return (
      <section className="panel" aria-labelledby="pipeline-heading">
        <header className="panel-header">
          <h2 id="pipeline-heading">Near-real-time pipeline</h2>
        </header>
        <p className="error-text">Could not load the pipeline status: {state.message}</p>
      </section>
    );
  }

  const { data } = state;
  const stages: PipelineStage[] = lastRun?.stages ?? data.stages;
  const freshness = lastRun ? lastRun.freshness : data.freshness.status;
  const age = lastRun ? lastRun.age_minutes : data.freshness.age_minutes;
  const latest = lastRun?.source_timestamp ?? data.latest_timestamp;
  const warnings = lastRun ? lastRun.warnings : data.warnings;
  const errors = lastRun ? lastRun.errors : data.errors;

  return (
    <section className="panel" aria-labelledby="pipeline-heading">
      <header className="panel-header">
        <h2 id="pipeline-heading">Near-real-time pipeline</h2>
        <div className="panel-actions">
          <span className="badge badge-synthetic">Research prototype</span>
          <span className={freshnessClass(freshness)}>{freshness}</span>
        </div>
      </header>

      <dl className="kv">
        <dt>State</dt>
        <dd>{lastRun ? lastRun.status : data.current_state}</dd>
        <dt>Provider</dt>
        <dd>
          {data.provider.name} {data.provider.available ? "" : "(unavailable)"}
        </dd>
        <dt>Latest data</dt>
        <dd>{formatTimestamp(latest)}</dd>
        <dt>Ingestion age</dt>
        <dd>{formatAge(age)}</dd>
        <dt>Last successful run</dt>
        <dd>{data.last_successful_run ?? "None yet"}</dd>
        <dt>Current stage</dt>
        <dd>{lastRun?.current_stage ?? data.current_stage ?? "Idle"}</dd>
      </dl>

      <p className="hint">
        Data is considered fresh up to{" "}
        {data.freshness.thresholds.fresh_within_hours} h old and stale up to{" "}
        {data.freshness.thresholds.stale_within_hours} h old. Beyond that it is
        expired and is not presented as current.
      </p>

      <div className="panel-actions">
        <label>
          <input
            type="checkbox"
            checked={dryRun}
            onChange={(event) => setDryRun(event.target.checked)}
          />{" "}
          Dry run (writes nothing)
        </label>
        <button type="button" onClick={triggerRun} disabled={busy}>
          {busy ? "Running..." : "Run pipeline"}
        </button>
        <button type="button" onClick={refresh} disabled={busy}>
          Refresh
        </button>
      </div>

      {actionError ? <p className="error-text">{actionError}</p> : null}

      {lastRun ? (
        <p className="hint">
          Last requested run: <code>{lastRun.run_id}</code>
          {lastRun.dry_run ? " (dry run - no artifacts were written)" : ""}
        </p>
      ) : null}

      <h3>Stages</h3>
      {stages.length === 0 ? (
        <p className="hint">The pipeline has not run yet.</p>
      ) : (
        <table className="table">
          <thead>
            <tr>
              <th>Stage</th>
              <th>Status</th>
              <th>Duration</th>
              <th>Detail</th>
            </tr>
          </thead>
          <tbody>
            {stages.map((stage) => (
              <tr key={stage.name}>
                <td>{STAGE_LABELS[stage.name] ?? stage.name}</td>
                <td>
                  <span
                    className={
                      stage.status === "FAILED"
                        ? "badge badge-error"
                        : stage.status === "WARNING"
                          ? "badge badge-warning"
                          : stage.status === "SKIPPED"
                            ? "badge"
                            : "badge badge-ok"
                    }
                  >
                    {stage.status}
                  </span>
                </td>
                <td>
                  {stage.duration === null || stage.duration === undefined
                    ? "-"
                    : `${stage.duration.toFixed(2)}s`}
                </td>
                <td>{stage.message}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {warnings.length > 0 ? (
        <>
          <h3>Warnings</h3>
          <ul>
            {warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
          </ul>
        </>
      ) : null}

      {errors.length > 0 ? (
        <>
          <h3>Errors</h3>
          <ul>
            {errors.map((message) => (
              <li key={message}>{message}</li>
            ))}
          </ul>
        </>
      ) : null}

      <p className="hint">
        A stage marked SKIPPED means it was not applicable - usually because no
        trained model is available. No prediction is produced in that case.
        {" "}
        {data.disclaimer ??
          "Research prototype. Not an operational emergency-warning system."}
      </p>
    </section>
  );
}
