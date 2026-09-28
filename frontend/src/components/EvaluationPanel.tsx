/**
 * Evaluation panel (SIH26078, Phase 9).
 *
 * This panel shows model measurements, so its display rules are the strictest in
 * the dashboard:
 *
 *  - a metric with no value renders as "Not available" with the reason. It is
 *    never rendered as `0.000`, because a zero reads as a measured result;
 *  - every metric carries its status, so a DERIVED interval width is never
 *    displayed with the same weight as a MEASURED error;
 *  - a model is never labelled "better" in prose. Only the backend's per-metric
 *    verdict is shown, and a loss is shown as a loss;
 *  - unavailable metrics get their own section, so what is missing is as
 *    visible as what was measured.
 *
 * There are no accuracy percentages in this panel, because no verified
 * ground-truth catalogue exists for this project and none is invented here.
 */
import { useCallback, useEffect, useState } from "react";

import {
  describeApiError,
  fetchDownscalingComparisonResult,
  fetchEvaluationStatus,
  fetchGnnEvaluation,
  fetchUncertainty,
  runEvaluation,
} from "../services/evaluationApi";
import type {
  DownscalingComparisonResult,
  EvaluationStatus,
  GnnEvaluation,
  MetricVerdict,
  UncertaintyEvaluation,
} from "../types/api";

type State =
  | { status: "loading" }
  | {
      status: "ready";
      evaluation: EvaluationStatus;
      gnn: GnnEvaluation | null;
      downscaling: DownscalingComparisonResult | null;
      uncertainty: UncertaintyEvaluation | null;
    }
  | { status: "error"; message: string };

/** Render a verdict without inflating it into an overall claim. */
function VerdictCell({ verdict }: { verdict: MetricVerdict }) {
  const label =
    verdict.verdict === "improves"
      ? "better"
      : verdict.verdict === "worse"
        ? "worse"
        : verdict.verdict === "tie"
          ? "tied"
          : "not computed";
  return (
    <td>
      <span
        className={
          verdict.verdict === "improves"
            ? "badge badge-ok"
            : verdict.verdict === "worse"
              ? "badge badge-warning"
              : "badge"
        }
      >
        {label}
      </span>
    </td>
  );
}

export default function EvaluationPanel() {
  const [state, setState] = useState<State>({ status: "loading" });
  const [busy, setBusy] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const evaluation = await fetchEvaluationStatus();
      // Each sub-resource 404s cleanly when no evaluation exists; that is an
      // absent measurement, not an error, so they are fetched independently.
      const [gnn, downscaling, uncertainty] = await Promise.all([
        evaluation.has_run
          ? fetchGnnEvaluation().catch(() => null)
          : Promise.resolve(null),
        evaluation.has_run
          ? fetchDownscalingComparisonResult().catch(() => null)
          : Promise.resolve(null),
        evaluation.has_run
          ? fetchUncertainty().catch(() => null)
          : Promise.resolve(null),
      ]);
      setState({ status: "ready", evaluation, gnn, downscaling, uncertainty });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const trigger = useCallback(async () => {
    setBusy(true);
    setActionError(null);
    try {
      await runEvaluation({ model: "fixture", sample_count: 60 });
      await refresh();
    } catch (error) {
      setActionError(describeApiError(error));
    } finally {
      setBusy(false);
    }
  }, [refresh]);

  if (state.status === "loading") {
    return (
      <section className="panel" aria-labelledby="evaluation-heading">
        <header className="panel-header">
          <h2 id="evaluation-heading">Model evaluation</h2>
        </header>
        <p className="hint">Loading evaluation status...</p>
      </section>
    );
  }

  if (state.status === "error") {
    return (
      <section className="panel" aria-labelledby="evaluation-heading">
        <header className="panel-header">
          <h2 id="evaluation-heading">Model evaluation</h2>
        </header>
        <p className="error-text">
          Could not load the evaluation status: {state.message}
        </p>
      </section>
    );
  }

  const { evaluation, gnn, downscaling, uncertainty } = state;

  return (
    <section className="panel" aria-labelledby="evaluation-heading">
      <header className="panel-header">
        <h2 id="evaluation-heading">Model evaluation</h2>
        <div className="panel-actions">
          <span className="badge badge-synthetic">Research prototype</span>
          {evaluation.latest_status ? (
            <span className="badge">{evaluation.latest_status}</span>
          ) : null}
        </div>
      </header>

      {!evaluation.has_run ? (
        <p className="hint">
          {evaluation.message ||
            "No evaluation has been run. No metric is reported, and none is assumed to be zero."}
        </p>
      ) : null}

      <div className="panel-actions">
        <button type="button" onClick={trigger} disabled={busy}>
          {busy ? "Evaluating..." : "Run evaluation"}
        </button>
        <button type="button" onClick={refresh} disabled={busy}>
          Refresh
        </button>
      </div>

      {actionError ? <p className="error-text">{actionError}</p> : null}

      {gnn ? (
        <>
          <h3>GNN versus baselines</h3>
          <p className="hint">
            {gnn.sample_count} sample(s) scored on the test split.{" "}
            {gnn.interpretation}
          </p>
          <table className="table">
            <thead>
              <tr>
                <th>Metric</th>
                <th>Persistence</th>
                <th>Constant velocity</th>
                <th>GNN</th>
                <th>vs persistence</th>
              </tr>
            </thead>
            <tbody>
              {Object.keys(
                gnn.comparisons.gnn_vs_persistence ?? {},
              ).map((metric) => {
                const verdict =
                  gnn.comparisons.gnn_vs_persistence?.[metric];
                const value = (key: string) => {
                  const entry = gnn.methods[key] as
                    | Record<string, number | null>
                    | undefined;
                  return entry?.[metric] ?? null;
                };
                return (
                  <tr key={metric}>
                    <td>{metric}</td>
                    <td>{format(value("persistence"))}</td>
                    <td>{format(value("constant_velocity"))}</td>
                    <td>{format(value("gnn"))}</td>
                    {verdict ? <VerdictCell verdict={verdict} /> : <td>-</td>}
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="hint">
            Verdicts are stated per metric. Beating one metric and losing another
            is not an overall win, and no overall win is claimed.
          </p>
        </>
      ) : null}

      {gnn && !gnn.model_available ? (
        <p className="hint">
          No trained GNN checkpoint was available, so the model column is
          absent. Its result is not simulated and no model performance is
          inferred from the baselines.
        </p>
      ) : null}

      {gnn?.baselines ? (
        <>
          <h3>What each baseline assumes</h3>
          <ul>
            {Object.entries(gnn.baselines).map(([key, entry]) => (
              <li key={key}>
                <strong>{entry.name}</strong> � <code>{entry.formula}</code>
                <div className="hint">{entry.why}</div>
              </li>
            ))}
          </ul>
        </>
      ) : null}

      {downscaling ? (
        <>
          <h3>Downscaling comparison</h3>
          {Object.keys(downscaling.methods).length === 0 ? (
            <p className="hint">No downscaling method could be scored.</p>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Method</th>
                  <th>MAE</th>
                  <th>RMSE</th>
                  <th>Bias</th>
                  <th>Correlation</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(downscaling.methods).map(([name, entry]) => {
                  const stats = entry as unknown as Record<string, number | null>;
                  return (
                    <tr key={name}>
                      <td>{name}</td>
                      <td>{format(stats.mae)}</td>
                      <td>{format(stats.rmse)}</td>
                      <td>{format(stats.bias)}</td>
                      <td>{format(stats.correlation)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          {Object.keys(downscaling.unavailable).length > 0 ? (
            <p className="hint">
              Unavailable:{" "}
              {Object.entries(downscaling.unavailable)
                .map(([name, entry]) => `${name} (${entry.reason})`)
                .join("; ")}
            </p>
          ) : null}
        </>
      ) : null}

      {uncertainty ? (
        <>
          <h3>Uncertainty</h3>
          <dl className="kv">
            <dt>Method</dt>
            <dd>{uncertainty.method || "Not computed"}</dd>
          </dl>
          <table className="table">
            <thead>
              <tr>
                <th>Nominal</th>
                <th>Latitude coverage</th>
                <th>Longitude coverage</th>
                <th>Intensity coverage</th>
                <th>n</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(uncertainty.calibration).map(([level, entry]) => {
                const record = entry as Record<string, unknown>;
                const coverage = (record.empirical_coverage ?? {}) as Record<
                  string,
                  number
                >;
                return (
                  <tr key={level}>
                    <td>{level}</td>
                    <td>{format(coverage.latitude)}</td>
                    <td>{format(coverage.longitude)}</td>
                    <td>{format(coverage.intensity)}</td>
                    <td>{String(record.sample_count ?? "-")}</td>
                    <td>{String(record.status ?? "NOT_AVAILABLE")}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          <p className="hint">
            {uncertainty.caveat ||
              "Empirical validation-residual quantiles. Not a probability."}{" "}
            Coverage is measured on the test split; where it is not reported, the
            interval is uncalibrated and its nominal level is a construction
            parameter, not a verified probability.
          </p>
        </>
      ) : null}

      <p className="hint">
        {evaluation.disclaimer ||
          "Research prototype. Not an operational emergency-warning system."}
      </p>
    </section>
  );
}

/** Render a number, or an explicit absence - never a zero for a missing value. */
function format(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "Not available";
  }
  return value.toFixed(4);
}
