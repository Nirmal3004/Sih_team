/**
 * AI downscaling panel (SIH26078, Phase 4).
 *
 * Shows the model status, the detected input/output resolutions, a method
 * selector, and the measured baseline-vs-AI comparison.
 *
 * Two things this panel deliberately does **not** do:
 *  - it never shows an "accuracy" percentage - the prototype has no skill score
 *    and any such number would be fabricated;
 *  - it never implies that the finer grid is more accurate. The comparison table
 *    reports what was measured, including when the AI loses to a baseline.
 */
import { useCallback, useEffect, useState } from "react";

import {
  describeApiError,
  evaluateDownscaling,
  fetchDownscalingStatus,
  runDownscaling,
} from "../services/downscalingApi";
import type {
  DownscalingEvaluation,
  DownscalingModelStatus,
  DownscalingResponse,
} from "../types/api";

const METHODS = ["nearest", "cubic", "ai"] as const;
const METHOD_LABELS: Record<string, string> = {
  nearest: "Nearest (baseline)",
  cubic: "Cubic (baseline)",
  ai: "AI (learned residual)",
};

type PanelState =
  | { status: "loading" }
  | {
      status: "ready";
      model: DownscalingModelStatus;
      method: string;
      run: DownscalingResponse | null;
      evaluation: DownscalingEvaluation | null;
      busy: boolean;
      error: string | null;
    }
  | { status: "error"; message: string };

function fmt(value: number | null | undefined, digits = 4): string {
  if (value === null || value === undefined) {
    return "n/a";
  }
  return value.toFixed(digits);
}

function ResolutionRow({
  label,
  degrees,
  km,
  shape,
}: {
  label: string;
  degrees: number | null;
  km?: number | null;
  shape: number[] | null;
}): React.JSX.Element {
  return (
    <tr>
      <th scope="row">{label}</th>
      <td>{fmt(degrees)} deg</td>
      <td>{km === undefined || km === null ? "" : `~${km.toFixed(1)} km`}</td>
      <td>{shape ? `${shape[0]} x ${shape[1]}` : "n/a"}</td>
    </tr>
  );
}

function MetricsTable({ evaluation }: { evaluation: DownscalingEvaluation }): React.JSX.Element {
  return (
    <table className="downscale-table">
      <caption>
        Agreement with the synthetic high-resolution reference, all methods on the same{" "}
        {evaluation.test_sample_count} test frame(s). Not real-world skill.
      </caption>
      <thead>
        <tr>
          <th scope="col">Method</th>
          <th scope="col">MAE</th>
          <th scope="col">RMSE</th>
          <th scope="col">Bias</th>
          <th scope="col">Corr.</th>
          <th scope="col">POD</th>
          <th scope="col">FAR</th>
        </tr>
      </thead>
      <tbody>
        {evaluation.methods.map((row) => (
          <tr key={row.method}>
            <th scope="row">{METHOD_LABELS[row.method] ?? row.method}</th>
            <td>{fmt(row.mae, 5)}</td>
            <td>{fmt(row.rmse, 5)}</td>
            <td>{fmt(row.bias, 5)}</td>
            <td>{fmt(row.correlation, 4)}</td>
            <td>{fmt(row.pod, 3)}</td>
            <td>{fmt(row.far, 3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function EventTable({ evaluation }: { evaluation: DownscalingEvaluation }): React.JSX.Element {
  return (
    <table className="downscale-table">
      <caption>
        Extreme-event preservation (peak, centre and wet area vs the synthetic reference).
      </caption>
      <thead>
        <tr>
          <th scope="col">Method</th>
          <th scope="col">Peak</th>
          <th scope="col">Ref. peak</th>
          <th scope="col">Peak ratio</th>
          <th scope="col">Centre shift</th>
          <th scope="col">Area ratio</th>
        </tr>
      </thead>
      <tbody>
        {evaluation.extreme_event.map((row) => (
          <tr key={row.method}>
            <th scope="row">{METHOD_LABELS[row.method] ?? row.method}</th>
            <td>{fmt(row.peak_value, 2)}</td>
            <td>{fmt(row.reference_peak, 2)}</td>
            <td>{fmt(row.peak_ratio, 3)}</td>
            <td>{fmt(row.centroid_distance_km, 2)} km</td>
            <td>{fmt(row.area_ratio, 3)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export default function DownscalingPanel(): React.JSX.Element {
  const [state, setState] = useState<PanelState>({ status: "loading" });

  const load = useCallback(async (): Promise<void> => {
    setState({ status: "loading" });
    try {
      const model = await fetchDownscalingStatus();
      setState({
        status: "ready",
        model,
        method: "nearest",
        run: null,
        evaluation: null,
        busy: false,
        error: null,
      });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const runOne = useCallback(async (method: string): Promise<void> => {
    setState((previous) =>
      previous.status === "ready" ? { ...previous, method, busy: true, error: null } : previous,
    );
    try {
      const result = await runDownscaling(method, 0);
      setState((previous) =>
        previous.status === "ready" ? { ...previous, run: result, busy: false } : previous,
      );
    } catch (error) {
      setState((previous) =>
        previous.status === "ready"
          ? { ...previous, busy: false, error: describeApiError(error) }
          : previous,
      );
    }
  }, []);

  const compare = useCallback(async (): Promise<void> => {
    setState((previous) =>
      previous.status === "ready" ? { ...previous, busy: true, error: null } : previous,
    );
    try {
      const evaluation = await evaluateDownscaling([...METHODS]);
      setState((previous) =>
        previous.status === "ready" ? { ...previous, evaluation, busy: false } : previous,
      );
    } catch (error) {
      setState((previous) =>
        previous.status === "ready"
          ? { ...previous, busy: false, error: describeApiError(error) }
          : previous,
      );
    }
  }, []);

  return (
    <section className="panel" aria-labelledby="downscaling-heading">
      <header className="panel-header">
        <h2 id="downscaling-heading">AI downscaling</h2>
        <div className="panel-actions">
          <button type="button" className="button" onClick={() => void load()}>
            Refresh
          </button>
        </div>
      </header>

      <p className="gnn-disclaimer" role="note">
        Research Prototype &mdash; Synthetic Data &mdash; Not an Operational Forecast. The
        high-resolution target is generated, not observed. Metrics measure agreement with that
        target; higher resolution does not imply higher accuracy.
      </p>

      {state.status === "loading" && (
        <p className="status status-loading" role="status">
          Loading downscaling status ...
        </p>
      )}

      {state.status === "error" && (
        <p className="status status-error" role="alert">
          {state.message}
        </p>
      )}

      {state.status === "ready" && (
        <>
          <dl className="kv">
            <div>
              <dt>Model</dt>
              <dd>
                {state.model.available ? (state.model.model_version ?? "available") : "not trained"}
              </dd>
            </div>
            <div>
              <dt>Variable</dt>
              <dd>{state.model.variable ?? "precipitation"}</dd>
            </div>
            <div>
              <dt>Validation loss</dt>
              <dd>{fmt(state.model.best_validation_loss, 6)}</dd>
            </div>
          </dl>

          <table className="downscale-table">
            <caption>Grid resolution, detected from the data (not hard-coded).</caption>
            <thead>
              <tr>
                <th scope="col">Grid</th>
                <th scope="col">Resolution</th>
                <th scope="col">Approx.</th>
                <th scope="col">Size (lat x lon)</th>
              </tr>
            </thead>
            <tbody>
              <ResolutionRow
                label="Input (coarse)"
                degrees={state.model.input_resolution_deg}
                shape={state.model.input_shape}
              />
              <ResolutionRow
                label="Output (fine)"
                degrees={state.model.output_resolution_deg}
                shape={state.model.output_shape}
              />
            </tbody>
          </table>

          {!state.model.available && state.model.message ? (
            <p className="hint">{state.model.message}</p>
          ) : null}

          <div className="gnn-controls">
            <label htmlFor="downscale-method">Method</label>
            <select
              id="downscale-method"
              value={state.method}
              onChange={(event) => void runOne(event.target.value)}
              disabled={state.busy}
            >
              {METHODS.map((method) => (
                <option key={method} value={method}>
                  {METHOD_LABELS[method]}
                </option>
              ))}
            </select>
            <button
              type="button"
              className="button"
              disabled={state.busy || (state.method === "ai" && !state.model.available)}
              onClick={() => void runOne(state.method)}
            >
              {state.busy ? "Working ..." : "Run"}
            </button>
            <button type="button" className="button" disabled={state.busy} onClick={() => void compare()}>
              Compare all methods
            </button>
          </div>

          {state.error ? (
            <p className="status status-error" role="alert">
              {state.error}
            </p>
          ) : null}

          {state.run ? (
            <dl className="kv">
              <div>
                <dt>Result grid</dt>
                <dd>
                  {state.run.output_shape[0]} x {state.run.output_shape[1]} at{" "}
                  {fmt(state.run.output_resolution_deg)} deg
                </dd>
              </div>
              <div>
                <dt>Max</dt>
                <dd>{fmt(state.run.statistics.max, 3)}</dd>
              </div>
              <div>
                <dt>Mean</dt>
                <dd>{fmt(state.run.statistics.mean, 3)}</dd>
              </div>
            </dl>
          ) : null}

          {state.evaluation ? (
            <>
              <MetricsTable evaluation={state.evaluation} />
              <EventTable evaluation={state.evaluation} />
              <p className="hint">
                Lowest MAE as measured: {state.evaluation.ranking.join(" &lt; ") || "n/a"}. This is
                an ordering of the numbers above, not a claim about real-world accuracy.
              </p>
            </>
          ) : null}

          <p className="hint">
            No accuracy figure is shown: the prototype has not been evaluated against observed
            weather, so any percentage would be meaningless.
          </p>
        </>
      )}
    </section>
  );
}