/**
 * GNN event-evolution panel (SIH26078, Phase 3).
 *
 * Shows the model status, the architecture, the recorded losses and a prototype
 * next-state prediction for a selected track.
 *
 * This panel deliberately shows **no accuracy figure**. The model is trained on
 * synthetic data and has not been evaluated against observed weather, so any
 * percentage would be misleading. The prototype notice is rendered above the
 * controls so it is visible before any prediction is requested.
 */
import { useCallback, useEffect, useState } from "react";

import {
  describeApiError,
  fetchGNNModelInfo,
  fetchGNNStatus,
  fetchPrediction,
} from "../services/gnnApi";
import { fetchTracks } from "../services/tracksApi";
import type { EventTrack, GNNModelInfo, GNNModelStatus, GNNPrediction } from "../types/api";

type PanelState =
  | { status: "loading" }
  | {
      status: "ready";
      tracks: EventTrack[];
      modelStatus: GNNModelStatus;
      modelInfo: GNNModelInfo | null;
      selectedTrackId: string;
      prediction: GNNPrediction | null;
      predicting: boolean;
      predictionError: string | null;
    }
  | { status: "error"; message: string };

function formatLoss(value: number | null): string {
  return value === null ? "n/a" : value.toFixed(6);
}

function PredictionBlock({ prediction }: { prediction: GNNPrediction }): React.JSX.Element {
  return (
    <div className="gnn-prediction">
      <dl className="kv">
        <div>
          <dt>Current position</dt>
          <dd>
            {prediction.current_latitude.toFixed(4)} N, {prediction.current_longitude.toFixed(4)} E
          </dd>
        </div>
        <div>
          <dt>Current intensity</dt>
          <dd>{prediction.current_intensity.toFixed(4)}</dd>
        </div>
        <div>
          <dt>Predicted position</dt>
          <dd>
            {prediction.predicted_latitude.toFixed(4)} N,{" "}
            {prediction.predicted_longitude.toFixed(4)} E
          </dd>
        </div>
        <div>
          <dt>Predicted intensity</dt>
          <dd>{prediction.predicted_intensity.toFixed(4)}</dd>
        </div>
        <div>
          <dt>Movement delta</dt>
          <dd>
            {prediction.delta_latitude >= 0 ? "+" : ""}
            {prediction.delta_latitude.toFixed(5)} lat, {prediction.delta_longitude >= 0 ? "+" : ""}
            {prediction.delta_longitude.toFixed(5)} lon
          </dd>
        </div>
        <div>
          <dt>Intensity delta</dt>
          <dd>
            {prediction.delta_intensity >= 0 ? "+" : ""}
            {prediction.delta_intensity.toFixed(4)}
          </dd>
        </div>
        <div>
          <dt>Implied speed</dt>
          <dd>
            {prediction.predicted_speed_kmh === null
              ? "n/a"
              : `${prediction.predicted_speed_kmh.toFixed(3)} km/h`}
          </dd>
        </div>
        <div>
          <dt>Direction</dt>
          <dd>{prediction.predicted_direction ?? "n/a"}</dd>
        </div>
      </dl>
    </div>
  );
}

export default function GNNPanel(): React.JSX.Element {
  const [state, setState] = useState<PanelState>({ status: "loading" });

  const load = useCallback(async (): Promise<void> => {
    setState({ status: "loading" });
    try {
      // The model info is descriptive only, so a failure there must not hide the
      // status: fetch it separately and degrade to null.
      const [modelStatus, trackResponse] = await Promise.all([fetchGNNStatus(), fetchTracks()]);
      let modelInfo: GNNModelInfo | null = null;
      try {
        modelInfo = await fetchGNNModelInfo();
      } catch {
        modelInfo = null;
      }
      setState({
        status: "ready",
        tracks: trackResponse.tracks,
        modelStatus,
        modelInfo,
        selectedTrackId: trackResponse.tracks[0]?.track_id ?? "",
        prediction: null,
        predicting: false,
        predictionError: null,
      });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const runPrediction = useCallback(async (trackId: string): Promise<void> => {
    if (!trackId) {
      return;
    }
    setState((previous) =>
      previous.status === "ready"
        ? { ...previous, predicting: true, prediction: null, predictionError: null }
        : previous,
    );
    try {
      const prediction = await fetchPrediction(trackId);
      setState((previous) =>
        previous.status === "ready"
          ? { ...previous, predicting: false, prediction, predictionError: null }
          : previous,
      );
    } catch (error) {
      setState((previous) =>
        previous.status === "ready"
          ? { ...previous, predicting: false, predictionError: describeApiError(error) }
          : previous,
      );
    }
  }, []);

  const selectTrack = useCallback((trackId: string): void => {
    setState((previous) =>
      previous.status === "ready"
        ? { ...previous, selectedTrackId: trackId, prediction: null, predictionError: null }
        : previous,
    );
  }, []);

  return (
    <section className="panel" aria-labelledby="gnn-heading">
      <header className="panel-header">
        <h2 id="gnn-heading">GNN event evolution</h2>
        <div className="panel-actions">
          <button type="button" className="button" onClick={() => void load()}>
            Refresh
          </button>
        </div>
      </header>

      <p className="gnn-disclaimer" role="note">
        Research Prototype - Not an Operational Forecast. Trained on synthetic data; predictions are
        not validated against observed weather and must not be used for emergency decisions.
      </p>

      {state.status === "loading" && (
        <p className="status status-loading" role="status">
          Loading model status ...
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
                {state.modelStatus.available
                  ? (state.modelStatus.model_version ?? "available")
                  : "not trained yet"}
              </dd>
            </div>
            <div>
              <dt>Backend</dt>
              <dd>{state.modelStatus.backend}</dd>
            </div>
            {state.modelInfo ? (
              <div>
                <dt>Architecture</dt>
                <dd>
                  {state.modelInfo.message_layers} message-passing layers, hidden{" "}
                  {state.modelInfo.hidden_dim}
                </dd>
              </div>
            ) : null}
            <div>
              <dt>Train loss</dt>
              <dd>{formatLoss(state.modelStatus.train_loss)}</dd>
            </div>
            <div>
              <dt>Validation loss</dt>
              <dd>{formatLoss(state.modelStatus.validation_loss)}</dd>
            </div>
            {state.modelStatus.best_epoch !== null ? (
              <div>
                <dt>Best epoch</dt>
                <dd>{state.modelStatus.best_epoch}</dd>
              </div>
            ) : null}
          </dl>

          {!state.modelStatus.available && state.modelStatus.message ? (
            <p className="hint">{state.modelStatus.message}</p>
          ) : null}

          {state.tracks.length === 0 ? (
            <p className="hint">No tracks available: run the tracking step first.</p>
          ) : (
            <>
              <div className="gnn-controls">
                <label htmlFor="gnn-track-select">Track</label>
                <select
                  id="gnn-track-select"
                  value={state.selectedTrackId}
                  onChange={(event) => selectTrack(event.target.value)}
                >
                  {state.tracks.map((track) => (
                    <option key={track.track_id} value={track.track_id}>
                      {track.track_id}
                    </option>
                  ))}
                </select>
                <button
                  type="button"
                  className="button"
                  disabled={!state.modelStatus.available || state.predicting}
                  onClick={() => void runPrediction(state.selectedTrackId)}
                >
                  {state.predicting ? "Predicting ..." : "Predict next state"}
                </button>
              </div>

              {state.predictionError ? (
                <p className="status status-error" role="alert">
                  {state.predictionError}
                </p>
              ) : null}

              {state.prediction ? <PredictionBlock prediction={state.prediction} /> : null}
            </>
          )}

          <p className="hint">
            No accuracy figure is shown: the prototype has not been evaluated against observed
            weather, so any percentage would be meaningless.
          </p>
        </>
      )}
    </section>
  );
}

