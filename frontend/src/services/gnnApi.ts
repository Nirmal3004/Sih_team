/**
 * API client for the DeepWeather AI GNN endpoints (Phase 3).
 *
 * The dashboard only reads what the backend produced: `scripts/train_gnn.py`
 * writes `models/gnn/best.json`, and these calls report the model status and its
 * prototype next-state predictions.
 *
 * **Research prototype.** Predictions come from a model trained on synthetic
 * data. They are not an operational forecast and must not be used for any
 * emergency decision. There is deliberately no accuracy percentage in this UI.
 */
import { api, describeApiError } from "./api";
import type { GNNModelInfo, GNNModelStatus, GNNPrediction } from "../types/api";

export { describeApiError };

/** Fetch whether a trained model is available, plus its provenance. */
export async function fetchGNNStatus(): Promise<GNNModelStatus> {
  const { data } = await api.get<GNNModelStatus>("/api/v1/gnn/status");
  return data;
}

/** Fetch the architecture and feature contract. */
export async function fetchGNNModelInfo(): Promise<GNNModelInfo> {
  const { data } = await api.get<GNNModelInfo>("/api/v1/gnn/model");
  return data;
}

/**
 * Request a prototype next-state prediction for one track.
 *
 * Throws on 404 (unknown track), 503 (model not trained) or 422 (invalid
 * request); use `describeApiError` to render the message.
 */
export async function fetchPrediction(trackId: string): Promise<GNNPrediction> {
  const { data } = await api.post<GNNPrediction>("/api/v1/gnn/predict", { track_id: trackId });
  return data;
}
