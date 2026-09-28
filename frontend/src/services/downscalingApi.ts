/**
 * API client for the DeepWeather AI downscaling endpoints (Phase 4).
 *
 * The backend trains a small CNN residual model against a **synthetic**
 * high-resolution reference. This client therefore only ever surfaces measured
 * agreement metrics - there is no accuracy percentage to show, and none is
 * invented here.
 */
import { api, describeApiError } from "./api";
import type {
  DownscalingEvaluation,
  DownscalingModelStatus,
  DownscalingResponse,
} from "../types/api";

export { describeApiError };

/** Fetch whether a trained downscaling model is available, plus provenance. */
export async function fetchDownscalingStatus(): Promise<DownscalingModelStatus> {
  const { data } = await api.get<DownscalingModelStatus>("/api/v1/downscaling/status");
  return data;
}

/** Downscale one frame with `nearest`, `cubic` or `ai`. */
export async function runDownscaling(
  method: string,
  timeIndex = 0,
): Promise<DownscalingResponse> {
  const { data } = await api.post<DownscalingResponse>("/api/v1/downscaling/run", {
    method,
    time_index: timeIndex,
  });
  return data;
}

/** Score every method on the same test samples. */
export async function evaluateDownscaling(
  methods: string[] = ["nearest", "cubic", "ai"],
): Promise<DownscalingEvaluation> {
  const { data } = await api.post<DownscalingEvaluation>("/api/v1/downscaling/evaluate", {
    methods,
  });
  return data;
}
