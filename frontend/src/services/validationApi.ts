/**
 * API client for the Phase 7 validation endpoints.
 *
 * The validation reports are the only place this project states what it can and
 * cannot demonstrate, so the client passes their contents through untouched.
 * In particular it never defaults a missing metric to zero, and it never
 * derives a percentage the backend did not compute: a report that says
 * "Ground truth unavailable - metric not computed" must reach the UI as exactly
 * that, not as `0.00`.
 */
import { api, describeApiError } from "./api";
import type {
  Phase7Summary,
  ValidationMetricsResponse,
  ValidationProvenance,
  ValidationStatus,
} from "../types/api";

export { describeApiError };

/** Whether a Phase 7 run exists and which stage reports are present. */
export async function fetchValidationStatus(): Promise<ValidationStatus> {
  const { data } = await api.get<ValidationStatus>("/api/v1/validation/status");
  return data;
}

/** The unified Phase 7 report. Rejects with a 404 when no run exists. */
export async function fetchValidationSummary(): Promise<Phase7Summary> {
  const { data } = await api.get<Phase7Summary>("/api/v1/validation/summary");
  return data;
}

/** Per-stage metrics, each marked available or explicitly not produced. */
export async function fetchValidationMetrics(): Promise<ValidationMetricsResponse> {
  const { data } = await api.get<ValidationMetricsResponse>(
    "/api/v1/validation/metrics",
  );
  return data;
}

/** The provenance record of the run that produced the reports. */
export async function fetchValidationProvenance(): Promise<ValidationProvenance> {
  const { data } = await api.get<ValidationProvenance>(
    "/api/v1/validation/provenance",
  );
  return data;
}
