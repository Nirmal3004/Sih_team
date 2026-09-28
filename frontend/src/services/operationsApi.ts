/**
 * API client for the Phase 10 operational endpoints.
 *
 * These endpoints report the state of the *system* - is it running, is it ready,
 * how many requests has it served. They say nothing about model quality, and
 * this client deliberately exposes no field that could be mistaken for an
 * accuracy figure.
 */
import { api, describeApiError } from "./api";
import type {
  OperationsIntegrity,
  OperationsMetrics,
  OperationsStatus,
} from "../types/api";

export { describeApiError };

/** Liveness. Answers only "is the process running". */
export async function fetchHealth(): Promise<Record<string, unknown>> {
  const { data } = await api.get<Record<string, unknown>>("/health");
  return data;
}

/** Readiness, with an explicit READY / DEGRADED / NOT_READY state. */
export async function fetchReadiness(): Promise<OperationsStatus> {
  const { data } = await api.get<OperationsStatus>("/ready");
  return data;
}

/** The full operational picture, mirroring /ready under the versioned prefix. */
export async function fetchOperationsStatus(): Promise<OperationsStatus> {
  const { data } = await api.get<OperationsStatus>(
    "/api/v1/operations/status",
  );
  return data;
}

/** System counters and duration histograms. Not model-quality metrics. */
export async function fetchOperationsMetrics(): Promise<OperationsMetrics> {
  const { data } = await api.get<OperationsMetrics>(
    "/api/v1/operations/metrics",
  );
  return data;
}

/** Model and data integrity states. */
export async function fetchIntegrity(): Promise<OperationsIntegrity> {
  const { data } = await api.get<OperationsIntegrity>(
    "/api/v1/operations/integrity",
  );
  return data;
}
