/**
 * API client for the DeepWeather AI FastAPI backend.
 *
 * The base URL is configured through `VITE_API_BASE_URL` (see
 * `frontend/.env.example`); it defaults to the local dev backend on port 8000.
 * No secrets are stored in the frontend - the backend needs no API keys.
 */
import axios from "axios";

import type {
  DatasetListResponse,
  DatasetSummaryResponse,
  HealthResponse,
  SystemStatusResponse,
} from "../types/api";

export const API_BASE_URL: string = (
  import.meta.env.VITE_API_BASE_URL ?? "http://localhost:8000"
).replace(/\/+$/, "");

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 10_000,
  headers: { Accept: "application/json" },
});

/** Fetch the backend health payload (`GET /health`). */
export async function fetchHealth(): Promise<HealthResponse> {
  const { data } = await api.get<HealthResponse>("/health");
  return data;
}

/** List the NetCDF datasets known to the backend (`GET /api/v1/datasets`). */
export async function fetchDatasets(): Promise<DatasetListResponse> {
  const { data } = await api.get<DatasetListResponse>("/api/v1/datasets");
  return data;
}

/**
 * Fetch the metadata summary of one dataset.
 *
 * @param datasetPath Path relative to the backend data directory, e.g.
 *   `raw/sample_history.nc`.
 */
export async function fetchDatasetSummary(datasetPath: string): Promise<DatasetSummaryResponse> {
  const { data } = await api.get<DatasetSummaryResponse>(
    `/api/v1/datasets/${datasetPath}/summary`,
  );
  return data;
}

/** Turn any thrown value into a message that can be shown in the UI. */
export function describeApiError(error: unknown): string {
  if (axios.isAxiosError(error)) {
    if (error.code === "ECONNABORTED") {
      return `Request to ${error.config?.url ?? "the backend"} timed out.`;
    }
    if (error.response) {
      const detail = (error.response.data as { detail?: string } | undefined)?.detail;
      return detail
        ? `Backend error ${error.response.status}: ${detail}`
        : `Backend error ${error.response.status} from ${error.config?.url ?? "the API"}.`;
    }
    return `Cannot reach the backend at ${API_BASE_URL}. Is uvicorn running?`;
  }
  return error instanceof Error ? error.message : "Unexpected error.";
}

/** Fetch the overall system status (`GET /api/v1/system/status`). */
export async function fetchSystemStatus(): Promise<SystemStatusResponse> {
  const { data } = await api.get<SystemStatusResponse>("/api/v1/system/status");
  return data;
}
