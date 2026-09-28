/**
 * API client for the Phase 6 real / historical data endpoints.
 *
 * Everything the dashboard shows about data provenance comes from these calls;
 * nothing is inferred in the browser. In particular the `origin`,
 * `is_synthetic` and `is_live` flags are rendered exactly as the backend
 * reports them, because a client that guessed "this looks real" would be the
 * easiest way for the project to present synthetic data as observed.
 */
import { api, describeApiError } from "./api";
import type {
  DatasetManifest,
  DataSourceListResponse,
  IngestedDatasetListResponse,
  IngestRequest,
  IngestResponse,
  ProcessResponse,
  QualityReport,
} from "../types/api";

export { describeApiError };

/** Fetch the registered data sources with their availability. */
export async function fetchDataSources(): Promise<DataSourceListResponse> {
  const { data } = await api.get<DataSourceListResponse>("/api/v1/data/sources");
  return data;
}

/** Filters accepted by `GET /api/v1/data/datasets`. */
export interface DatasetFilters {
  origin?: string;
  source?: string;
  start_time?: string;
  end_time?: string;
}

/** Fetch the ingested datasets, optionally filtered. */
export async function fetchDatasets(
  filters: DatasetFilters = {},
): Promise<IngestedDatasetListResponse> {
  const params: Record<string, string> = {};
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== null && value !== "") {
      params[key] = String(value);
    }
  }
  const { data } = await api.get<IngestedDatasetListResponse>("/api/v1/data/datasets", {
    params,
  });
  return data;
}

/** Fetch one dataset's manifest. */
export async function fetchDataset(datasetId: string): Promise<DatasetManifest> {
  const { data } = await api.get<DatasetManifest>(
    `/api/v1/data/datasets/${encodeURIComponent(datasetId)}`,
  );
  return data;
}

/** Re-run the quality check for one dataset. */
export async function fetchDatasetQuality(datasetId: string): Promise<QualityReport> {
  const { data } = await api.get<QualityReport>(
    `/api/v1/data/datasets/${encodeURIComponent(datasetId)}/quality`,
  );
  return data;
}

/** Quality-check a file without writing anything. */
export async function validateDataFile(
  path: string,
  strictRanges = false,
): Promise<QualityReport> {
  const { data } = await api.post<QualityReport>("/api/v1/data/validate", {
    path,
    strict_ranges: strictRanges,
  });
  return data;
}

/** Ingest a file: validate, normalise, manifest and register. */
export async function ingestDataFile(request: IngestRequest): Promise<IngestResponse> {
  const { data } = await api.post<IngestResponse>("/api/v1/data/ingest", request);
  return data;
}

/** Run the Phase 0 preprocessing pipeline over an ingested dataset. */
export async function processDataset(
  datasetId: string,
  maxGap = 3,
): Promise<ProcessResponse> {
  const { data } = await api.post<ProcessResponse>(
    `/api/v1/data/datasets/${encodeURIComponent(datasetId)}/process`,
    { max_gap: maxGap },
  );
  return data;
}
