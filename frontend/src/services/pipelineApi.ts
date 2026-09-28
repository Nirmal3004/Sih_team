/**
 * API client for the Phase 8 near-real-time pipeline endpoints.
 *
 * Two rules are inherited from the Phase 7 client and are deliberate:
 *
 *  - an unavailable value is never rendered as zero. A `null` freshness age
 *    stays `null`, so the UI can say "unknown" instead of showing "0 min old",
 *    which would be a claim about the data that was never made;
 *  - the prototype disclaimer returned by the backend is passed through
 *    untouched rather than replaced with something shorter.
 */
import { api, describeApiError } from "./api";
import type {
  PipelineRun,
  PipelineRunList,
  PipelineRunRequest,
  PipelineStatus,
} from "../types/api";

export { describeApiError };

/** Current pipeline state, freshness and the latest stage summary. */
export async function fetchPipelineStatus(): Promise<PipelineStatus> {
  const { data } = await api.get<PipelineStatus>("/api/v1/pipeline/status");
  return data;
}

/** Persisted run history, newest first. */
export async function fetchPipelineRuns(
  limit = 20,
): Promise<PipelineRunList> {
  const { data } = await api.get<PipelineRunList>(
    `/api/v1/pipeline/runs?limit=${limit}`,
  );
  return data;
}

/** One run in full. Rejects with a 404 for an unknown id. */
export async function fetchPipelineRun(runId: string): Promise<PipelineRun> {
  const { data } = await api.get<PipelineRun>(
    `/api/v1/pipeline/runs/${encodeURIComponent(runId)}`,
  );
  return data;
}

/**
 * Execute the pipeline now.
 *
 * `dryRun` maps to the query parameter, so a planning request cannot be
 * confused with a real one by the backend.
 */
export async function runPipeline(
  request: PipelineRunRequest = {},
  dryRun = false,
): Promise<PipelineRun> {
  const { data } = await api.post<PipelineRun>(
    `/api/v1/pipeline/run?dry_run=${dryRun}`,
    { provider: "fixture", hours: 6, ...request },
  );
  return data;
}
