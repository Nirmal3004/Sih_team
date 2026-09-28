/**
 * API client for the Phase 9 evaluation endpoints.
 *
 * The display rules this project has held to since Phase 7 apply here too:
 *
 *  - an absent metric is never rendered as zero. `value: null` stays null, so
 *    the UI can render "Not available" rather than "0.000";
 *  - `status` is passed through untouched, because a MEASURED value and a
 *    DERIVED one may not be read the same way;
 *  - the prototype disclaimer travels with the payload and is never replaced
 *    with a shorter phrase.
 */
import { api, describeApiError } from "./api";
import type {
  DownscalingComparisonResult,
  EvaluationRunRequest,
  EvaluationStatus,
  EvaluationSummary,
  EventExplanation,
  GnnEvaluation,
  UncertaintyEvaluation,
} from "../types/api";

export { describeApiError };

/** Whether an evaluation exists and what it covered. */
export async function fetchEvaluationStatus(): Promise<EvaluationStatus> {
  const { data } = await api.get<EvaluationStatus>("/api/v1/evaluation/status");
  return data;
}

/** The unified report. Rejects with a 404 when no evaluation has been run. */
export async function fetchEvaluationSummary(): Promise<EvaluationSummary> {
  const { data } = await api.get<EvaluationSummary>("/api/v1/evaluation/summary");
  return data;
}

/** GNN results beside persistence and constant velocity. */
export async function fetchGnnEvaluation(): Promise<GnnEvaluation> {
  const { data } = await api.get<GnnEvaluation>("/api/v1/evaluation/gnn");
  return data;
}

/** Downscaling results beside the interpolation baselines. */
export async function fetchDownscalingComparisonResult(): Promise<DownscalingComparisonResult> {
  const { data } = await api.get<DownscalingComparisonResult>(
    "/api/v1/evaluation/downscaling",
  );
  return data;
}

/** Intervals, the method used, and any measured coverage. */
export async function fetchUncertainty(): Promise<UncertaintyEvaluation> {
  const { data } = await api.get<UncertaintyEvaluation>(
    "/api/v1/evaluation/uncertainty",
  );
  return data;
}

/** Supporting evidence for one event. Rejects with a 404 for an unknown id. */
export async function fetchEventExplanation(
  eventId: string,
): Promise<EventExplanation> {
  const { data } = await api.get<EventExplanation>(
    `/api/v1/evaluation/explain/${encodeURIComponent(eventId)}`,
  );
  return data;
}

/** Execute an evaluation. `dryRun` plans without persisting. */
export async function runEvaluation(
  request: EvaluationRunRequest = {},
): Promise<EvaluationSummary> {
  const { data } = await api.post<EvaluationSummary>(
    "/api/v1/evaluation/run",
    { model: "fixture", ...request },
  );
  return data;
}
