/**
 * API client for the DeepWeather AI event-tracking endpoints (Phase 2).
 *
 * The dashboard only reads what the backend produced: `scripts/track_events.py`
 * links the Phase 1 detections into `tracks.json`, and these calls serve that
 * file. Nothing is computed in the browser.
 *
 * Trajectory rendering on the map and prediction cones remain future work; this
 * client exposes the trajectory payload for the map layer only.
 */
import { api, describeApiError } from "./api";
import type { TrackFilters, TrackListResponse } from "../types/api";

export { describeApiError };

/** Fetch the tracked events, optionally filtered. */
export async function fetchTracks(filters: TrackFilters = {}): Promise<TrackListResponse> {
  const params: Record<string, string> = {};
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== null && value !== "") {
      params[key] = String(value);
    }
  }
  const { data } = await api.get<TrackListResponse>("/api/v1/tracks", { params });
  return data;
}

/** Fetch a single track by its deterministic id. */
export async function fetchTrack(trackId: string): Promise<TrackListResponse["tracks"][number]> {
  const { data } = await api.get<TrackListResponse["tracks"][number]>(
    `/api/v1/tracks/${encodeURIComponent(trackId)}`,
  );
  return data;
}

