/**
 * API client for the DeepWeather AI alerting endpoints (Phase 5).
 *
 * The dashboard only reads what the backend produced: `scripts/generate_alerts.py`
 * scores the Phase 1-4 outputs and writes `alerts.json`, and these calls serve
 * that file. Nothing is scored or decided in the browser.
 *
 * Two rules shape this client:
 *  - the severities are prototype categories, never official warnings, so the
 *    `notice` returned by the backend is passed through untouched;
 *  - acknowledge/resolve are the *only* writes, and they are explicit operator
 *    actions - never fired automatically.
 */
import { api, describeApiError } from "./api";
import type {
  AlertActionResponse,
  AlertDetailResponse,
  AlertFilters,
  AlertGenerateRequest,
  AlertGenerateResponse,
  AlertListResponse,
  AlertSummaryResponse,
  NotificationListResponse,
  SimulationRequest,
  SimulationResponse,
} from "../types/api";

export { describeApiError };

/** Fetch the stored alerts, optionally filtered. */
export async function fetchAlerts(filters: AlertFilters = {}): Promise<AlertListResponse> {
  const params: Record<string, string> = {};
  for (const [key, value] of Object.entries(filters)) {
    if (value !== undefined && value !== null && value !== "") {
      params[key] = String(value);
    }
  }
  const { data } = await api.get<AlertListResponse>("/api/v1/alerts", { params });
  return data;
}

/** Fetch one alert with its lifecycle timeline. */
export async function fetchAlert(alertId: string): Promise<AlertDetailResponse> {
  const { data } = await api.get<AlertDetailResponse>(
    `/api/v1/alerts/${encodeURIComponent(alertId)}`,
  );
  return data;
}

/** Fetch the dashboard headline counters. */
export async function fetchAlertSummary(): Promise<AlertSummaryResponse> {
  const { data } = await api.get<AlertSummaryResponse>("/api/v1/alerts/summary");
  return data;
}

/**
 * Run the alert generation pipeline.
 *
 * `dry_run: true` asks the backend to compute everything and write nothing,
 * which is the safe way to preview a run against a real store.
 */
export async function generateAlerts(
  request: AlertGenerateRequest = {},
): Promise<AlertGenerateResponse> {
  const { data } = await api.post<AlertGenerateResponse>("/api/v1/alerts/generate", request);
  return data;
}

/** Record that an operator has acknowledged an alert. */
export async function acknowledgeAlert(
  alertId: string,
  note?: string,
): Promise<AlertActionResponse> {
  const { data } = await api.post<AlertActionResponse>(
    `/api/v1/alerts/${encodeURIComponent(alertId)}/acknowledge`,
    { note: note ?? null },
  );
  return data;
}

/** Close an alert. `RESOLVED` is terminal and cannot be undone. */
export async function resolveAlert(
  alertId: string,
  reason?: string,
): Promise<AlertActionResponse> {
  const { data } = await api.post<AlertActionResponse>(
    `/api/v1/alerts/${encodeURIComponent(alertId)}/resolve`,
    { reason: reason ?? null },
  );
  return data;
}

/** Walk the deterministic lifecycle simulation. Always synthetic data. */
export async function simulateAlerts(
  request: SimulationRequest = {},
): Promise<SimulationResponse> {
  const { data } = await api.post<SimulationResponse>("/api/v1/alerts/simulate", request);
  return data;
}

/** Fetch the local notification log (nothing is delivered externally). */
export async function fetchNotifications(limit?: number): Promise<NotificationListResponse> {
  const { data } = await api.get<NotificationListResponse>("/api/v1/notifications", {
    params: limit ? { limit } : {},
  });
  return data;
}
