/**
 * Alert lifecycle timeline (SIH26078, Phase 5).
 *
 * Renders the stage sequence
 *
 *   detected -> tracked -> predicted -> downscaled -> alert created
 *            -> acknowledged -> resolved
 *
 * using the timestamps the backend actually recorded. A stage that did not
 * happen is shown as such with `n/a` rather than being given a made-up time -
 * the component fetches the timeline from `GET /api/v1/alerts/{id}` so the
 * values are the stored ones, not a client-side reconstruction.
 */
import { useCallback, useEffect, useState } from "react";

import { describeApiError, fetchAlert } from "../services/alertsApi";
import type { Alert, AlertTimelineStep } from "../types/api";

const STAGE_LABELS: Record<string, string> = {
  detected: "Detected",
  tracked: "Tracked",
  predicted: "Prediction generated",
  downscaled: "Downscaled (~5 km)",
  alert_created: "Alert created",
  acknowledged: "Acknowledged",
  resolved: "Resolved",
  expired: "Expired",
  no_input: "No input",
};

function fmt(value: string | null): string {
  return value ? new Date(value).toLocaleString() : "n/a";
}

export default function AlertTimeline({ alert }: { alert: Alert }): React.JSX.Element {
  const [steps, setSteps] = useState<AlertTimelineStep[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const load = useCallback(async (): Promise<void> => {
    setLoading(true);
    try {
      const detail = await fetchAlert(alert.alert_id);
      setSteps(detail.timeline);
      setError(null);
    } catch (caught) {
      setError(describeApiError(caught));
    } finally {
      setLoading(false);
    }
  }, [alert.alert_id]);

  useEffect(() => {
    void load();
  }, [load]);

  if (error) {
    return (
      <div className="alert-timeline">
        <h4>Timeline</h4>
        <p className="status status-error" role="alert">
          {error}
        </p>
      </div>
    );
  }

  return (
    <div className="alert-timeline">
      <h4>Timeline</h4>
      {loading && steps.length === 0 ? (
        <p className="status status-loading" role="status">
          Loading timeline ...
        </p>
      ) : null}
      <ol className="timeline">
        {steps.map((step) => (
          <li key={`${step.index}-${step.stage}`} className="timeline-step">
            <span className="timeline-stage">{STAGE_LABELS[step.stage] ?? step.stage}</span>
            <span className="timeline-time">{fmt(step.timestamp)}</span>
            {step.status ? <span className="timeline-status">{step.status}</span> : null}
            {step.severity ? <span className="timeline-severity">{step.severity}</span> : null}
            {step.score !== null ? (
              <span className="timeline-score">score {step.score.toFixed(2)}</span>
            ) : null}
            <p className="timeline-description">{step.description}</p>
          </li>
        ))}
      </ol>
      <p className="hint">
        Timestamps are the ones the pipeline recorded. A stage that did not happen is shown
        with `n/a` rather than an invented time.
      </p>
    </div>
  );
}
