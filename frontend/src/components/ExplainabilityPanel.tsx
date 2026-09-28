/**
 * Explainability panel (SIH26078, Phase 9).
 *
 * Shows the evidence behind one detected event. Its central rule is the same
 * one the backend enforces: **evidence is not cause**.
 *
 *  - the panel never uses causal wording, and the backend disclaimer is
 *    rendered verbatim at the top so the caveat cannot be scrolled past;
 *  - unavailable evidence is listed as explicitly as available evidence, in
 *    its own section. A missing number must be visible, not quietly absent;
 *  - `context` items are labelled as recorded facts rather than model inputs,
 *    because no model in this project consumed them.
 */
import { useCallback, useEffect, useState } from "react";

import { describeApiError, fetchEventExplanation } from "../services/evaluationApi";
import type { EventExplanation, ExplanationItem } from "../types/api";

type State =
  | { status: "loading" }
  | { status: "ready"; explanation: EventExplanation }
  | { status: "error"; message: string };

/** Label for the provenance of an evidence item. */
function sourceLabel(item: ExplanationItem): string {
  switch (item.source) {
    case "phase1.anomaly":
      return "Phase 1 detection";
    case "phase2.tracking":
      return "Phase 2 tracking";
    case "phase3.gnn":
      return "Phase 3 model";
    case "phase4.downscaling":
      return "Phase 4 downscaling";
    case "phase5.alerting":
      return "Phase 5 alerting";
    case "phase8.realtime":
      return "Phase 8 data freshness";
    default:
      return item.source || "unknown";
  }
}

export default function ExplainabilityPanel({ eventId }: { eventId?: string }) {
  const [state, setState] = useState<State>({ status: "loading" });
  const [input, setInput] = useState(eventId ?? "");

  const load = useCallback(async (id: string) => {
    if (!id) {
      setState({
        status: "error",
        message: "Enter a detected event id to see its evidence.",
      });
      return;
    }
    try {
      setState({ status: "loading" });
      setState({ status: "ready", explanation: await fetchEventExplanation(id) });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, []);

  useEffect(() => {
    if (eventId) {
      void load(eventId);
    }
  }, [eventId, load]);

  return (
    <section className="panel" aria-labelledby="explainability-heading">
      <header className="panel-header">
        <h2 id="explainability-heading">Explainability</h2>
        <div className="panel-actions">
          <span className="badge badge-synthetic">Evidence, not causation</span>
        </div>
      </header>

      <div className="panel-actions">
        <label htmlFor="event-id">Event id</label>
        <input
          id="event-id"
          type="text"
          value={input}
          placeholder="e.g. EXTREME_RAINFALL_20240630T1200Z_0"
          onChange={(event) => setInput(event.target.value)}
        />
        <button type="button" onClick={() => void load(input)}>
          Show evidence
        </button>
      </div>

      {state.status === "loading" ? (
        <p className="hint">Loading evidence...</p>
      ) : null}

      {state.status === "error" ? (
        <p className="error-text">{state.message}</p>
      ) : null}

      {state.status === "ready" ? (
        <>
          <p className="hint">
            <strong>{state.explanation.event_id}</strong> �{" "}
            {state.explanation.evidence_counts.available} piece(s) of supporting
            evidence, {state.explanation.evidence_counts.unavailable} unavailable.
          </p>

          <p className="hint">{state.explanation.disclaimer}</p>

          <h3>Supporting evidence</h3>
          {state.explanation.available_evidence.length === 0 ? (
            <p className="hint">No evidence is available for this event.</p>
          ) : (
            <table className="table">
              <thead>
                <tr>
                  <th>Feature</th>
                  <th>Value</th>
                  <th>Type</th>
                  <th>Recorded by</th>
                  <th>Context</th>
                </tr>
              </thead>
              <tbody>
                {state.explanation.available_evidence.map((item) => (
                  <tr key={item.feature}>
                    <td>{item.feature}</td>
                    <td>
                      {item.value}
                      {item.unit ? ` ${item.unit}` : ""}
                    </td>
                    <td>{item.contribution_type}</td>
                    <td>{sourceLabel(item)}</td>
                    <td>{item.context}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}

          <h3>Evidence not available</h3>
          {state.explanation.unavailable_evidence.length === 0 ? (
            <p className="hint">
              Every item sought was available. That is unusual for this
              prototype, where most runs lack a trained model.
            </p>
          ) : (
            <ul>
              {state.explanation.unavailable_evidence.map((item) => (
                <li key={item.feature}>
                  <strong>{item.feature}</strong> � {item.context}
                </li>
              ))}
            </ul>
          )}

          <h3>Limitations</h3>
          <ul>
            {state.explanation.limitations.map((limitation) => (
              <li key={limitation}>{limitation}</li>
            ))}
          </ul>
        </>
      ) : null}

      {!eventId && state.status !== "ready" ? (
        <p className="hint">
          Explanations are produced for real detections only. No evidence is
          generated for an unknown event id.
        </p>
      ) : null}
    </section>
  );
}
