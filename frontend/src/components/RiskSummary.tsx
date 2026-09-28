/**
 * Risk explanation view (SIH26078, Phase 5).
 *
 * The point of this component is that **every alert must be able to say why it
 * exists**. It shows the score, the severity it maps onto, and every factor
 * that contributed - with the weight, the normalised value and the points
 * actually added, so the arithmetic can be checked by hand.
 *
 * What it deliberately never shows:
 *  - a percentage accuracy or confidence figure. The score is a rule-based
 *    triage aid with no probability meaning;
 *  - any "guaranteed" phrasing or "exact evacuation zone" style claim.
 */
import type { Alert } from "../types/api";

/**
 * Split the backend's multi-line explanation into one reason per line.
 *
 * The server builds the explanation as "\n"-joined lines (see
 * `RiskAssessment.reason_lines`), so splitting here shows exactly the reasons
 * the backend recorded - no client-side re-interpretation.
 */
function reasonLines(explanation: string): string[] {
  return explanation
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.length > 0);
}

export default function RiskSummary({ alert }: { alert: Alert }): React.JSX.Element {
  const { risk } = alert;
  const reasons = reasonLines(risk.explanation);
  const contributing = risk.contributing_factors.filter((factor) => factor.points > 0);
  const inactive = risk.contributing_factors.filter((factor) => factor.points === 0);

  return (
    <section className="risk-summary" aria-labelledby="risk-summary-heading">
      <h3 id="risk-summary-heading">
        Risk explanation &mdash; priority {risk.level}
      </h3>

      <div className="risk-meter" role="img" aria-label={`Risk score ${risk.score} of 100`}>
        <div className="risk-meter-fill" style={{ width: `${Math.min(100, risk.score)}%` }} />
        <span className="risk-meter-label">{risk.score.toFixed(2)} / 100</span>
      </div>

      <ul className="risk-reasons">
        {reasons.map((line, index) => (
          <li key={`${index}-${line.slice(0, 24)}`}>{line}</li>
        ))}
      </ul>

      {contributing.length > 0 ? (
        <table className="risk-table">
          <caption>
            Every point of the score, by factor. The sum of the contributed points is the
            score; nothing is hidden.
          </caption>
          <thead>
            <tr>
              <th scope="col">Factor</th>
              <th scope="col">Raw value</th>
              <th scope="col">Normalised</th>
              <th scope="col">Weight</th>
              <th scope="col">Points</th>
            </tr>
          </thead>
          <tbody>
            {contributing.map((factor) => (
              <tr key={factor.name}>
                <th scope="row">{factor.label}</th>
                <td>{factor.raw_value === null ? "n/a" : factor.raw_value.toFixed(2)}</td>
                <td>{factor.normalized_value.toFixed(3)}</td>
                <td>{factor.weight.toFixed(2)}</td>
                <td>{factor.points.toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}

      {inactive.length > 0 ? (
        <details className="risk-inactive">
          <summary>
            {inactive.length} factor(s) contributed nothing (unavailable or zero)
          </summary>
          <ul>
            {inactive.map((factor) => (
              <li key={factor.name}>
                <strong>{factor.label}</strong>: {factor.detail}
              </li>
            ))}
          </ul>
        </details>
      ) : null}

      <p className="hint">
        Thresholds used: WARNING &ge; {risk.severity_thresholds.warning}, SEVERE_WARNING &ge;{" "}
        {risk.severity_thresholds.severe_warning}, EXTREME &ge;{" "}
        {risk.severity_thresholds.extreme}. Assessed at{" "}
        {new Date(risk.generated_at).toLocaleString()}.
      </p>
      <p className="hint">{risk.notice}</p>
      <p className="hint">
        Input completeness: <strong>{alert.confidence_label}</strong>. This describes which
        pipeline stages were available, not how likely the alert is to be correct.
      </p>
    </section>
  );
}
