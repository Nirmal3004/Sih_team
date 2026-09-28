import AlertMap from "./components/AlertMap";
import AlertsPanel from "./components/AlertsPanel";
import DataSourcePanel from "./components/DataSourcePanel";
import DatasetBrowser from "./components/DatasetBrowser";
import DownscalingPanel from "./components/DownscalingPanel";
import GNNPanel from "./components/GNNPanel";
import HealthStatus from "./components/HealthStatus";
import SystemStatus from "./components/SystemStatus";
import TracksPanel from "./components/TracksPanel";
import ValidationPanel from "./components/ValidationPanel";
import PipelineStatusPanel from "./components/PipelineStatusPanel";
import OperationsPanel from "./components/OperationsPanel";
import EvaluationPanel from "./components/EvaluationPanel";
import ExplainabilityPanel from "./components/ExplainabilityPanel";
import WeatherIntelligencePanel from "./components/WeatherIntelligencePanel";
import WeatherMap from "./components/WeatherMap";

const PROJECT_ID = "SIH26078";
const PROJECT_NAME = "DeepWeather AI";
const PHASE = "Phase 11 - SIH demo, integration and submission readiness";

export default function App(): React.JSX.Element {
  return (
    <div className="app">
      <header className="app-header">
        <div>
          <h1>{PROJECT_NAME}</h1>
          <p className="subtitle">
            Extreme weather intelligence <span className="muted">({PROJECT_ID})</span>
          </p>
        </div>
        <span className="badge badge-phase">{PHASE}</span>
      </header>

      <div className="banner" role="note">
        <strong>Research prototype — not an operational warning system.</strong>{" "}
        No forecast skill has been demonstrated and no accuracy, lives-saved or economic-benefit
        figure has been measured. The alert severities shown are this project's own categories,
        not official IMD or WMO warnings. Do not use for emergency decisions.
      </div>

      <main className="grid">
        <SystemStatus />
        <HealthStatus />
        <AlertsPanel />
        <PipelineStatusPanel />
        <OperationsPanel />
        <EvaluationPanel />
        <ExplainabilityPanel />
        <ValidationPanel />
        <DatasetBrowser />
        <DataSourcePanel />
        <TracksPanel />
        <GNNPanel />
        <DownscalingPanel />
        <WeatherIntelligencePanel />
        <AlertMap />
        <WeatherMap />
      </main>

      <footer className="app-footer">
        <p className="hint">
          Data is either generated locally or retrieved as historical reanalysis; nothing here is a
          live feed or a forecast. Phase 3 predicts the next state of a tracked event, Phase 4
          produces a ~5 km field from a ~12 km one, and Phase 5 raises rule-based alerts on
          top of both. All three are research prototypes. The alert severities are project
          prototype categories - they are NOT official IMD or WMO warnings and must not be used
          for emergency decisions.
        </p>
      </footer>
    </div>
  );
}
