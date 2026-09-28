/**
 * Placeholder "weather intelligence" panel.
 *
 * Phase 0 only renders the capabilities the platform is designed for. Every card
 * is explicitly marked FUTURE, so the judges can see the roadmap without any
 * backend AI endpoint being present yet.
 */
import { useEffect, useState } from "react";

import { describeApiError, fetchDatasets } from "../services/api";
import type { DatasetFileInfo } from "../types/api";

interface Capability {
  title: string;
  phase: string;
  description: string;
}

const CAPABILITIES: Capability[] = [
  {
    title: "Extreme anomaly detection",
    phase: "Phase 1",
    description:
      "Percentile and standardised-anomaly detection over historical baselines, producing anomaly fields per time step.",
  },
  {
    title: "Event identification & tracking",
    phase: "Phase 2",
    description:
      "Connected-component blobs with stable IDs, trajectories, speed, heading and life-cycle intensity.",
  },
  {
    title: "GNN spatio-temporal tracking",
    phase: "Phase 3",
    description:
      "Graph Neural Network (PyTorch Geometric) refining event linkage across the time axis.",
  },
  {
    title: "12 km -> 5 km downscaling",
    phase: "Phase 4",
    description:
      "AI super-resolution (diffusion/CNN) behind the same registry as the Phase 0 nearest/cubic baselines.",
  },
  {
    title: "Alerts & notifications",
    phase: "Phase 5",
    description:
      "Threshold- and model-driven alert generation with region, severity and confidence.",
  },
];

type DatasetState =
  | { status: "loading" }
  | { status: "success"; dataDir: string; datasets: DatasetFileInfo[] }
  | { status: "error"; message: string };

export default function WeatherIntelligencePanel(): React.JSX.Element {
  const [state, setState] = useState<DatasetState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;

    const load = async (): Promise<void> => {
      try {
        const response = await fetchDatasets();
        if (!cancelled) {
          setState({ status: "success", dataDir: response.data_dir, datasets: response.datasets });
        }
      } catch (error) {
        if (!cancelled) {
          setState({ status: "error", message: describeApiError(error) });
        }
      }
    };

    void load();
    return () => {
      cancelled = true;
    };
  }, []);

  return (
    <section className="panel" aria-labelledby="intelligence-heading">
      <header className="panel-header">
        <h2 id="intelligence-heading">Weather intelligence</h2>
        <span className="badge badge-phase">FUTURE</span>
      </header>

      <p className="hint">
        Detection, tracking and downscaling arrive in the next phases. Phase 0 ships the
        deterministic data foundation these modules will run on.
      </p>

      <ul className="capability-list">
        {CAPABILITIES.map((capability) => (
          <li key={capability.title} className="capability">
            <div className="capability-header">
              <span className="capability-title">{capability.title}</span>
              <span className="badge badge-phase">{capability.phase}</span>
            </div>
            <p className="capability-description">{capability.description}</p>
          </li>
        ))}
      </ul>

      <div className="datasets" aria-live="polite">
        <h3>Available datasets (Phase 0 data layer)</h3>
        {state.status === "loading" && <p className="status status-loading">Loading datasets ...</p>}
        {state.status === "error" && (
          <p className="status status-error" role="alert">
            {state.message}
          </p>
        )}
        {state.status === "success" && (
          <>
            <p className="hint">Data directory: {state.dataDir}</p>
            {state.datasets.length === 0 ? (
              <p className="hint">
                No NetCDF files yet - run <code>python scripts/generate_sample_data.py</code>.
              </p>
            ) : (
              <ul className="dataset-list">
                {state.datasets.map((dataset) => (
                  <li key={dataset.name}>
                    <code>{dataset.name}</code>{" "}
                    <span className="muted">
                      ({(dataset.size_bytes / 1024).toFixed(1)} KiB, modified{" "}
                      {new Date(dataset.modified_utc).toLocaleString()})
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </>
        )}
      </div>
    </section>
  );
}
