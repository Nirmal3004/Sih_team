/**
 * Data source panel (SIH26078, Phase 6).
 *
 * Lists the registered ingestion sources and, for each, whether it can be used
 * right now and what it would require.
 *
 * A source that is registered but unusable (no GRIB engine, no NWP endpoint) is
 * shown with its reason rather than hidden: a dashboard that silently omits
 * GRIB would read as "this project cannot read GRIB", which is different from
 * "you have not installed the optional backend".
 */
import { useCallback, useEffect, useState } from "react";

import { describeApiError, fetchDataSources } from "../services/dataApi";
import type { DataSourceInfo } from "../types/api";

type State =
  | { status: "loading" }
  | { status: "ready"; sources: DataSourceInfo[] }
  | { status: "error"; message: string };

/** Read the trailing `available=True/False` the backend appends to the notes. */
function availabilityOf(source: DataSourceInfo): boolean | null {
  const match = /available=(True|False)\s*$/.exec(source.notes.trim());
  if (!match) {
    return null;
  }
  return match[1] === "True";
}

export default function DataSourcePanel(): React.JSX.Element {
  const [state, setState] = useState<State>({ status: "loading" });

  const load = useCallback(async (): Promise<void> => {
    setState({ status: "loading" });
    try {
      const response = await fetchDataSources();
      setState({ status: "ready", sources: response.sources });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  if (state.status === "loading") {
    return (
      <section className="panel" aria-labelledby="data-sources-heading">
        <header className="panel-header">
          <h2 id="data-sources-heading">Data sources</h2>
        </header>
        <p className="status status-loading" role="status">
          Loading data sources ...
        </p>
      </section>
    );
  }

  if (state.status === "error") {
    return (
      <section className="panel" aria-labelledby="data-sources-heading">
        <header className="panel-header">
          <h2 id="data-sources-heading">Data sources</h2>
          <button type="button" className="button" onClick={() => void load()}>
            Retry
          </button>
        </header>
        <p className="status status-error" role="alert">
          {state.message}
        </p>
      </section>
    );
  }

  return (
    <section className="panel" aria-labelledby="data-sources-heading">
      <header className="panel-header">
        <h2 id="data-sources-heading">Data sources</h2>
        <div className="panel-actions">
          <span className="badge badge-phase">{state.sources.length} registered</span>
          <button type="button" className="button" onClick={() => void load()}>
            Refresh
          </button>
        </div>
      </header>

      <div className="source-list">
        {state.sources.map((source) => {
          const available = availabilityOf(source);
          return (
            <article key={source.name} className="source">
              <div className="source-header">
                <span className="source-name">{source.name}</span>
                <span className="badge badge-source-kind">{source.kind}</span>
                {available === null ? null : (
                  <span
                    className={
                      available ? "badge badge-source-ok" : "badge badge-source-missing"
                    }
                  >
                    {available ? "available" : "not available"}
                  </span>
                )}
                <span className="badge badge-origin">{source.default_origin}</span>
              </div>
              <p className="source-description">{source.description}</p>
              <p className="hint">
                Formats: {source.formats.length > 0 ? source.formats.join(", ") : "n/a"}
                {source.optional_dependency
                  ? ` · Dependency: ${source.optional_dependency}`
                  : ""}
                {source.requires_credentials
                  ? ` · Requires: ${source.required_environment.join(", ")}`
                  : " · No credentials required"}
              </p>
              {source.notes ? <p className="hint">{source.notes}</p> : null}
            </article>
          );
        })}
      </div>

      <p className="hint">
        A source that is registered but unavailable is listed with its reason rather than
        hidden. No source in this project is a live feed: every one reads a file or a
        bounded historical request, and none of them is an operational forecast.
      </p>
    </section>
  );
}
