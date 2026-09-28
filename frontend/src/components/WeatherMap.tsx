/**
 * Interactive map shell (Leaflet + React-Leaflet).
 *
 * Phase 0 only proves that the map stack works end to end: the base layer is
 * rendered and the domain of the first available synthetic dataset is
 * highlighted. The FUTURE anomaly/forecast layers, event markers and
 * downscaled tiles will be added to this same component later.
 */
import { useEffect, useState } from "react";
import { Circle, MapContainer, Popup, TileLayer } from "react-leaflet";

import { describeApiError, fetchDatasets, fetchDatasetSummary } from "../services/api";
import type { DatasetSummaryResponse } from "../types/api";

const TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const TILE_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

const DEFAULT_CENTER: [number, number] = [22.5, 79.0];
const DEFAULT_ZOOM = 5;

type MapState =
  | { status: "loading" }
  | { status: "success"; summary: DatasetSummaryResponse }
  | { status: "error"; message: string };

interface MapExtent {
  latitudeMin: number;
  latitudeMax: number;
  longitudeMin: number;
  longitudeMax: number;
  center: [number, number];
}

/** Extract a non-nullable extent, or null when the summary lacks coordinates. */
function resolveExtent(summary: DatasetSummaryResponse | null): MapExtent | null {
  if (summary === null) {
    return null;
  }
  const { latitude_min, latitude_max, longitude_min, longitude_max } = summary;
  if (
    latitude_min === null ||
    latitude_max === null ||
    longitude_min === null ||
    longitude_max === null
  ) {
    return null;
  }
  return {
    latitudeMin: latitude_min,
    latitudeMax: latitude_max,
    longitudeMin: longitude_min,
    longitudeMax: longitude_max,
    center: [
      (latitude_min + latitude_max) / 2,
      (longitude_min + longitude_max) / 2,
    ],
  };
}

export default function WeatherMap(): React.JSX.Element {
  const [state, setState] = useState<MapState>({ status: "loading" });

  useEffect(() => {
    let cancelled = false;

    const load = async (): Promise<void> => {
      try {
        const listing = await fetchDatasets();
        const first = listing.datasets[0];
        if (!first) {
          if (!cancelled) {
            setState({
              status: "error",
              message:
                "No dataset available yet - run `python scripts/generate_sample_data.py` to create one.",
            });
          }
          return;
        }
        const summary = await fetchDatasetSummary(first.name);
        if (!cancelled) {
          setState({ status: "success", summary });
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

  const summary = state.status === "success" ? state.summary : null;
  const extent = resolveExtent(summary);
  const center: [number, number] = extent === null ? DEFAULT_CENTER : extent.center;
  const zoom = extent === null ? DEFAULT_ZOOM : 6;

  return (
    <section className="panel" aria-labelledby="map-heading">
      <header className="panel-header">
        <h2 id="map-heading">Interactive weather map</h2>
        <span className="badge badge-phase">Phase 0 shell</span>
      </header>

      <div className="map">
        <MapContainer
          center={center}
          zoom={zoom}
          scrollWheelZoom={false}
          className="map-canvas"
        >
          <TileLayer url={TILE_URL} attribution={TILE_ATTRIBUTION} />
          {extent !== null && summary !== null ? (
            <Circle
              center={center}
              radius={400_000}
              pathOptions={{ color: "#38bdf8", fillColor: "#38bdf8", fillOpacity: 0.12 }}
            >
              <Popup>
                {summary.name} ({summary.path})
                <br />
                Synthetic domain: {extent.latitudeMin}&deg; to {extent.latitudeMax}&deg; N
                <br />
                {extent.longitudeMin}&deg; to {extent.longitudeMax}&deg; E
                {summary.approximate_grid_resolution_km !== null ? (
                  <>
                    <br />
                    Grid: {summary.grid_resolution_deg}&deg; (
                    {summary.approximate_grid_resolution_km} km)
                  </>
                ) : null}
              </Popup>
            </Circle>
          ) : null}
        </MapContainer>
      </div>

      {state.status === "loading" && (
        <p className="status status-loading" role="status">
          Loading dataset extent ...
        </p>
      )}
      {state.status === "error" && (
        <p className="status status-error" role="alert">
          {state.message}
        </p>
      )}
      {summary !== null && (
        <p className="hint">
          Synthetic dataset shown: <code>{summary.path}</code> - {summary.time_steps} time steps,{" "}
          {summary.dimensions.latitude}x{summary.dimensions.longitude} grid.
        </p>
      )}
      <p className="hint">
        FUTURE: anomaly overlays, event trajectories, downscaled 5 km tiles and alert polygons are
        rendered on this map from Phase 1 onwards.
      </p>
    </section>
  );
}
