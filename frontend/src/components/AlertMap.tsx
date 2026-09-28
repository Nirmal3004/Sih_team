/**
 * Alert map overlays (SIH26078, Phase 5).
 *
 * A **separate** map panel rather than a change to `WeatherMap`: the existing
 * Phase 0 map is proven and must not be broken, so this component reuses the
 * same Leaflet/React-Leaflet stack in its own container and adds only the alert
 * layers.
 *
 * Visual states are deliberately distinguishable without relying on colour
 * alone - each marker shape means one thing, and each carries a popup:
 *   - filled circle    = the current (observed) event position
 *   - dashed outline   = the affected radius, in km
 *   - outlined circle  = the GNN predicted position (only when one exists)
 *   - dashed line      = the move from the current to the predicted position
 *
 * Nothing here is a forecast product: the markers show stored pipeline output
 * on synthetic data, and the panel says so.
 */
import { useCallback, useEffect, useMemo, useState } from "react";
import {
  Circle,
  CircleMarker,
  MapContainer,
  Polyline,
  Popup,
  TileLayer,
  Tooltip,
} from "react-leaflet";

import { describeApiError, fetchAlerts } from "../services/alertsApi";
import type { Alert, AlertSeverity } from "../types/api";

const TILE_URL = "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png";
const TILE_ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors';

const DEFAULT_CENTER: [number, number] = [22.5, 79.0];
const DEFAULT_ZOOM = 5;

const SEVERITY_COLORS: Record<AlertSeverity, string> = {
  INFO: "#94a3b8",
  WATCH: "#eab308",
  ADVISORY: "#eab308",
  WARNING: "#f97316",
  SEVERE_WARNING: "#ef4444",
  EXTREME: "#dc2626",
};

type MapState =
  | { status: "loading" }
  | { status: "ready"; alerts: Alert[] }
  | { status: "error"; message: string };

/** Centre the map on the alerts, falling back to the default view. */
function resolveCenter(alerts: Alert[]): [number, number] {
  if (alerts.length === 0) {
    return DEFAULT_CENTER;
  }
  const lat = alerts.reduce((sum, alert) => sum + alert.current_lat, 0) / alerts.length;
  const lon = alerts.reduce((sum, alert) => sum + alert.current_lon, 0) / alerts.length;
  return [lat, lon];
}

function AlertMarker({ alert }: { alert: Alert }): React.JSX.Element {
  const color = SEVERITY_COLORS[alert.severity];
  const position: [number, number] = [alert.current_lat, alert.current_lon];
  const predicted: [number, number] | null =
    alert.predicted_lat !== null && alert.predicted_lon !== null
      ? [alert.predicted_lat, alert.predicted_lon]
      : null;

  return (
    <>
      <Circle
        center={position}
        radius={Math.max(alert.affected_radius_km * 1000, 1000)}
        pathOptions={{ color, fillColor: color, fillOpacity: 0.08, weight: 1, dashArray: "4 4" }}
      />
      <CircleMarker
        center={position}
        radius={7}
        pathOptions={{ color, fillColor: color, fillOpacity: 0.9, weight: 2 }}
      >
        <Tooltip direction="top" opacity={1}>
          {alert.severity} - {alert.status}
        </Tooltip>
        <Popup>
          <strong>{alert.title}</strong>
          <br />
          {alert.alert_id}
          <br />
          Current: {alert.current_lat.toFixed(3)} N, {alert.current_lon.toFixed(3)} E
          <br />
          Predicted:{" "}
          {predicted
            ? `${predicted[0].toFixed(3)} N, ${predicted[1].toFixed(3)} E`
            : "no GNN prediction available"}
          <br />
          Intensity: {alert.intensity.toFixed(2)}
          <br />
          Affected radius: {alert.affected_radius_km.toFixed(1)} km
          <br />
          Movement: {alert.movement_speed_kmh.toFixed(2)} km/h {alert.movement_direction}
          <br />
          Status: {alert.status}
          <br />
          Risk score: {alert.risk.score.toFixed(2)}/100
          <br />
          <em>Prototype output on synthetic data - not an official warning.</em>
        </Popup>
      </CircleMarker>
      {predicted ? (
        <>
          <Polyline
            positions={[position, predicted]}
            pathOptions={{ color, weight: 2, dashArray: "6 6", opacity: 0.8 }}
          />
          <CircleMarker
            center={predicted}
            radius={6}
            pathOptions={{ color, fillColor: "#0f172a", fillOpacity: 1, weight: 2 }}
          >
            <Tooltip direction="top" opacity={1}>
              GNN predicted position (prototype)
            </Tooltip>
            <Popup>
              <strong>Predicted position</strong>
              <br />
              {predicted[0].toFixed(3)} N, {predicted[1].toFixed(3)} E
              <br />
              Predicted intensity:{" "}
              {alert.predicted_intensity === null ? "n/a" : alert.predicted_intensity.toFixed(2)}
              <br />
              <em>Phase 3 prototype prediction - not a forecast.</em>
            </Popup>
          </CircleMarker>
        </>
      ) : null}
    </>
  );
}


export default function AlertMap(): React.JSX.Element {
  const [state, setState] = useState<MapState>({ status: "loading" });

  const load = useCallback(async (): Promise<void> => {
    setState({ status: "loading" });
    try {
      const response = await fetchAlerts();
      setState({ status: "ready", alerts: response.alerts });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const alerts = useMemo(
    () => (state.status === "ready" ? state.alerts : []),
    [state],
  );
  const center = resolveCenter(alerts);

  return (
    <section className="panel" aria-labelledby="alert-map-heading">
      <header className="panel-header">
        <h2 id="alert-map-heading">Alert map</h2>
        <div className="panel-actions">
          {state.status === "ready" ? (
            <span className="badge badge-phase">{alerts.length} plotted</span>
          ) : null}
          <button type="button" className="button" onClick={() => void load()}>
            Refresh
          </button>
        </div>
      </header>

      <div className="map">
        <MapContainer
          center={center}
          zoom={alerts.length > 0 ? 6 : DEFAULT_ZOOM}
          scrollWheelZoom={false}
          className="map-canvas"
        >
          <TileLayer url={TILE_URL} attribution={TILE_ATTRIBUTION} />
          {state.status === "ready"
            ? alerts.map((alert) => <AlertMarker key={alert.alert_id} alert={alert} />)
            : null}
        </MapContainer>
      </div>

      {state.status === "loading" ? (
        <p className="status status-loading" role="status">
          Loading alert positions ...
        </p>
      ) : null}
      {state.status === "error" ? (
        <p className="status status-error" role="alert">
          {state.message}
        </p>
      ) : null}
      {state.status === "ready" && alerts.length === 0 ? (
        <p className="hint">
          No alerts to plot. Run <code>python scripts/generate_alerts.py</code> first.
        </p>
      ) : null}

      <p className="hint">
        Markers show stored pipeline output, not a live feed: the filled circle is the observed
        event position, the dashed outline is the affected radius in km, the outlined circle is
        the Phase 3 GNN predicted position (shown only when a prediction exists), and the
        dashed line joins them. All data is synthetic and the severities are prototype
        categories, not official warnings.
      </p>
    </section>
  );
}

