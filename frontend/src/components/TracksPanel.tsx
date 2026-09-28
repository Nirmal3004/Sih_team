/**
 * Tracked extreme weather events (Phase 2).
 *
 * Renders what the backend tracker produced: track count, event type, start/end,
 * duration, peak intensity, maximum speed, direction, latest location and the
 * lifecycle status.
 *
 * Not implemented here (future phases): trajectory overlays on the map,
 * prediction cones, GNN-based linking and alerts.
 */
import { useCallback, useEffect, useState } from "react";

import { describeApiError, fetchTracks } from "../services/tracksApi";
import type { EventTrack, TrackStatus } from "../types/api";

type TracksState =
  | { status: "loading" }
  | { status: "success"; tracks: EventTrack[]; message: string | null; generatedAt: string | null }
  | { status: "error"; message: string };

const STATUS_ORDER: TrackStatus[] = ["ACTIVE", "NEW", "ENDED"];

function formatRange(track: EventTrack): string {
  const start = new Date(track.start_time);
  const end = new Date(track.end_time);
  const sameDay = start.toDateString() === end.toDateString();
  const options: Intl.DateTimeFormatOptions = sameDay
    ? { hour: "2-digit", minute: "2-digit" }
    : { day: "2-digit", month: "short", hour: "2-digit", minute: "2-digit" };
  const endLabel = end.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
  return `${start.toLocaleString(undefined, options)} - ${endLabel}`;
}

function statusBadge(track: EventTrack): React.JSX.Element {
  return <span className={`badge badge-${track.status.toLowerCase()}`}>{track.status}</span>;
}

function TrackCard({ track }: { track: EventTrack }): React.JSX.Element {
  const latest = track.points[track.points.length - 1];
  return (
    <article className="track">
      <div className="track-header">
        <span className="track-title">{track.track_id}</span>
        {statusBadge(track)}
      </div>
      <p className="track-type">
        {track.event_type} &middot; {track.points.length} point
        {track.points.length === 1 ? "" : "s"}
      </p>
      <dl className="kv">
        <div>
          <dt>Period</dt>
          <dd>{formatRange(track)}</dd>
        </div>
        <div>
          <dt>Duration</dt>
          <dd>{track.duration_hours.toFixed(2)} h</dd>
        </div>
        <div>
          <dt>Peak intensity</dt>
          <dd>
            {track.peak_intensity.toFixed(2)} {track.units ?? ""}
          </dd>
        </div>
        <div>
          <dt>Max speed</dt>
          <dd>{track.max_speed_kmh.toFixed(2)} km/h</dd>
        </div>
        <div>
          <dt>Direction</dt>
          <dd>
            {track.direction}
            {track.bearing_degrees !== null ? ` (${track.bearing_degrees.toFixed(0)} deg)` : ""}
          </dd>
        </div>
        <div>
          <dt>Latest location</dt>
          <dd>
            {latest.latitude.toFixed(3)} N, {latest.longitude.toFixed(3)} E
          </dd>
        </div>
        <div>
          <dt>Trend</dt>
          <dd>{track.intensity_trend}</dd>
        </div>
        <div>
          <dt>Distance</dt>
          <dd>{track.distance_travelled_km.toFixed(1)} km</dd>
        </div>
      </dl>
    </article>
  );
}

export default function TracksPanel(): React.JSX.Element {
  const [state, setState] = useState<TracksState>({ status: "loading" });

  const load = useCallback(async (): Promise<void> => {
    setState({ status: "loading" });
    try {
      const response = await fetchTracks();
      setState({
        status: "success",
        tracks: response.tracks,
        message: response.message,
        generatedAt: response.generated_at,
      });
    } catch (error) {
      setState({ status: "error", message: describeApiError(error) });
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const tracks = state.status === "success" ? state.tracks : [];
  const statusSummary = STATUS_ORDER.filter((status) =>
    tracks.some((track) => track.status === status),
  )
    .map((status) => `${status}=${tracks.filter((track) => track.status === status).length}`)
    .join(", ");

  return (
    <section className="panel" aria-labelledby="tracks-heading">
      <header className="panel-header">
        <h2 id="tracks-heading">Tracked extreme events</h2>
        <div className="panel-actions">
          {state.status === "success" ? (
            <span className="badge badge-phase">{tracks.length} tracks</span>
          ) : null}
          <button type="button" className="button" onClick={() => void load()}>
            Refresh
          </button>
        </div>
      </header>

      {state.status === "loading" && (
        <p className="status status-loading" role="status">
          Loading tracked events ...
        </p>
      )}

      {state.status === "error" && (
        <p className="status status-error" role="alert">
          {state.message}
        </p>
      )}

      {state.status === "success" && (
        <>
          {state.message ? <p className="hint">{state.message}</p> : null}
          {state.generatedAt ? (
            <p className="hint">Tracking run: {new Date(state.generatedAt).toLocaleString()}</p>
          ) : null}
          {statusSummary ? <p className="hint">By status: {statusSummary}</p> : null}

          {tracks.length === 0 ? null : (
            <div className="track-list">
              {tracks.map((track) => (
                <TrackCard key={track.track_id} track={track} />
              ))}
            </div>
          )}

          <p className="hint">
            Trajectories on the map, prediction cones, GNN linking and alerts are FUTURE phases.
          </p>
        </>
      )}
    </section>
  );
}
