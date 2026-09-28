import sys, json
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, xarray as xr
from extreme_weather.anomaly.service import AnomalyDetectionService, EventStore
from extreme_weather.tracking.service import TrackingService
from extreme_weather.tracking.store import TrackStore
proc = Path("scratch_pipe/processed/era5_processed.nc")
d = xr.open_dataset(proc)
print("precip range", float(d.precipitation.min()), float(d.precipitation.max()))
svc = AnomalyDetectionService(store=EventStore(Path("scratch_pipe/processed/events.json")))
res = svc.run(proc, min_cells=1)
print("events", res.event_count, [ (e.event_id, e.variable, e.intensity, str(e.severity)) for e in res.events])
ts = TrackingService(event_store=EventStore(Path("scratch_pipe/processed/events.json")),
                     track_store=TrackStore(Path("scratch_pipe/processed/tracks.json")))
tr = ts.run(write_output=True)
print("tracks", tr.track_count, tr.event_count)
