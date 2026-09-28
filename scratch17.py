import sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import xarray as xr
from extreme_weather.anomaly.service import AnomalyDetectionService, EventStore
from extreme_weather.anomaly.thresholds import ThresholdLadder, ThresholdSet
from extreme_weather.schemas.event import EventType, Severity
proc = Path("scratch_pipe/processed/era5_processed.nc")
ladder = ThresholdLadder(name="precip_era5", event_type=EventType.EXTREME_RAINFALL,
                         variable="precipitation", steps=[(0.1, Severity.WATCH), (0.2, Severity.MODERATE), (0.3, Severity.EXTREME)])
ts = ThresholdSet(ladders=(ladder,))
svc = AnomalyDetectionService(store=EventStore(Path("scratch_pipe/processed/events.json")), thresholds=ts, min_cells=1)
res = svc.run(proc, min_cells=1, write_outputs=True)
print("events", res.event_count)
for e in res.events: print(" ", e.event_id, e.variable, round(e.intensity,3), e.severity, str(e.timestamp), e.latitude, e.longitude)
