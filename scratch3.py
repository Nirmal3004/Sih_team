import sys, json
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, xarray as xr
from extreme_weather.ingestion import check_quality, normalize_dataset
from tests.ingestion_fixtures import make_dataset, make_times

r = normalize_dataset(make_dataset(latitude=[13.0, 12.5, 12.0]))
print("lat", r.dataset.latitude.values, r.coordinate_actions)

r2 = normalize_dataset(make_dataset(longitude=[0.0, 120.0, 240.0]))
print("lon", r2.dataset.longitude.values, r2.coordinate_actions)

t = make_times(4)
tt = np.array([t[0], t[1], t[1], t[3]], dtype="datetime64[ns]")
q = check_quality(make_dataset(times=tt))
print("dup q", q.duplicate_timestamps, q.valid, q.errors, q.warnings)
r3 = normalize_dataset(make_dataset(times=tt))
print("dedup", r3.dataset.time.values, r3.coordinate_actions, r3.warnings)

tt2 = np.array([t[2], t[0], t[1], t[3]], dtype="datetime64[ns]")
r4 = normalize_dataset(make_dataset(times=tt2))
print("sorted", r4.dataset.time.values, r4.coordinate_actions)

shape=(4,3,3)
qn = check_quality(make_dataset(variables={"temperature": (np.full(shape, np.nan), "K")}))
print("allnan", qn.valid, qn.errors, qn.warnings)

vals = np.full(shape, 300.0); vals[0,0,:] = np.nan
q2 = check_quality(make_dataset(variables={"temperature": (vals, "K")}))
print("highnan", q2.valid, q2.warnings, q2.missing_fraction)
