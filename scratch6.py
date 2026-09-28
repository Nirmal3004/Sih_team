import sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, xarray as xr
print("DataArray.duplicated exists:", hasattr(xr.DataArray, "duplicated"))
print("DataArray.duplicated_like:", hasattr(xr.DataArray, "duplicated_like"))
from tests.ingestion_fixtures import make_dataset, make_times
from extreme_weather.ingestion import check_quality, normalize_dataset
t = make_times(4)
tt2 = np.array([t[2], t[0], t[1], t[3]], dtype="datetime64[ns]")
r4 = normalize_dataset(make_dataset(times=tt2))
print("sorted", r4.dataset.time.values, r4.coordinate_actions)

shape=(4,3,3)
qn = check_quality(make_dataset(variables={"temperature": (np.full(shape, np.nan), "K")}))
print("allnan", qn.valid, qn.errors, qn.warnings)
vals = np.full(shape, 300.0); vals[0,0,:] = np.nan
q2 = check_quality(make_dataset(variables={"temperature": (vals, "K")}))
print("highnan", q2.valid, q2.warnings, round(q2.missing_fraction,4))
