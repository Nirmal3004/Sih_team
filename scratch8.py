import sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, xarray as xr
from tests.ingestion_fixtures import make_dataset, make_times
from extreme_weather.ingestion import check_quality
ds2 = make_dataset()
ds2["surface"] = xr.DataArray(np.full((4,3,3,2), 1.0), dims=("time","latitude","longitude","member"))
q = check_quality(ds2)
print("unexp", q.unexpected_dimensions, q.warnings, q.valid)
t = make_times(4)
tt = np.array([t[2],t[0],t[1],t[3]], dtype="datetime64[ns]")
q2 = check_quality(make_dataset(times=tt))
print("nonmono", q2.valid, q2.errors, q2.warnings)
q3 = check_quality(make_dataset(variables={"temperature": (np.full((4,3,3),300.0),"K")}))
print("missingstd", q3.missing_standard_variables, q3.valid)
