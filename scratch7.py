import sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, xarray as xr
from tests.ingestion_fixtures import make_dataset, make_times
from extreme_weather.ingestion import check_quality
shape=(4,3,3)
vals = np.full(shape, 300.0); vals[0,0,:] = np.nan; vals[0,1,0]=np.nan
q = check_quality(make_dataset(variables={"temperature": (vals, "K")}))
print("11pct", q.valid, q.warnings, round(q.missing_fraction,4))
# implausible
bad = np.full(shape, 400.0)
qb = check_quality(make_dataset(variables={"temperature": (bad, "K")}))
print("implaus warn", qb.valid, qb.warnings)
qbs = check_quality(make_dataset(variables={"temperature": (bad, "K")}), strict_ranges=True)
print("implaus strict", qbs.valid, qbs.errors)
# bad coords
q2 = check_quality(make_dataset(latitude=[-95.0, 12.0, 13.0]))
print("lat err", q2.valid, q2.errors)
q3 = check_quality(make_dataset(longitude=[0.0, 400.0, 77.0]))
print("lon err", q3.valid, q3.errors)
# dup coords
q4 = check_quality(make_dataset(latitude=[12.0, 12.0, 13.0]))
print("dupcoord", q4.duplicate_coordinates, q4.warnings)
# missing units
ds = make_dataset(); ds["temperature"].attrs.pop("units")
q5 = check_quality(ds)
print("nounits", q5.valid, q5.warnings)
# empty
q6 = check_quality(xr.Dataset())
print("empty", q6.valid, q6.errors)
# unexpected dim
ds7 = make_dataset().assign_coords(level=("time", np.array([1000.]*4)))
q7 = check_quality(ds7)
print("unexpdim", q7.unexpected_dimensions, q7.warnings)
