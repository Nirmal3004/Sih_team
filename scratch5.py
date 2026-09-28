import sys
sys.path.insert(0, "src")
import numpy as np, xarray as xr
t = np.array(["2024-06-01T00:00"], dtype="datetime64[ns]")
lat = np.array([1.0,2.0]); lon = np.array([3.0,4.0])
data = np.ones((1,2,2))
# current fixture form
try:
    xr.Dataset({"v": ((t, lat, lon), data, {"units":"K"})},
               coords={"time": t, "latitude": ("latitude", lat), "longitude": ("longitude", lon)})
    print("array-dims: OK")
except Exception as e:
    print("array-dims FAILS:", type(e).__name__)
# string-dims form
ds = xr.Dataset({"v": (("time","latitude","longitude"), data, {"units":"K"})},
           coords={"time": t, "latitude": ("latitude", lat), "longitude": ("longitude", lon)})
print("string-dims: OK", ds.sizes)
