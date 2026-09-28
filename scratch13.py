import sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, xarray as xr
from extreme_weather.ingestion import NetCDFSource, default_chunks
from tests.ingestion_fixtures import make_dataset, write_netcdf
import shutil
tmp = Path("scratch_tmp5"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
src = NetCDFSource()
f = write_netcdf(make_dataset(), tmp/"a.nc")
d = src.load(f)
print("no-chunks dask?", d.temperature.chunks is not None)
print("encoding source present:", "source" in d.temperature.encoding)
d.close()
d2 = src.load(f, chunks=default_chunks(4))
print("chunked dask?", d2.temperature.chunks)
d2.close()
# file lock proof: cannot delete while open
p = tmp/"lock.nc"; write_netcdf(make_dataset(), p)
op = src.load(p)
try:
    p.unlink(); print("deleted while open (no lock)")
except OSError as e:
    print("locked while open:", type(e).__name__)
op.close()
p.unlink(); print("deleted after close")
