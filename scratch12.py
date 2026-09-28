import sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, xarray as xr
from extreme_weather.ingestion import NetCDFSource, default_chunks
from extreme_weather.ingestion.netcdf import file_timestamp
from extreme_weather.ingestion.exceptions import *
from tests.ingestion_fixtures import make_dataset, write_netcdf
import shutil
tmp = Path("scratch_tmp4"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
src = NetCDFSource()
f = write_netcdf(make_dataset(), tmp/"a.nc")
print("fetch", src.fetch(f) == f.resolve())
print("chunks", default_chunks(4), default_chunks(1000), default_chunks(0))
print("ts", file_timestamp(f))
# corrupt
bad = tmp/"bad.nc"; bad.write_text("not netcdf at all")
try: src.load(bad)
except Exception as e: print("corrupt:", type(e).__name__, str(e)[:60])
# missing
try: src.load(tmp/"nope.nc")
except Exception as e: print("missing:", type(e).__name__)
# wrong ext
w = tmp/"x.txt"; w.write_text("hi")
try: src.fetch(w)
except Exception as e: print("wrongext fetch:", type(e).__name__)
# base_dir traversal
base = tmp/"base"; base.mkdir()
s2 = NetCDFSource(base_dir=base)
write_netcdf(make_dataset(), base/"in.nc")
print("inside ok:", s2.fetch("in.nc").name)
try: s2.fetch("../a.nc")
except Exception as e: print("traversal:", type(e).__name__)
# open_context closes
with src.open_context(f) as d:
    print("ctx loaded", d.temperature.shape)
print("closed ok")
# lazy
d = src.load(f)
print("lazy", d.temperature.values.__class__.__name__)
d.close()
