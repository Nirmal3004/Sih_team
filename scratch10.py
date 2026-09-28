import sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, xarray as xr
from extreme_weather.config import Settings
from extreme_weather.ingestion import (IngestionService, IngestionRequest, NetCDFSource,
  ManifestStore, DatasetRegistry, build_manifest, file_hash, content_hash)
from extreme_weather.ingestion.exceptions import *
from tests.ingestion_fixtures import make_dataset, write_netcdf
import shutil
tmp = Path("scratch_tmp2"); shutil.rmtree(tmp, ignore_errors=True)
st = Settings(data_dir=tmp, model_dir=tmp/"models"); st.ensure_directories()
st.real_raw_dir.mkdir(parents=True, exist_ok=True)
f = write_netcdf(make_dataset(), tmp/"in" /"contract.nc")
print("before hash", file_hash(f)[:12])
reg = DatasetRegistry(ManifestStore(st.manifests_dir))
svc = IngestionService(registry=reg, settings=st)
r = svc.ingest(IngestionRequest(dataset_id="contract", path=f, origin="HISTORICAL"))
print("after  hash", file_hash(f)[:12], "equal:", file_hash(f)==r.manifest.file_hash)
print("input still valid netcdf:", sorted(xr.open_dataset(f).data_vars))
