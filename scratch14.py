import sys, json
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, xarray as xr
from extreme_weather.config import Settings
from extreme_weather.ingestion import (era5_payload_to_dataset, IngestionService, IngestionRequest,
  ManifestStore, DatasetRegistry)
from extreme_weather.data.preprocessing import preprocess_file
import shutil
tmp = Path("scratch_pipe"); shutil.rmtree(tmp, ignore_errors=True)
st = Settings(data_dir=tmp, model_dir=tmp/"models"); st.ensure_directories()
st.real_raw_dir.mkdir(parents=True, exist_ok=True)
payload = json.loads(Path("tests/fixtures/era5_archive_sample.json").read_text())
ds = era5_payload_to_dataset(payload)
src_nc = st.real_raw_dir/"era5_source.nc"
ds.to_netcdf(src_nc)
reg = DatasetRegistry(ManifestStore(st.manifests_dir))
svc = IngestionService(registry=reg, settings=st)
r = svc.ingest(IngestionRequest(dataset_id="era5_demo", path=src_nc, origin="HISTORICAL"))
print("ingested", r.output_path, r.quality.valid, r.quality.errors)
print("vars", r.manifest.variables, "missing", r.manifest.variables_missing)
print("coord actions", r.manifest.coordinate_actions)
ing = r.output_path
res = preprocess_file(ing, st.processed_dir/"era5_processed.nc", max_gap=3)
print("preprocessed", sorted(res.dataset.data_vars), res.report.warnings[:3])
