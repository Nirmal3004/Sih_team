import sys, json
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np, xarray as xr
from extreme_weather.config import Settings
from extreme_weather.ingestion import (IngestionService, IngestionRequest, NetCDFSource,
  ManifestStore, DatasetRegistry, build_manifest, file_hash, content_hash, grib_backend_available)
from tests.ingestion_fixtures import make_dataset, write_netcdf

tmp = Path("scratch_tmp"); import shutil; shutil.rmtree(tmp, ignore_errors=True)
st = Settings(data_dir=tmp, model_dir=tmp/"models"); st.ensure_directories()
st.real_raw_dir.mkdir(parents=True, exist_ok=True); st.synthetic_raw_dir.mkdir(parents=True, exist_ok=True)
f = write_netcdf(make_dataset(), st.real_raw_dir/"contract.nc")
reg = DatasetRegistry(ManifestStore(st.manifests_dir))
svc = IngestionService(registry=reg, settings=st)
r = svc.ingest(IngestionRequest(dataset_id="contract", path=f, origin="HISTORICAL"))
print("output", r.output_path)
print("valid", r.quality.valid, r.quality.errors, r.quality.warnings)
m = r.manifest
print("source", m.source, m.source_url, "hash", m.file_hash[:12], m.content_hash[:12])
print("stages", [s.stage for s in m.processing_steps])
print("vars", m.variables)
print("registered", reg.ids())
# determinism
r2 = svc.ingest(IngestionRequest(dataset_id="contract", path=f, origin="HISTORICAL"))
print("same content hash:", m.content_hash == r2.manifest.content_hash)
# synthetic
r3 = svc.ingest(IngestionRequest(dataset_id="synth", path=f, origin="SYNTHETIC"))
print("synth out", r3.output_path, r3.manifest.is_synthetic, r3.manifest.status_label)
# dry run
r4 = svc.ingest(IngestionRequest(dataset_id="dry", path=f, origin="HISTORICAL", dry_run=True))
print("dry", r4.output_path, r4.entry, (st.real_raw_dir/"dry.nc").exists())
print("grib", grib_backend_available())
