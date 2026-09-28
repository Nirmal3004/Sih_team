import sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from extreme_weather.ingestion import (DatasetManifest, build_manifest, file_hash, content_hash,
   ManifestStore, ProvenanceStep, DataOrigin, SOURCE_LOCAL_FILE)
from extreme_weather.ingestion.exceptions import IngestionError
from datetime import datetime, timezone
from tests.ingestion_fixtures import make_dataset, write_netcdf
import shutil, json
import xarray as xr
tmp = Path("scratch_man"); shutil.rmtree(tmp, ignore_errors=True); tmp.mkdir(parents=True)
ds = make_dataset()
f = write_netcdf(ds, tmp/"a.nc")
m = build_manifest(ds, dataset_id="a", source_path=f, source=SOURCE_LOCAL_FILE, origin="HISTORICAL")
print("dims", m.dimensions, "res", m.resolution_deg, m.resolution_km)
print("time", m.time_start, m.time_end, m.time_step_seconds)
print("hash", len(m.file_hash), m.file_hash==file_hash(f))
print("live", m.is_live, "synth", m.is_synthetic, "label", m.status_label)
try:
    DatasetManifest(dataset_id="x", source=SOURCE_LOCAL_FILE, retrieved_at=datetime.now(timezone.utc),
                    file_name="x.nc", file_hash="h", format="netcdf", source_url="https://evil")
except ValueError as e:
    print("refuses url OK")
# content hash stability across rewrite
h1 = content_hash(ds)
write_netcdf(ds, tmp/"b.nc")
print("content stable:", content_hash(xr.open_dataset(tmp/"b.nc")) == h1)
