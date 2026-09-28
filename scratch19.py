import sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import shutil
from extreme_weather.config import Settings
from extreme_weather.data.generator import SyntheticConfig, ExtremeEventConfig, generate_synthetic_dataset
from extreme_weather.ingestion import IngestionService, IngestionRequest, ManifestStore, DatasetRegistry
tmp = Path("scratch_syn"); shutil.rmtree(tmp, ignore_errors=True)
st = Settings(data_dir=tmp, model_dir=tmp/"models"); st.ensure_directories()
st.synthetic_raw_dir.mkdir(parents=True, exist_ok=True)
cfg = SyntheticConfig(seed=7, time_steps=8, start_time="2024-06-01T00:00:00", time_step_hours=1,
  latitude_min=12.0, latitude_max=13.0, longitude_min=76.0, longitude_max=77.0, resolution_deg=0.5,
  event=ExtremeEventConfig(enabled=True, name="t", center_latitude=12.5, center_longitude=76.5,
    radius_deg=0.3, start_index=2, duration_steps=3, peak_precipitation_mm_per_hour=55.0,
    temperature_anomaly_k=-4.0))
ds = generate_synthetic_dataset(cfg, created_utc="2024-06-01T00:00:00Z")
print("vars", sorted(ds.data_vars), ds.attrs.get("is_synthetic"))
src = st.synthetic_raw_dir/"syn.nc"; ds.to_netcdf(src)
reg = DatasetRegistry(ManifestStore(st.manifests_dir))
svc = IngestionService(registry=reg, settings=st)
r = svc.ingest(IngestionRequest(dataset_id="syn", path=src, origin="SYNTHETIC"))
print("out", r.output_path, r.quality.valid, r.manifest.is_synthetic, r.manifest.status_label, r.manifest.source)
