import sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import extreme_weather.config as cfg
from extreme_weather.config import Settings
from extreme_weather.ingestion import ManifestStore, DatasetRegistry, build_manifest
from tests.ingestion_fixtures import make_dataset, write_netcdf
import shutil
tmp = Path("scratch_tmp3"); shutil.rmtree(tmp, ignore_errors=True)
st = Settings(data_dir=tmp, model_dir=tmp/"models"); st.ensure_directories()
st.real_raw_dir.mkdir(parents=True, exist_ok=True)
f = write_netcdf(make_dataset(), st.real_raw_dir/"c.nc")
m = build_manifest(make_dataset(), dataset_id="c", source_path=f, source="local_file", origin="HISTORICAL")
reg = DatasetRegistry(ManifestStore(st.manifests_dir)); reg.register(m)
# patch get_settings inside the config module
cfg.get_settings = lambda: st
import extreme_weather.ingestion.registry as R
print("validate:", reg.validate("c").valid)
try:
    reg.validate("nope")
except Exception as e:
    print("unknown:", type(e).__name__)
