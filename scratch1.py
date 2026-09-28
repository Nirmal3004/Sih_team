import json, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, xarray as xr
from extreme_weather.ingestion import era5_payload_to_dataset, normalize_dataset, check_quality
payload = json.loads(Path("tests/fixtures/era5_archive_sample.json").read_text())
ds = era5_payload_to_dataset(payload)
print(ds)
print("attrs", ds.attrs)
r = normalize_dataset(ds)
print(sorted(r.dataset.data_vars))
print("T", float(r.dataset.temperature.min()), float(r.dataset.temperature.max()))
print("P", float(r.dataset.pressure.min()), float(r.dataset.pressure.max()))
print("conv", [c.as_dict() for c in r.conversions])
print("missing", r.missing_variables)
q = check_quality(r.dataset)
print("quality", q.as_dict())
