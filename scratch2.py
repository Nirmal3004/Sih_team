import json, sys
from pathlib import Path
sys.path.insert(0, "src")
import numpy as np, xarray as xr
from extreme_weather.ingestion import era5_payload_to_dataset
payload = json.loads(Path("tests/fixtures/era5_archive_sample.json").read_text())
ds = era5_payload_to_dataset(payload)
print("declared hourly_units:", payload[0]["hourly_units"]["wind_speed_10m"])
print("code-recorded units  :", ds.wind_speed_10m.attrs["units"])
print("value                :", float(ds.wind_speed_10m[0,0,0]))
