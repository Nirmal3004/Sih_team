# Data ingestion (Phase 6)

How a **real** or **historical** weather file becomes a dataset the Phase 1-5
pipeline can consume, without changing that pipeline at all.

```text
WeatherDataSource            netcdf | grib | era5 | nwp
        |
        |  fetch()      a real path, or a clear error - never fabricated data
        v
   load()                   opened in the SOURCE's own units
        |
        v
   check_quality()          DataQualityReport: valid / errors / warnings
        |
        v
   normalize_dataset()      standard units + coordinates, every step recorded
        |
        v
   build_manifest()         provenance, hashes, origin, synthetic/real flag
        |
        v
   DatasetRegistry          a queryable entry
        |
        v
   Phase 1 anomaly detection  (unchanged)
```

## Three rules

1. **Synthetic and real never mix.** Every dataset declares an `origin` and is
   written to `data/raw/synthetic` or `data/raw/real`. There is *no default
   origin*, because guessing is how real data ends up presented as synthetic.
2. **Never fabricate.** A missing file, an unreachable provider, a missing
   optional dependency and a missing credential each raise a specific,
   actionable error. None produces a plausible-looking empty dataset.
3. **Never invent provenance.** A local file has `source="local_file"` and no
   URL. The manifest constructor *refuses* that combination outright, so no call
   site can bypass it.

## Quick start

No API key is needed for local-file ingestion.

```bash
# Look at a file before committing anything to the data directory.
python scripts/ingest_weather.py --file data/raw/real/my_data.nc --validate

# Ingest it: validate -> normalise -> manifest -> register.
python scripts/ingest_weather.py --file data/raw/real/my_data.nc

# Ingest and additionally run the Phase 0 preprocessing pipeline.
python scripts/ingest_weather.py --file data/raw/real/my_data.nc --process

# Fetch a bounded ERA5 subset (real past weather, no key required).
python scripts/ingest_weather.py --fetch-era5 \
    --era5-lats 12.5,13.0,13.5 --era5-lons 76.5,77.0,77.5 \
    --start 2024-06-01 --end 2024-06-03 --dataset-id era5_real --process

# Continue the pipeline on the processed file.
python scripts/detect_anomalies.py --input data/processed/era5_real_processed.nc \
    --events-output data/processed/real_events.json
python scripts/track_events.py --events data/processed/real_events.json \
    --tracks-output data/processed/real_tracks.json
python scripts/generate_alerts.py --events data/processed/real_events.json \
    --tracks data/processed/real_tracks.json \
    --alerts-output data/processed/real_alerts.json
```

From Python:

```python
from extreme_weather.ingestion import IngestionRequest, IngestionService

service = IngestionService()
result = service.ingest(
    IngestionRequest(
        dataset_id="era5_demo",
        path="data/cache/era5_demo.nc",
        origin="HISTORICAL",       # SYNTHETIC | HISTORICAL | EXTERNAL
    )
)
print(result.quality.as_dict())
print(result.manifest.status_label)   # "HISTORICAL"
```

## Sources

Four sources ship, all satisfying `WeatherDataSource`
(`name`, `available`, `fetch`, `load`, `metadata`, `default_origin`). No provider
is hard-coded into the system; a fifth is one function away.

| Source    | Kind   | Formats                 | Credentials | Notes                                |
| --------- | ------ | ----------------------- | ----------- | ------------------------------------ |
| `netcdf`  | file   | `.nc` `.nc4` `.cdf`     | none        | Always available. Lazy loading.       |
| `grib`    | file   | `.grib` `.grib2` `.grb` | none        | **Optional dependency**, see below.   |
| `era5`    | remote | JSON, then NetCDF       | none        | Real past reanalysis via Open-Meteo.  |
| `nwp`     | both   | GRIB, NetCDF            | optional    | Model output; `NWP_API_TOKEN` if needed. |

A source that is registered but unusable is still **listed**, with
`available: false` and the reason. A dashboard can then show *why* GRIB is
unavailable rather than implying the format is unsupported.

### GRIB is an optional dependency

Reading GRIB needs an Xarray backend (`cfgrib`/`eccodes` or `pygrib`). These
are **not** hard dependencies: they are large native builds, painful to install
on Windows, and unnecessary for the NetCDF path the project already uses. When
one is absent, `GRIBSource.load()` raises a `MissingDependencyError` naming the
install command. It never degrades into an empty dataset, which would look like
"no events found".

```bash
pip install cfgrib eccodes        # optional; only needed for GRIB
```

## Standard data model

Normalised datasets carry the Phase 0 contract, which Phases 1-5 already
expect — so nothing downstream needed to change.

| Variable            | Standard unit    |
| ------------------- | ---------------- |
| `temperature`       | K                |
| `precipitation`     | mm h-1 (a rate)  |
| `pressure`          | Pa               |
| `relative_humidity` | %                |
| `u_wind` / `v_wind` | m s-1            |

with `time` / `latitude` / `longitude`, latitude **ascending** and longitude in
**[-180, 180]**.

**A dataset that lacks a variable is reported, not completed.** A subset of ERA5
without `u_wind` is a valid dataset; Phase 1 simply does not use wind. Missing
variables appear in `manifest.variables_missing` and in a normalization warning.

## Unit normalisation

Every conversion is explicit and recorded as a `ConversionRecord`:

```json
{
  "variable": "temperature",
  "original_units": "degC",
  "normalized_units": "K",
  "conversion_method": "Celsius -> Kelvin (offset +273.15)",
  "scale": 1.0
}
```

Handled conversions include `degC -> K` (offset), `hPa -> Pa` (x100),
`km/h -> m s-1` (/3.6), `knot -> m s-1`, `mm -> mm h-1` (relabel), and
`kg m-2 -> mm h-1` (1 kg/m² is exactly 1 mm of liquid equivalent).

**An unrecognised unit raises `NormalizationError` rather than being guessed.**
Guessing that `mm/day` is `mm/h` would silently corrupt every downstream
threshold, and the Phase 1 severity ladder is calibrated in specific units.

A wind-speed unit on a non-wind variable is refused too — converting `km/h` into
`mm h-1` would change what the variable *means*.

### Precipitation accumulation → rate

Only performed when the source metadata makes it unambiguous. An hourly `mm`
accumulation **is** an hourly rate and is relabelled. A *daily* accumulation is
**not** converted, because the metadata does not say what the averaging window
is; guessing would be wrong for every downstream threshold.

## Coordinate normalisation

| Transform             | Behaviour                                    |
| --------------------- | -------------------------------------------- |
| Descending latitude   | Sorted ascending, recorded in the manifest.   |
| Non-monotonic time    | Sorted ascending, recorded.                   |
| Duplicate timestamps  | Counted, first occurrence kept, recorded.     |
| `0-360` longitude     | Wrapped into `[-180, 180]`, recorded.         |
| Missing time/lat/lon  | `NormalizationError` — cannot be processed.   |

Every transform is written into `manifest.coordinate_actions`. Nothing changes
the geographic meaning of a dataset silently.

### Wind components from speed + direction

Many archives (including the Open-Meteo one used here) publish `wind_speed` and
`wind_direction` rather than u/v components, while the Phase 0 contract needs
components. The standard reversible conversion is applied:

```text
u = -speed * sin(radians(direction))
v = -speed * cos(radians(direction))
```

using the **meteorological** convention already documented in
`extreme_weather.data.derived`. The manifest records that the components were
**derived**, not observed, and the polar intermediates are then dropped.

## Registry

The registry **extends** rather than replaces the Phase 0 dataset API. The Phase
0 API lists `.nc` files on disk; the registry knows which were deliberately
ingested, where they came from, whether they are synthetic, and what was done to
them. Both views are useful and neither replaces the other.

```python
registry = DatasetRegistry()
registry.list(origin="HISTORICAL")          # separate the two
registry.find_by_variable("precipitation")
registry.find_by_time("2024-06-01", "2024-06-03")
registry.validate("era5_real")              # re-run the quality check
```

## API

| Method | Path                                | Purpose                                  |
| ------ | ----------------------------------- | ---------------------------------------- |
| GET    | `/api/v1/data/sources`              | Registered sources + availability.        |
| GET    | `/api/v1/data/datasets`             | Ingested datasets (filterable).           |
| GET    | `/api/v1/data/datasets/{id}`        | One manifest.                            |
| GET    | `/api/v1/data/datasets/{id}/quality`| Its quality report.                      |
| POST   | `/api/v1/data/validate`             | Quality-check a path. **Writes nothing.** |
| POST   | `/api/v1/data/ingest`               | Fetch → normalise → manifest.            |
| POST   | `/api/v1/data/datasets/{id}/process`| Run Phase 0 preprocessing.               |

Paths are resolved inside the configured data directory; a traversal attempt is
rejected, mirroring the Phase 0 API's guard.

## Large data safety

A real global reanalysis file is tens of gigabytes, so:

* `NetCDFSource.load()` opens **lazily** — the caller decides when to
  materialise. `open_context()` loads and closes for callers that need to write.
* `chunks=` is forwarded to Xarray/Dask. Recommended chunking is roughly **one
  day per chunk along time** (`default_chunks()`), leaving spatial dimensions
  whole.
* A download larger than `MAX_DOWNLOAD_MB` (default 512 MB) is refused before it
  is written, so a mistyped global request cannot fill the disk.
* `file_hash` reads the file in 1 MiB blocks rather than loading it.

## Security

* No API key, token or password is committed, logged or returned.
* `NWP_API_TOKEN` and `OPEN_METEO_API_KEY` are optional and only ever *read*.
* A missing credential raises `ProviderCredentialsMissingError` naming the
  environment variable, never its value.
* The API never returns a secret, and the source list was tested to contain no
  credential markers.

See `docs/DATA_QUALITY.md` for the checks, `docs/REAL_DATA_VALIDATION.md` for
the evaluation methodology and limitations, and `docs/ARCHITECTURE.md` for how
this sits alongside Phases 0-5.

