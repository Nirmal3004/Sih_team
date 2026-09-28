import sys, json
sys.path.insert(0, 'src')
import numpy as np
from pathlib import Path
from extreme_weather.ingestion import (IngestionRequest, IngestionService, OpenMeteoRequest,
    ERA5Source, era5_payload_to_dataset, normalize_dataset, check_quality)

payload = json.load(open('era5_sample.json'))
print('records:', len(payload), 'points:', payload[0]['latitude'], payload[0]['longitude'])
ds = era5_payload_to_dataset(payload)
print('raw dims   :', dict(ds.sizes))
print('raw vars   :', sorted(ds.data_vars))
print('raw units  :', {k: ds[k].attrs['units'] for k in sorted(ds.data_vars)})
print('raw is_syn :', ds.attrs.get('is_synthetic'))
print('lat asc    :', ds.latitude.values)
r = normalize_dataset(ds, variables=('precipitation','temperature','pressure','relative_humidity'), drop_unknown=True)
print('after norm :', sorted(r.dataset.data_vars))
print('converted  :', sorted(r.conversion_map().keys()))
for k, v in r.conversion_map().items():
    print('   ', k, v['original_units'], '->', v['normalized_units'], '|', v['conversion_method'])
print('temp range :', float(r.dataset['temperature'].min()), float(r.dataset['temperature'].max()))
print('pres range :', float(r.dataset['pressure'].min()), float(r.dataset['pressure'].max()))
q = check_quality(r.dataset)
print('valid:', q.valid, 'errors:', len(q.errors), 'warnings:', len(q.warnings))
for w in q.warnings[:4]: print('   W:', w)
