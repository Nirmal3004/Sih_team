import sys
sys.path.insert(0, 'src')
import extreme_weather.ingestion as ing
print('OK exports:', len(ing.__all__))
print('sources:', sorted(ing.default_sources().keys()))
print('grib backend:', ing.grib_backend_name())
