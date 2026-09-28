import sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
import numpy as np
from tests.ingestion_fixtures import make_dataset
try:
    ds = make_dataset()
    print("default OK", ds.sizes)
except Exception as e:
    print("default FAILS:", type(e).__name__, e)
