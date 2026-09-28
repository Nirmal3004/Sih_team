import sys
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from extreme_weather.anomaly.thresholds import ThresholdSet, ThresholdLadder, default_thresholds
ts = default_thresholds()
for l in ts.ladders:
    print(l.name, l.variable, l.event_type, l.steps)
