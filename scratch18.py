import sys
from pathlib import Path
sys.path.insert(0, "src"); sys.path.insert(0, ".")
from extreme_weather.tracking.service import TrackingService
from extreme_weather.tracking.store import TrackStore
from extreme_weather.anomaly.service import EventStore
from extreme_weather.alerts.service import AlertService
from extreme_weather.alerts.store import AlertStore
from extreme_weather.alerts.notifications import MockNotificationProvider, NotificationStore
p = Path("scratch_pipe/processed")
ts = TrackingService(event_store=EventStore(p/"events.json"), track_store=TrackStore(p/"tracks.json"))
tr = ts.run(write_output=True)
print("tracks", tr.track_count, tr.event_count, tr.matched_event_count)
for t in tr.tracks: print(" ", t.track_id, t.status, t.point_count, t.duration_hours)
svc = AlertService(alert_store=AlertStore(p/"alerts.json"), event_store=EventStore(p/"events.json"),
                   track_store=TrackStore(p/"tracks.json"),
                   notification_providers=[MockNotificationProvider(NotificationStore(p/"notifications.json"))])
out = svc.generate()
print("alerts", out.stored.alert_count, "created", out.stored.created_count)
for a in out.stored.alerts: print(" ", a.alert_id, a.severity, a.status, a.event_id)
