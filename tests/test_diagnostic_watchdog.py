"""Actual child-process supervision tests; these assert no robot result."""
import importlib.util
import json
from pathlib import Path
import sys

spec=importlib.util.spec_from_file_location('watch_diagnostic',Path('scripts/watch_diagnostic.py'))
watch=importlib.util.module_from_spec(spec)
spec.loader.exec_module(watch)


def test_completed_child_preserves_exit_status(tmp_path):
    rc=watch.main([sys.executable,'-c','raise SystemExit(2)'],report=tmp_path,poll_interval=.02)
    assert rc==2
    record=json.loads((tmp_path/'diagnostic_watchdog.json').read_text())
    assert record['interrupted'] is False


def test_log_heartbeat_does_not_hide_stalled_physics(tmp_path):
    child=tmp_path/'child.py'
    child.write_text('''import json, os, signal, sys, time
from pathlib import Path
signal.signal(signal.SIGUSR1, lambda *_: None)
signal.signal(signal.SIGINT, lambda *_: sys.exit(12))
now=time.time()
Path(sys.argv[1]).write_text(json.dumps(dict(started=now,pid=os.getpid(),last_physics_step=now)))
while True:
    Path(sys.argv[2]).write_text(str(time.time()))
    time.sleep(.01)
''')
    rc=watch.main([sys.executable,str(child),str(tmp_path/'diagnostic_live_progress.json'),str(tmp_path/'heartbeat.log')],
        report=tmp_path,stall_timeout=.15,startup_timeout=5,grace=2,poll_interval=.02)
    assert rc==12
    record=json.loads((tmp_path/'diagnostic_watchdog.json').read_text())
    assert record['interrupted'] is True
    assert 'No completed physics step' in record['stall_record']['reason']
