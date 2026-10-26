"""Supervise actual simulator progress, not monitor/log heartbeats."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from forcecritic.runtime import Run, atomic_json


def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}


def main(command, *, startup_timeout=300, stall_timeout=180, grace=30, report=None, poll_interval=2):
    report=Path(report or 'reports/20261023T104759Z')
    path=report/'diagnostic_live_progress.json'
    started=time.time()
    child=subprocess.Popen(command,start_new_session=True)
    interrupted=False
    deadline=None
    while child.poll() is None:
        current=read(path)
        fresh=current.get('started',0)>=started
        last=current.get('last_physics_step') if fresh else None
        age=time.time()-(last or started)
        timeout=stall_timeout if last else startup_timeout
        if not interrupted and age>timeout:
            interrupted=True
            deadline=time.time()+grace
            record=dict(status='STALLED',time=time.time(),cli_pid=child.pid,
                        producer_pid=current.get('pid') if fresh else None,
                        reason=f'No completed physics step for {age:.1f}s',last_progress=current,
                        action='Dump stack, interrupt simulator, allow bounded accounting flush; no blind retry')
            atomic_json(report/'diagnostic_watchdog.json',record)
            print('WATCHDOG',record['reason'],flush=True)
            producer=record['producer_pid']
            if producer:
                try:
                    os.kill(producer,signal.SIGUSR1)
                    os.kill(producer,signal.SIGINT)
                except ProcessLookupError:pass
            else:os.killpg(child.pid,signal.SIGINT)
        if interrupted and time.time()>deadline:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait()
            run=Run('configs/minimal_validation.yaml','20261023T104759Z')
            # The killed CLI could not reconcile its outcome. Preserve the live
            # lower-bound counts once; unknown unflushed steps stay explicit.
            counts=current.get('accounting',{}).get('individual_control_step_equivalents',{}) if fresh else {}
            for k,v in counts.items():run.manifest['steps'][k]+=v
            run.manifest.setdefault('accounting_uncertainties',[]).append(dict(
                source='diagnostic_watchdog',started=started,
                measured_lower_bound=counts,unflushed_steps='UNKNOWN; do not treat as zero',
                retry_requires_budget_reconciliation=True))
            run.record('diagnostic','FAILED_IMPLEMENTATION','Watchdog: simulator made no physical progress; stack and lower-bound accounting retained',124,
                       [report/'diagnostic_watchdog.json',path] if fresh else [report/'diagnostic_watchdog.json'])
            return 124
        time.sleep(poll_interval)
    atomic_json(report/'diagnostic_watchdog.json',dict(status='FINISHED',
        cli_returncode=child.returncode,interrupted=interrupted,started=started,finished=time.time(),
        stall_record=record if interrupted else None))
    return child.returncode


if __name__=='__main__':
    sys.exit(main(sys.argv[1:]))
