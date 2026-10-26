"""Supervise only the preregistered matched comparison and one bounded extension."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from forcecritic.runtime import atomic_json, digest
from forcecritic.staged.common import locked_plan
from forcecritic.diagnostics.recovery import recover_accounting

REPORT=Path('reports/stage_v3_20261026')
PYTHON=sys.executable

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def job(status,phase,**extra):
    atomic_json(REPORT/'job.json',dict(status=status,phase=phase,pid=os.getpid(),updated=time.time(),**extra))
    print(status,phase,extra,flush=True)

def report():
    subprocess.run([PYTHON,'scripts/report_stage_v3.py'],check=True)
    return read(REPORT/'phase1_decision.json')

def evaluate(arm,batch):
    name=f'eval_{arm}_{batch}';log=REPORT/(name+'.log')
    supervisor=REPORT/(name+'.supervisor.json')
    prior=read(supervisor)
    if prior:
        artifact=read(REPORT/(name+'.json'))
        if prior.get('returncode')==0 and artifact.get('status')=='PASSED' and artifact.get('episodes')==32 and digest(Path(artifact['csv']))==artifact['csv_sha']:
            return
        raise RuntimeError(f'Existing incomplete attempt {name}; preserve and audit before retry')
    command=[PYTHON,'-u','scripts/evaluate_stage_v3.py','--arm',arm,'--batch',batch,
        '--headless','--enable_cameras',
        '--kit_args=--/exts/omni.kit.registry.nucleus/registries/0/url=https://ovextensionsprod.blob.core.windows.net/exts/kit/prod/107/shared']
    started=time.time();progress_path=REPORT/name/'diagnostic_live_progress.json'
    with log.open('x') as output:
        child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
        job('RUNNING',name,child_pid=child.pid,started=started,log=str(log))
        interrupted=None
        while child.poll() is None:
            progress=read(progress_path)
            last=progress.get('last_physics_step') if progress.get('started',0)>=started else None
            age=time.time()-(last or started)
            if interrupted is None and age>(180 if last else 300):
                interrupted=time.time()
                job('RUNNING',name,child_pid=child.pid,log=str(log),watchdog=f'No completed physics for {age:.1f}s; interrupting')
                if last:
                    try:os.kill(child.pid,signal.SIGUSR1)
                    except ProcessLookupError:pass
                try:os.killpg(child.pid,signal.SIGINT)
                except ProcessLookupError:pass
            if interrupted and time.time()-interrupted>30:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            time.sleep(2)
        code=child.wait()
    artifact=read(REPORT/(name+'.json'));counts=artifact.get('accounting')
    if not counts:
        progress=read(progress_path)
        if progress.get('started',0)>=started:counts=recover_accounting(progress)
    outcome=dict(command=command,returncode=code,started=started,finished=time.time(),log=str(log),
        status=artifact.get('status','FAILED_IMPLEMENTATION'),accounting=counts)
    atomic_json(supervisor,outcome)
    manifest=read(REPORT/'execution_manifest.json')
    manifest.setdefault('new_evaluations',{})[name]=outcome
    atomic_json(REPORT/'execution_manifest.json',manifest)
    if code or artifact.get('status')!='PASSED' or artifact.get('episodes')!=32:
        raise RuntimeError(f'{name} incomplete; exit={code}; inspect {log}')

def main():
    lock=Path('.cache/stage_v3.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    locked_plan()
    if read(REPORT/'A1_training.json').get('status')!='PASSED':raise RuntimeError('A1 training incomplete')
    if read(REPORT/'a2_reproduction_audit.json').get('status')!='PASSED':raise RuntimeError('A2 reproduction failed')
    if not (REPORT/'execution_manifest.json').exists():
        atomic_json(REPORT/'execution_manifest.json',dict(
            training=dict(command='env PYTHONPATH=src .venv/bin/python -u scripts/train_stage_v3_a1.py',
                returncode=None,returncode_note='Detached training launcher did not retain OS exit code; completion artifact and checkpoint audited',
                deployed_A1_updates=200,discarded_A2_reproduction_updates=1,new_environment_interactions=0),
            reused=dict(A2_actor_updates=200,A2_critic_updates=2000,A0_episodes=32,A2_episodes=32),new_evaluations={}))
    try:
        evaluate('A1','initial');decision=report()
        if decision['extension_allowed']:
            evaluate('A1','extension');evaluate('A2','extension');decision=report()
        job(decision['status'],'phase1_complete',decision=decision)
    except BaseException as error:
        job('FAILED_IMPLEMENTATION','stopped',error=repr(error));raise
    finally:
        report()
        publication=subprocess.run([PYTHON,'scripts/publish_stage_v3.py'])
        if publication.returncode:print('Publication failed; local evidence retained',flush=True)

if __name__=='__main__':main()
