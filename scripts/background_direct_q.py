"""Bounded, supervised ordinary direct-Q evaluation after a replay no-go."""
import csv
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from forcecritic.runtime import Run,atomic_json,digest
from forcecritic.diagnostics.recovery import recover_accounting
from background_training import publish

RUN='20261023T104759Z'
REPORT=Path('reports')/RUN
RESULT=Path('results')/RUN
PYTHON=sys.executable

def read(path):
    try:return json.loads(path.read_text())
    except (OSError,ValueError):return {}

def job(status,phase,**kw):
    atomic_json(REPORT/'direct_q_job.json',dict(status=status,phase=phase,pid=os.getpid(),
        updated=time.time(),log='.cache/direct_q_background.log',**kw))
    print('DIRECT_Q_JOB',status,phase,kw,flush=True)

def evaluate_arm(arm):
    log=REPORT/f'{arm}_evaluation.log'
    command=[PYTHON,'-u','scripts/evaluate_direct_q.py','--arm',arm,'--headless','--enable_cameras',
        '--kit_args=--/exts/omni.kit.registry.nucleus/registries/0/url=https://ovextensionsprod.blob.core.windows.net/exts/kit/prod/107/shared']
    started=time.time()
    progress_path=REPORT/arm/'diagnostic_live_progress.json'
    with log.open('x') as stream:
        child=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        job('RUNNING',arm,child_pid=child.pid,stage_log=str(log),started=started)
        interrupted=None
        while child.poll() is None:
            progress=read(progress_path)
            last=progress.get('last_physics_step') if progress.get('started',0)>=started else None
            age=time.time()-(last or started)
            timeout=180 if last else 300
            if interrupted is None and age>timeout:
                interrupted=time.time()
                job('STALLED',arm,child_pid=child.pid,reason=f'No completed physics step for {age:.1f}s',stage_log=str(log))
                if last:
                    try:os.kill(child.pid,signal.SIGUSR1)
                    except ProcessLookupError:pass
                try:os.killpg(child.pid,signal.SIGINT)
                except ProcessLookupError:pass
            if interrupted and time.time()-interrupted>30:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                break
            time.sleep(2)
        code=child.wait()
    record=read(REPORT/f'{arm}_evaluation.json')
    counts=record.get('accounting')
    if not counts:
        progress=read(progress_path)
        if progress.get('started',0)>=started:
            counts=recover_accounting(progress)
    outcome=dict(arm=arm,returncode=code,started=started,finished=time.time(),log=str(log),
        command=command,accounting=counts,status=record.get('status','FAILED_IMPLEMENTATION'))
    atomic_json(REPORT/f'{arm}_supervisor.json',outcome)
    run=Run('configs/minimal_validation.yaml',RUN)
    for key,value in (counts or {}).get('individual_control_step_equivalents',{}).items():
        run.manifest['steps'][key]+=value
    run.manifest.setdefault('ordinary_direct_q_evaluations',[]).append(outcome)
    run.save()
    if code or record.get('status')!='COMPLETED' or record.get('episodes')!=32:
        raise RuntimeError(f'{arm}: incomplete real evaluation, exit={code}; see {log}')

def summarize():
    arms={}
    for arm in ('direct_B0','direct_Q'):
        rows=list(csv.DictReader((RESULT/f'{arm}_validation.csv').open()))
        if len(rows)!=32 or any(r['source']!='real_forge' for r in rows):raise ValueError('Expected 32 actual robot episodes')
        n=len(rows);k=sum(int(r['stable_success']) for r in rows);p=k/n;z=1.959963984540054
        center=(p+z*z/(2*n))/(1+z*z/n)
        radius=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
        arms[arm]=dict(episodes=n,successes=k,success_rate=p,wilson95=[center-radius,center+radius],
            mean_cost=sum(float(r['cost']) for r in rows)/n,
            mean_return=sum(float(r['return']) for r in rows)/n,
            unsafe_episodes=sum(r['termination_reason']=='unsafe' for r in rows),
            policy_sha=rows[0]['policy_sha'],csv_sha=digest(RESULT/f'{arm}_validation.csv'))
    result=dict(status='COMPLETED',scope='Ordinary direct-Q fallback; formal paired D1/D2/B1-B3 remain NOT_RUN',
        arms=arms,delta_success_rate=arms['direct_Q']['success_rate']-arms['direct_B0']['success_rate'],
        interpretation='Real batch observations only; additional BC is a confound for attributing differences to Q alone. No paired-calibration claim.',
        plan_sha=digest(REPORT/'direct_q_plan.json'),finished=time.time())
    atomic_json(REPORT/'direct_q_result.json',result)
    with (RESULT/'direct_q_summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['arm','episodes','successes','success_rate','mean_cost','mean_return','unsafe_episodes'])
        writer.writeheader()
        for name,values in arms.items():writer.writerow(dict(arm=name,**{k:values[k] for k in writer.fieldnames if k!='arm'}))

def main():
    lock=Path('.cache/direct_q.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if read(REPORT/'direct_q_training.json').get('status')!='TRAINED_NOT_EVALUATED':
        raise RuntimeError('Finish the locked training attempt before launching evaluation')
    code=0
    try:
        for arm in ('direct_B0','direct_Q'):
            if (REPORT/f'{arm}_evaluation.json').exists() or (RESULT/f'{arm}_validation.csv').exists():
                raise FileExistsError('Existing evaluation attempt; inspect it rather than silently repeating')
            evaluate_arm(arm)
        summarize()
        job('COMPLETED','real_evaluations_complete')
    except BaseException as error:
        code=1
        job('FAILED_IMPLEMENTATION','stopped',error=repr(error))
        raise
    finally:
        subprocess.run([PYTHON,'scripts/write_repair_report.py'],check=True)
        publication=publish(refresh_report=False)
        if publication:print('PUBLICATION_FAILED; evidence remains local',flush=True)
    return code

if __name__=='__main__':sys.exit(main())
