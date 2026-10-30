"""Bounded background development checks for renderer/input repeatability."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import argparse
import numpy as np
import torch
from forcecritic.diagnostics.artifacts import atomic_json
from forcecritic.diagnostics.live import compare
from forcecritic.runtime import Run,digest

run_id='20261023T104759Z'
parser=argparse.ArgumentParser()
parser.add_argument('--reset-render-history',action='store_true')
parser.add_argument('--spatial-render',action='store_true')
parser.add_argument('--fixed-shadow-seed',action='store_true')
args=parser.parse_args()
report=Path('reports')/run_id
tag='render_repair_'+str(time.time_ns())
root=report/tag
root.mkdir()
launcher=json.loads(Path('configs/isaac_launcher.json').read_text())
accounted=[]
result=dict(status='RUNNING',purpose='Development-only engineering repair; unchanged scientific gates',attempts=[],comparisons=[])
def save():atomic_json(root/'result.json',result)
def execute(name,mode,full=False,reference=None):
    output=root/(name+'.json')
    cmd=[sys.executable,'-u','scripts/probe_diagnostic_replay.py','--output',str(output),
         '--headless','--enable_cameras','--kit_args='+launcher['kit_args'],
         '--render-aa',mode,'--fresh-full' if full else '--only-prefix']
    if reference:cmd+=['--reference',str(reference)]
    if args.reset_render_history:cmd+=['--reset-render-history']
    if args.spatial_render:cmd+=['--spatial-render']
    if args.fixed_shadow_seed:cmd+=['--fixed-shadow-seed']
    started=time.time()
    with output.with_suffix('.log').open('w') as f:
        p=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            if time.time()-started>300:
                import signal
                os.killpg(p.pid,signal.SIGKILL);p.wait()
                break
            time.sleep(2)
    measured=json.loads(output.read_text()) if output.exists() else {}
    result['attempts'].append(dict(name=name,returncode=p.returncode,command=cmd,output=str(output),elapsed=time.time()-started))
    save()
    if p.returncode or measured.get('status')!='MEASURED':
        from forcecritic.diagnostics.recovery import recover_accounting
        live_path=output.parent/output.stem/'diagnostic_live_progress.json'
        if live_path.exists():
            recovery=recover_accounting(json.loads(live_path.read_text()))
            recovered_path=output.with_suffix('.recovered.json')
            atomic_json(recovered_path,recovery)
            accounted.append((recovered_path,recovery))
        raise RuntimeError(f'{name}: exit {p.returncode}, outcome {measured.get("status")}')
    accounted.append((output,measured['accounting']))
    return output,measured
try:
    save()
    for mode in (('FXAA',) if args.spatial_render else ('DLSS',) if args.reset_render_history else ('FXAA','Off')):
        a,am=execute(mode+'_input_a',mode)
        b,bm=execute(mode+'_input_b',mode)
        x=torch.load(am['raw_inputs'],weights_only=False);y=torch.load(bm['raw_inputs'],weights_only=False)
        error={k:float(np.max(abs(v.astype(float)-y['observation'][0][k].astype(float)))) for k,v in x['observation'][0].items()}
        action_error=float(np.max(abs(x['plan']-y['plan'])))
        result['comparisons'].append(dict(mode=mode,input_max_errors=error,action_max_error=action_error))
        save()
        if all(v==0 for v in error.values()) and action_error==0:
            a,am=execute(mode+'_full_a',mode,True)
            b,bm=execute(mode+'_full_b',mode,True,reference=a)
            x=torch.load(am['raw_trajectory'],weights_only=False);y=torch.load(bm['raw_trajectory'],weights_only=False)
            check=compare(x,y,json.loads((report/'protocol_effect_thresholds.json').read_text())['fidelity_tolerances'],
                          json.loads((report/'baseline_training.json').read_text())['return_scale'])
            result['comparisons'].append(dict(mode=mode,fixed_prefix_and_first_action=True,full_remaining_horizon=check))
            result['status']='ENGINEERING_WITNESS_PASSED' if check['passed'] else 'ENGINEERING_FIDELITY_FAILED'
            result['selected_mode']=mode
            break
    else:result['status']='ENGINEERING_INPUTS_DIFFER'
except BaseException as e:
    result.update(status='FAILED_IMPLEMENTATION',error=repr(e))
    raise
finally:
    # Every completed probe is charged once, separate from scientific outcomes.
    run=Run('configs/minimal_validation.yaml',run_id)
    repair_path=report/'diagnostic_repair.json'
    repair=json.loads(repair_path.read_text())
    for path,account in accounted:
        for k,v in account['individual_control_step_equivalents'].items():run.manifest['steps'][k]+=v
        repair['engineering_control_steps']+=sum(account['individual_control_step_equivalents'].values())
        if '_full_' in path.name:repair['previous_continuations']+=1
    result['accounting']=[dict(path=str(p),sha256=digest(p),accounting=a) for p,a in accounted]
    atomic_json(repair_path,repair);run.save();save()
    atomic_json(report/'render_repair_latest.json',dict(result=str(root/'result.json'),status=result['status']))
    print(json.dumps(result),flush=True)
