"""Bounded same-version recollection, with whole collection-run split isolation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from forcecritic.runtime import atomic_json, digest

p=argparse.ArgumentParser()
p.add_argument('--run-id',required=True)
p.add_argument('--checkpoint',required=True)
p.add_argument('--output',default='data/act_demonstrations')
a=p.parse_args()
out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
acceptance=Path('reports')/a.run_id/'p0_acceptance.json'
if json.loads(acceptance.read_text())['status']!='PASSED':raise ValueError('P0 must pass first')
plan=[dict(index=i,seed=1100+i,split='train' if i<14 else 'validation' if i<17 else 'test',
           successful_target=16,attempt_cap=32) for i in range(20)]
lock={'collection_runs':plan,'split_unit':'collection_run','ratio':[.7,.15,.15],
      'development_excluded':['development_1000','development_1000_retry'],
      'acceptance_sha256':digest(acceptance),'expert_sha256':digest(a.checkpoint)}
lock_path=out/'collection_plan.json'
if lock_path.exists() and json.loads(lock_path.read_text())!=lock:raise ValueError('Collection plan changed')
atomic_json(lock_path,lock)
episodes=[];contract=None
for spec in plan:
    directory=out/f"{spec['index']:02d}_{spec['split']}_{spec['seed']}"
    path=directory/'manifest.json'
    if not path.exists():
        cmd=[sys.executable,'scripts/run_forge_probe.py','collect','--run-id',a.run_id,
             '--output',str(directory),'--checkpoint',a.checkpoint,'--acceptance',str(acceptance),
             '--num_envs','8','--episodes','16','--attempt-cap','32',
             '--seed',str(spec['seed']),'--split',spec['split']]
        rc=subprocess.run(cmd).returncode
        if rc:raise RuntimeError(f'Real collection command failed: run {spec}, exit {rc}; inspect immutable attempt')
    data=json.loads(path.read_text())
    if data['status']!='COLLECTED' or data['successful_episodes']<16:
        raise RuntimeError(f'BLOCKED_COLLECTION: {spec}; do not hide unsuccessful bounded attempts')
    if contract is None:contract=data['contract'].copy()
    elif data['contract']!=contract:raise ValueError('Mixed collection contracts')
    for row in data['episodes']:
        for entry in row['files'].values():entry['path']=os.path.relpath(directory/entry['path'],out)
        episodes.append(row)
    atomic_json(out/'progress.json',dict(completed_collection_runs=spec['index']+1,
        successful_episodes=sum(x['successful'] for x in episodes),all_episodes=len(episodes)))
    print(f"Completed {spec['index']+1}/20 collection runs",flush=True)
contract['acceptance_evidence']=os.path.relpath(acceptance,out)
atomic_json(out/'manifest.json',dict(status='COLLECTED',source='same_version_recollection',
    contract=contract,episodes=episodes,collection_plan_sha256=digest(lock_path)))
