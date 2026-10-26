"""Check measured ordinary-control provenance before reporting robot outcomes."""
import csv
import json
import math
from pathlib import Path
from forcecritic.runtime import Run,digest,atomic_json

run=Run('configs/minimal_validation.yaml','20261023T104759Z')
report=run.report
plan=json.loads((report/'direct_q_plan.json').read_text())
plan_sha=digest(report/'direct_q_plan.json')
training=json.loads((report/'direct_q_training.json').read_text())
result=json.loads((report/'direct_q_result.json').read_text())
if not run.verified('baseline'):raise ValueError('Baseline evidence changed')
if training['plan_sha']!=plan_sha or result['plan_sha']!=plan_sha:raise ValueError('Plan mismatch')
if digest(training['checkpoint'])!=training['sha256']:raise ValueError('Trained checkpoint changed')
updates=list(csv.DictReader((run.result/'direct_Q_actor_updates.csv').open()))
if [int(r['update']) for r in updates]!=list(range(1,201)):raise ValueError('Missing/duplicate actor updates')
if any(not all(math.isfinite(float(v)) for v in r.values()) or float(r['q_grad_norm'])<=0 or float(r['mean_abs_delta_action'])<=0 for r in updates):
    raise ValueError('Nonfinite or disconnected actor update')
arms={}
for arm,sha in [('direct_B0',plan['act_sha']),('direct_Q',training['sha256'])]:
    path=run.result/f'{arm}_validation.csv'
    rows=list(csv.DictReader(path.open()))
    evaluation=json.loads((report/f'{arm}_evaluation.json').read_text())
    supervisor=json.loads((report/f'{arm}_supervisor.json').read_text())
    if [int(r['seed']) for r in rows]!=plan['evaluation_seeds']:raise ValueError('Changed, missing or repeated evaluation seeds')
    if any(r['policy_sha']!=sha or r['source']!='real_forge' or r['arm']!=arm for r in rows):raise ValueError('Policy/source mismatch')
    if any(not all(math.isfinite(float(r[k])) for k in ('return','cost','peak_force_N','peak_torque_Nm','steps')) for r in rows):raise ValueError('Nonfinite actual robot metrics')
    if evaluation['status']!='COMPLETED' or supervisor['returncode']!=0:raise ValueError('Incomplete simulator process')
    if evaluation['plan_sha']!=plan_sha or evaluation['csv_sha']!=digest(path):raise ValueError('Evaluation provenance mismatch')
    count=sum(int(r['stable_success']) for r in rows)
    if count!=result['arms'][arm]['successes'] or len(rows)!=result['arms'][arm]['episodes']:raise ValueError('Summary does not match real rows')
    accounting=evaluation['accounting']
    if sum(accounting['physics_substeps'].values())>accounting['control_step_cap']*accounting['decimation']:raise ValueError('Evaluation budget exceeded')
    arms[arm]=dict(episodes=len(rows),successes=count,policy_sha=sha,csv_sha=digest(path),accounting=accounting)
record=dict(status='PASSED',scope='Artifact consistency audit, not a significance or calibration claim',
    baseline_preserved=True,actor_updates=200,arms=arms,plan_sha=plan_sha,
    paired_diagnostic_status=run.manifest['stages']['diagnostic']['status'])
atomic_json(report/'direct_q_integrity.json',record)
print(json.dumps(record,indent=2))
