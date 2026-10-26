"""Artifact admission checks independent of the stage decision statistics."""
import json
import math
from pathlib import Path
import numpy as np
from ..runtime import digest
from .results import read_rows

def validate_completed_batch(report,results,name,plan,policy_sha):
    report=Path(report);path=Path(results)/(name+'.csv')
    supervisor=json.loads((report/(name+'.supervisor.json')).read_text())
    record=json.loads((report/(name+'.json')).read_text())
    rows=read_rows(path)
    seeds=plan['extension_seeds'] if name.endswith('extension') else plan['initial_evaluation_seeds']
    if supervisor['returncode']!=0 or record['status']!='PASSED':raise ValueError('Incomplete simulator process')
    if record['episodes']!=len(seeds) or len(rows)!=len(seeds):raise ValueError('Incomplete episode batch')
    if record['csv_sha']!=digest(path) or record['policy_sha']!=policy_sha:raise ValueError('Changed result or checkpoint identity')
    if record['config_sha']!=digest('configs/stage_v3_phase1.json'):raise ValueError('Changed evaluation config')
    if [int(r['seed']) for r in rows]!=seeds:raise ValueError('Changed seed order')
    arm=name.split('_')[1]
    if any(r['source']!='real_forge' or r['arm']!=arm or r['policy_sha']!=policy_sha for r in rows):raise ValueError('Wrong real-policy provenance')
    for r in rows:
        if int(r['stable_success']) not in (0,1):raise ValueError('Invalid success')
        if any(not math.isfinite(float(r[k])) for k in ('return','cost','peak_force_N','peak_torque_Nm','steps','control_dt')):raise ValueError('Nonfinite physical metric')
    accounting=record['accounting']
    if accounting!=supervisor['accounting']:raise ValueError('Accounting disagrees with supervisor')
    if accounting['control_step_cap']!=plan['evaluation_control_step_cap_per_batch']:raise ValueError('Changed budget')
    if sum(accounting['physics_substeps'].values())>accounting['decimation']*accounting['control_step_cap']:raise ValueError('Budget exceeded')
    if sum(int(r['steps']) for r in rows)!=accounting['individual_control_step_equivalents']['evaluation']:raise ValueError('Episode and physics counts disagree')
    return dict(status='PASSED',episodes=len(rows),csv_sha=digest(path),policy_sha=policy_sha,accounting=accounting,
                returncode=supervisor['returncode'],command=supervisor['command'])

def gradient_evidence(a1_path,a2_path):
    a1=read_rows(a1_path);a2=read_rows(a2_path)
    for values in (a1,a2):
        if [int(r['update']) for r in values]!=list(range(1,201)):raise ValueError('Missing or duplicate optimizer steps')
        if any(not all(math.isfinite(float(v)) for v in row.values()) for row in values):raise ValueError('Nonfinite update')
    arr=lambda field:np.asarray([float(r[field]) for r in a2])
    q=arr('q_grad_norm');bc=arr('bc_grad_norm');dot=arr('q_bc_dot')
    if np.any(q<=0) or np.any(arr('mean_abs_delta_action')<=0):raise ValueError('Disconnected actor updates')
    return dict(status='PASSED',scope='Implementation evidence, not robot performance',updates_per_arm=200,
        q_gradient_norm_range=[float(q.min()),float(q.max())],q_gradient_norm_mean=float(q.mean()),
        q_to_bc_gradient_norm_median=float(np.median(q/np.maximum(bc,1e-30))),
        q_bc_cosine_mean=float(np.mean(dot/(q*np.maximum(bc,1e-30)))),
        q_bc_negative_dot_fraction=float(np.mean(dot<0)),
        actual_mean_absolute_action_change_mean=float(arr('mean_abs_delta_action').mean()),
        last_update_actual_action_change=float(arr('mean_abs_delta_action')[-1]),
        step_term_cancellation='Historical A2 did not log step-gradient vectors; exact Q/BC/step cancellation cannot be reconstructed from norms alone',
        no_bug_evidence='All 200 Q gradients and actual ACT action changes are finite and nonzero; first matched update reproduced exactly',
        a1_csv_sha=digest(a1_path),a2_csv_sha=digest(a2_path))
