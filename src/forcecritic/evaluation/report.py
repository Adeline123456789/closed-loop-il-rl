"""Reports derive exclusively from manifests/raw CSV, never expected metrics."""
import csv
import json
from pathlib import Path
from forcecritic.runtime import atomic_json,digest

SCHEMAS={
'fidelity.csv':'parent_episode,split,policy_sha,restore_method,action_variant,repeat,pose_error,velocity_error,force_error,return_error,cost_error,success_difference,env_steps,status',
'pairs.csv':'parent_episode,split,continuation_sha,restore_method,checkpoint_id,variant,action,direction,epsilon,physical_units,noise_id,return,cost,success,unsafe,prefix_steps,restore_steps,branch_steps,status',
'critic_diagnostics.csv':'parent_episode,split,policy_sha,arm,point_error,pair_error,predicted_sign,observed_sign,ci_low,ci_high,status',
'actor_step.csv':'parent_episode,split,policy_sha,arm,before_action,after_action,delta_action,q_grad_norm,bc_grad_norm,q_bc_dot,angle,saturation,real_return_change,real_cost_change,forgetting,status',
'episode_metrics.csv':'parent_episode,split,policy_sha,arm,seed,stable_success,ever_success,final_success,completion_time,force_peak,torque_peak,overload,termination_reason,env_steps,status',
'expert_smoke.csv':'episode,policy_sha,num_envs,stable_success,ever_success,final_success,env_steps,status'}


def initialize_csvs(run):
    for name,header in SCHEMAS.items():
        p=run.result/name
        if not p.exists():p.write_text(header+'\n')


def report(run):
    initialize_csvs(run)
    (run.result/'figures').mkdir(exist_ok=True)
    (run.result/'figures'/'README.md').write_text('No robot performance plot: no accepted robot episodes. Empty CSV files contain schemas only.\n')
    episodes=list(csv.DictReader((run.result/'episode_metrics.csv').open()))
    summary=[]
    for arm in ('B0','B1','B2','B3'):
        rows=[r for r in episodes if r['arm']==arm and r['status']=='COMPLETED']
        # All completed episodes, not only successful ones, enter load statistics.
        summary.append({'arm':arm,'status':'COMPLETED' if rows else 'NOT_RUN','episodes':len(rows),
                        'stable_success_rate':sum(float(r['stable_success']) for r in rows)/len(rows) if rows else 'NA',
                        'overload_rate':sum(float(r['overload']) for r in rows)/len(rows) if rows else 'NA'})
    with (run.result/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0]);writer.writeheader();writer.writerows(summary)
    stages=run.manifest['stages']
    lines=['# Measured validation / stop report', '',f'Protocol: `{run.config["protocol"]}`. Run: `{run.run_id}`.',
           '', '## Outcome', '',
           'P0 has not passed. No ACT pretraining or robot-method experiment is claimed. '
           'The software gradient path is tested separately; it is not evidence of improved insertion.', '',
           '| Stage | Status | Reason |','|---|---|---|']
    lines += [f'| {s} | {e["status"]} | {e.get("reason", "Predecessor gate not passed")} |' for s,e in stages.items()]
    lines += ['', '## Questions', '',
              '1. Critic → ACT: implemented in `training/updates.py`; CPU tests change critic slope on fixed ACT/data, observe changed ACT output, and verify nonzero Q-only gradients with critic parameters fixed.',
              '2. Ordinary direct update improves insertion: **insufficient evidence**, B0/B1 not run.',
              '3. Replay/pair fidelity: **NOT_RUN**; no same-state claim and no HF snapshot reconstruction.',
              '4. B2 vs B1 filtering: **insufficient evidence**, NOT_RUN.',
              '5. B3 vs B2 calibration: **insufficient evidence**, NOT_RUN. Endpoint labels already contain contrast information.',
              '6. Continue: repair P0 environment and measured sensor/action/camera contracts first. No GO_PILOT.',
              '7. P1–P4 skipped by the P0 gate, not an algorithm negative result.', '',
              '## Raw result summary', '', '| Arm | Status | Episodes | Success | Overload |','|---|---|---:|---|---|']
    lines += [f'| {r["arm"]} | {r["status"]} | {r["episodes"]} | {r["stable_success_rate"]} | {r["overload_rate"]} |' for r in summary]
    lines += ['', '## Cost and artifacts', '',f'Individual simulator transitions by category: `{json.dumps(run.manifest["steps"])}`.',
              f'Total accounted simulator transitions: {sum(run.manifest["steps"].values())}. CPU unit fixtures are excluded from robot metrics.',
              'No trained checkpoint or Release asset exists; weights: **未发布**. No fake download URL.',
              'Header-only fidelity/pair/actor/episode CSV files mean NOT_RUN, not zero success.',
              '', '## Reproduce / resume', '', '```bash',
              'FORCECRITIC_PYTHON=.venv-cpu/bin/python bash scripts/run_end_to_end.sh --config configs/minimal_validation.yaml --resume auto --publish',
              '```', '', 'This resumes audit artifacts, rechecks blocked prerequisites, and reports/publishes even when a gate stops training.',
              'Use the audited Isaac Python instead once installed. Installation alone does not resolve collection semantics.',
              '', '## Scope still unverified', '',
              'Real simulator integration, expert compatibility, six-axis sign/reference/baseline, camera alignment, physical safety limits, replay fidelity, and all robot training/evaluation remain unverified.',
              'Stage execution deliberately fails closed while P0 is incomplete. Higher-stage robot orchestration is not presented as experimentally validated.',
              'See `data_audit.json`, `doctor.json`, `action_contract.md`, `sensor_contract.md`, `implementation_audit.md`, and command logs for exact evidence.']
    (run.report/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    artifacts={str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for folder in (run.report,run.result)
               for p in sorted(folder.rglob('*')) if p.is_file()}
    atomic_json('artifacts_manifest.json',{'run_id':run.run_id,'artifacts':artifacts,'weights':[]})
    return run.report/'FINAL_REPORT.md'
