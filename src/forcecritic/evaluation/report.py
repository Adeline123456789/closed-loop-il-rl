"""Reports derive exclusively from manifests/raw CSV, never expected metrics."""
import csv
import json
from pathlib import Path
from forcecritic.runtime import atomic_json,digest
from .metrics import summary_rows,paired_comparisons

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
    episodes=list(csv.DictReader((run.result/'episode_metrics.csv').open()))
    summary=summary_rows(episodes)
    comparisons=paired_comparisons(episodes)
    has_episodes=any(r['status']=='COMPLETED' for r in summary)
    (run.result/'figures'/'README.md').write_text(
        'Episode summaries and parent-paired comparisons are in CSV; no curve is inferred from optimizer losses.\n' if has_episodes
        else 'No robot performance plot: no accepted robot episodes. Empty CSV files contain schemas only.\n')
    comparison_path=run.result/'paired_comparisons.csv'
    with comparison_path.open('w') as f:
        fields=list(comparisons[0]) if comparisons else ['comparison','seed','split','status','paired_parents','success_difference_pp','success_ci_low_pp','success_ci_high_pp','overload_difference_pp','reason']
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(comparisons)
    with (run.result/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=summary[0],lineterminator='\n');writer.writeheader();writer.writerows(summary)
    stages=run.manifest['stages']
    unpassed=[name for name,event in stages.items() if event['status'] not in ('PASSED','COMPLETED')]
    decision_path=run.report/'decision.json'
    decision=json.loads(decision_path.read_text()) if decision_path.exists() else {'status':'NOT_RUN','reason':'No continuation decision recorded'}
    checkpoint_count=len(run.manifest.get('checkpoints',[]))
    fidelity_count=sum(1 for _ in csv.DictReader((run.result/'fidelity.csv').open()))
    if has_episodes:
        from .plots import plot_summary
        plot_summary(summary,run.result/'figures'/'episode_summary.png')
    lines=['# Measured validation / stop report', '',f'Protocol: `{run.config["protocol"]}`. Run: `{run.run_id}`.',
           '', '## Outcome', '',
           ('Raw completed episodes are summarized separately by seed, split and checkpoint; no pooling across policy versions. See paired_comparisons.csv for matched-parent differences.' if has_episodes else 'P0 has not passed. No ACT pretraining or robot-method experiment is claimed. The software gradient path is tested separately; it is not evidence of improved insertion.'), '',
           '| Stage | Status | Reason |','|---|---|---|']
    lines += [f'| {s} | {e["status"]} | {e.get("reason", "Predecessor gate not passed")} |' for s,e in stages.items()]
    lines += ['', '## Questions', '',
              '1. Critic → ACT: implemented in `training/updates.py`; CPU tests change critic slope on fixed ACT/data, observe changed ACT output, and verify nonzero Q-only gradients with critic parameters fixed.',
              ('2. Ordinary direct update: see B1-B0 matched-parent comparison; causal/continuation decisions require all diagnostic gates.' if has_episodes else '2. Ordinary direct update improves insertion: **insufficient evidence**, B0/B1 not run.'),
              f'3. Replay fidelity rows: {fidelity_count}; diagnostic stage status: {stages["diagnostic"]["status"]}. Row count alone is not evidence of equivalence.',
              ('4. B2 vs B1 filtering: see matched-parent comparison and independent diagnostic evidence.' if has_episodes else '4. B2 vs B1 filtering: **insufficient evidence**, NOT_RUN.'),
              ('5. B3 vs B2 calibration: see matched-parent comparison; endpoint labels already contain contrast information.' if has_episodes else '5. B3 vs B2 calibration: **insufficient evidence**, NOT_RUN. Endpoint labels already contain contrast information.'),
              f'6. Recorded continuation decision: **{decision["status"]}** — {decision.get("reason","See decision.json")}.',
              '7. Stages not passed/completed: '+', '.join(unpassed)+'. Reasons are listed above; an environment block is not an algorithm negative result.', '',
              '## Raw result summary', '', '| Arm / seed / split / policy | Status | Episodes | Success | Overload |','|---|---|---:|---|---|']
    lines += [f'| {r["arm"]} / {r["seed"]} / {r["split"]} / {r["policy_sha"][:7]} | {r["status"]} | {r["episodes"]} | {r["stable_success_rate"]} | {r["overload_rate"]} |' for r in summary]
    lines += ['', '## Cost and artifacts', '',f'Individual simulator transitions by category: `{json.dumps(run.manifest["steps"])}`.',
              f'Total accounted simulator transitions: {sum(run.manifest["steps"].values())}. CPU unit fixtures are excluded from robot metrics.',
              f'Checkpoint index entries: {checkpoint_count}. Publication is not implied by a local checkpoint; only verified Release entries may be used as download links.',
              'Header-only fidelity/pair/actor/episode CSV files mean NOT_RUN, not zero success.',
              '', '## Reproduce / resume', '', '```bash',
              'FORCECRITIC_PYTHON=.venv-cpu/bin/python bash scripts/run_end_to_end.sh --config configs/minimal_validation.yaml --resume auto --publish',
              '```', '', 'This resumes audit artifacts, rechecks blocked prerequisites, and reports/publishes even when a gate stops training.',
              'Use the audited Isaac Python instead once installed. Installation alone does not resolve collection semantics.',
              '', '## Scope still unverified', '',
              'Consult the stage table and implementation_audit.md for unresolved interfaces and unimplemented orchestration; CPU software tests never establish robot efficacy.',
              'Stage execution deliberately fails closed while P0 is incomplete. Higher-stage robot orchestration is not presented as experimentally validated.',
              'See `data_audit.json`, `doctor.json`, `action_contract.md`, `sensor_contract.md`, `implementation_audit.md`, and command logs for exact evidence.']
    (run.report/'FINAL_REPORT.md').write_text('\n'.join(lines)+'\n')
    artifacts={str(p):{'sha256':digest(p),'bytes':p.stat().st_size} for folder in (run.report,run.result)
               for p in sorted(folder.rglob('*')) if p.is_file()}
    atomic_json('artifacts_manifest.json',{'run_id':run.run_id,'artifacts':artifacts,'weights':[]})
    return run.report/'FINAL_REPORT.md'
