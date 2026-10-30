"""Reconcile real interrupted trajectories and preserve the actual D0 no-go."""
import csv
import json
from pathlib import Path
import torch
from forcecritic.runtime import Run,digest
from forcecritic.diagnostics.artifacts import atomic_json
from forcecritic.diagnostics.live import compare
from forcecritic.diagnostics.recovery import recover_accounting

run=Run('configs/minimal_validation.yaml','20261023T104759Z')
final=run.report/'diagnostic_repair_result.json'
if final.exists():raise SystemExit('Already finalized; do not double-count')
live_path=run.report/'diagnostic_live_progress.json'
live=json.loads(live_path.read_text())
atomic_json(run.report/'diagnostic_native_crash_progress.json',live)
directory=Path(live['last_trajectory']).parent
paths=sorted(directory.glob('trajectory_*.pt'))
if len(paths)!=7:raise ValueError('Recovery expects one original, three R and three P saved trajectories')
saved=[torch.load(p,weights_only=False,map_location='cpu') for p in paths]
if [(s['context']['candidate'],s['context']['method']) for s in saved] != [(0,'original')]+[(0,'snapshot_restore')]*3+[(0,'prefix_replay')]*3:
    raise ValueError('Saved trajectories are not the intended paired candidate')
thresholds=json.loads((run.report/'protocol_effect_thresholds.json').read_text())['fidelity_tolerances']
scale=json.loads((run.report/'baseline_training.json').read_text())['return_scale']
comparisons=[]
for path,row in zip(paths[1:],saved[1:]):
    comparisons.append(dict(method=row['context']['method'],repeat=row['context']['repeat'],candidate=0,
        **compare(saved[0]['trajectory'],row['trajectory'],thresholds,scale),
        source=str(path),sha256=digest(path)))
accepted=[m for m in ('snapshot_restore','prefix_replay') if all(c['passed'] for c in comparisons if c['method']==m)]
if accepted:raise ValueError('Partial data cannot establish full positive admission')
account=recover_accounting(live)
atomic_json(run.report/'diagnostic_native_recovery.json',dict(accounting=account,
    producer_pid=live['pid'],outcome_accounting='diagnostic_1793004526346629461_outcome.accounting.json',
    reason='Native exit -11 bypassed finally. The original zero aggregate was missing evidence, not zero work.'))
for k,v in account['individual_control_step_equivalents'].items():run.manifest['steps'][k]+=v
run.manifest.setdefault('accounting_recoveries',[]).append(str(run.report/'diagnostic_native_recovery.json'))
repair_path=run.report/'diagnostic_repair.json'
repair=json.loads(repair_path.read_text())
repair['native_control_steps']=sum(account['individual_control_step_equivalents'].values())
repair['previous_continuations']+=8 # Seven complete and one potentially incomplete continuation reserved.
repair['native_incomplete_continuations_charged']=1
probe_records=[]
for name in ('fresh_a','fresh_b','inputs_a','inputs_b'):
    path=run.report/f'replay_probe_{name}.json'
    record=json.loads(path.read_text())
    if record['status']!='MEASURED':raise ValueError('Engineering evidence incomplete')
    counts=record['accounting']['individual_control_step_equivalents']
    for k,v in counts.items():run.manifest['steps'][k]+=v
    repair['engineering_control_steps']+=sum(counts.values())
    if name.startswith('fresh_'):repair['previous_continuations']+=1
    probe_records.append(dict(path=str(path),sha256=digest(path),accounting=record['accounting']))
repair['additional_engineering_probes']=probe_records
repair['status']='NO_GO_REPLAY'
atomic_json(repair_path,repair)
summary=dict(status='NO_GO_REPLAY',methods=[],continuations_used=7,
    measured_parent_ids=['diagnostic-40000'],parents_fully_completed=0,parents_with_failed_admission=1,
    planned_parents=8,candidates_measured=1,candidates_planned=3,
    comparisons=comparisons,thresholds_sha256=digest(run.report/'protocol_effect_thresholds.json'),
    raw_original=dict(path=str(paths[0]),sha256=digest(paths[0])),
    native_failure_preserved='diagnostic_1793004526346629461_outcome.json',
    accounting=account,
    reason='Saved real original and three repeats of each replay method all violate locked tolerances for candidate A0. This already falsifies all-comparisons admission. Other candidates/parents and D1/D2/B1-B3 remain NOT_RUN; the native crash is a separate implementation failure, not a scientific result.',
    scope='Reanalysis of real saved trajectories, not new simulator execution or CPU-gradient validation')
atomic_json(final,summary)
atomic_json(run.report/'d0_initial.json',{k:v for k,v in summary.items() if k not in ('comparisons','accounting')})
policy_sha=digest(Path('checkpoints')/run.run_id/'act0.pt')
path=run.result/'fidelity.csv'
fields=path.read_text().splitlines()[0].split(',')
with path.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for c in comparisons:
        e=c['errors']
        writer.writerow(dict(parent_episode='diagnostic-40000',split='admission',policy_sha=policy_sha,
            restore_method=c['method'],action_variant=0,repeat=c['repeat'],pose_error=e['position_m'],
            velocity_error=e['linear_velocity_m_s'],force_error=e['force_N'],return_error=e['scaled_return'],
            cost_error=e['scaled_cost'],success_difference=e['success'],env_steps='',status='FAILED'))
run.manifest.setdefault('accounting_uncertainties',[]).append(dict(
    source=str(run.report/'diagnostic_native_recovery.json'),known_lower_bound=account['individual_control_step_equivalents'],
    upper_bound=None,reason='No per-step journal existed before this native crash. Last live sample recovered; unflushed steps unknown. Future formal diagnostic retry blocked until reconciled.'))
run.record('diagnostic','NO_GO_REPLAY',summary['reason'],2,[final,run.report/'d0_initial.json',path,
    run.report/'diagnostic_native_recovery.json',run.report/'replay_input_audit.json',
    run.report/'diagnostic_1793004526346629461_outcome.json'])
atomic_json(run.report/'decision.json',dict(status='NO_GO_CALIBRATION',from_stage='diagnostic',
    reason='D0 replay admission failed on saved real comparisons; D1/D2/short ablations not authorized by the scientific gate.',
    evidence_hashes={str(final):digest(final)},config_hash=run.config_hash))
print(json.dumps(dict(replay_comparisons=len(comparisons),passed=sum(c['passed'] for c in comparisons),
    stage=run.manifest['stages']['diagnostic']['status'],baseline_verified=run.verified('baseline'),
    native_accounting=account['status'])))
