"""P1 driver: accepted demonstrations -> ACT -> measured validation -> real TD."""
from pathlib import Path
import json
import csv
import time
import numpy as np
import torch
from ..runtime import atomic_json, digest
from ..data.demonstrations import Demonstrations
from .pretrain import baseline_actor, pretrain
from .live import LivePolicy, evaluate, collect_shared, fit_online_critic


def reused_validation(run, checkpoint):
    """Only an explicit, hash-bound recovery receipt may reuse prior robot data."""
    receipt_path=run.report/'baseline_validation_resume.json'
    if not receipt_path.exists():return None
    receipt=json.loads(receipt_path.read_text())
    if receipt['status']!='ACCEPTED_FOR_RESUME' or receipt['config_hash']!=run.config_hash:
        raise ValueError('invalid validation recovery receipt')
    if receipt['checkpoint_sha256']!=digest(checkpoint):
        raise ValueError('validation checkpoint changed')
    for path,sha in receipt['evidence_hashes'].items():
        if digest(path)!=sha:raise ValueError('validation recovery evidence changed: '+path)
    with (run.result/'act0_validation.csv').open() as stream:rows=list(csv.DictReader(stream))
    if len(rows)!=32 or [int(r['seed']) for r in rows]!=list(range(20000,20032)):
        raise ValueError('recovery requires all 32 locked initial conditions')
    if any(r['policy_sha']!=receipt['checkpoint_sha256'] or r['source']!='real_forge' for r in rows):
        raise ValueError('recovery requires measured episodes from this exact policy')
    for row in rows:row['stable_success']=int(row['stable_success'])
    print('RESUME: reusing 32 hash-verified ACT0 validation episodes; no new evaluation steps charged.',flush=True)
    return rows


def accepted_limits(path):
    acceptance = json.loads(Path(path).read_text())
    flags = ('action_alignment','sensor_alignment','camera_alignment','causal_observations')
    if acceptance.get('status') != 'PASSED' or any(acceptance.get(k) is not True for k in flags):
        raise ValueError('Measured P0 contracts have not passed')
    evidence = acceptance.get('evidence_hashes', {})
    if not evidence:
        raise ValueError('P0 acceptance must reference measured evidence')
    for filename, sha in evidence.items():
        if digest(filename) != sha:
            raise ValueError('P0 evidence changed')
    limits = np.asarray(acceptance['physical_limits'], dtype=float)
    if limits.shape != (6,) or not np.isfinite(limits).all() or (limits <= 0).any():
        raise ValueError('Six positive physical N/Nm limits must be locked before testing')
    if acceptance.get('physical_limits_source_split') not in ('train', 'development'):
        raise ValueError('Physical limits cannot come from test data')
    hard = np.asarray(acceptance['physical_protection_limits'], dtype=float)
    if hard.shape != (6,) or not np.isfinite(hard).all() or (hard < limits).any():
        raise ValueError('Independent physical hard protection must be locked before testing')
    return limits


def baseline(run, manifest, make_adapter, *, device='cuda:0', phase):
    if phase not in ('evaluation','shared'):
        raise ValueError('baseline phases require separate Isaac processes')
    def progress(phase, **values):
        record=dict(phase=phase,time=time.time(),**values)
        atomic_json(run.report/'baseline_live_progress.json',record)
        print('BASELINE_PROGRESS '+json.dumps(record),flush=True)
    limits = accepted_limits(run.report / 'p0_acceptance.json')
    dataset = Demonstrations(manifest, episode_limit=200)
    if len(dataset.rows) < 200:
        raise ValueError('P1 requires 200 accepted successful training episodes; collect more before pretraining')
    checkpoint = Path('checkpoints') / run.run_id / 'act0.pt'
    actor = baseline_actor(run.config['seed']).to(device)
    optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=1e-4)
    progress('act_pretraining')
    pretrain(actor, dataset, optimizer, checkpoint, config_hash=run.config_hash, device=device,
             updates=run.config['baseline']['act_pretrain_updates'],
             batch_size=run.config['algorithm']['batch_size'], seed=run.config['seed'])
    policy = LivePolicy(actor, dataset.normalizer, device)
    results = reused_validation(run,checkpoint)
    if results is None:
        if phase=='shared':raise ValueError('shared phase requires accepted prior validation receipt')
        progress('act_validation')
        adapter, ledger = make_adapter(1, limits)
        try:
            results = evaluate(adapter, policy, range(20000,20032), ledger,
                run.result / 'act0_validation.csv', arm='B0', policy_sha=digest(checkpoint))
            from ..evaluation.export import export_episodes
            export_episodes(run,{'B0':results},'baseline_32_validation',[run.result/'act0_validation.csv'])
            atomic_json(run.report / 'baseline_validation.json', dict(
                status='PASSED' if any(r['stable_success'] for r in results) else 'BLOCKED_BASELINE',
                successful_starts=sum(r['stable_success'] for r in results), independent_starts=len(results),
                checkpoint_sha256=digest(checkpoint), accounting=ledger.record()))
        finally:
            adapter.close()
            ledger.close()
    if not any(r['stable_success'] for r in results):
        # A failed 10k baseline triggers interface review, not a fake calibration failure.
        return {'status':'BLOCKED_BASELINE', 'reason':'No stable insertions in 32 real validation starts; interface audit required before the 20k cap'}
    if phase=='evaluation':
        receipt=run.report/'baseline_validation_resume.json'
        if not receipt.exists():
            evidence=[run.report/'baseline_validation.json',run.result/'act0_validation.csv',
                      run.report/'episode_cohort.json',run.report/'p0_acceptance.json']
            atomic_json(receipt,dict(status='ACCEPTED_FOR_RESUME',config_hash=run.config_hash,
                checkpoint_sha256=digest(checkpoint),evidence_hashes={str(p):digest(p) for p in evidence},
                reason='Completed real evaluation; shared collection must start in a fresh Isaac process'))
        return {'status':'PASSED','reason':'ACT pretraining and 32 real validation starts complete; Critic not yet fitted'}
    progress('shared_environment_initialization',num_envs=8)
    adapter, ledger = make_adapter(8, limits)
    progress('shared_collection',completed_episodes=0,episode_cap=100,individual_transitions=0)
    try:
        bundle = collect_shared(adapter, policy, actor, ledger, policy_sha=digest(checkpoint),
            episode_cap=run.config['baseline']['shared_collection_episode_cap'],
            transition_cap=run.config['baseline']['shared_collection_transition_cap'],
            progress=lambda **v:progress('shared_collection',**v))
        replay_path = checkpoint.parent / 'shared_replay.pt'
        torch.save(bundle, replay_path)
        terminal_rows = [r for r in bundle['rows'] if not r['bootstrap']]
        success_count = sum(r['success'] for r in terminal_rows)
        failure_count = len(terminal_rows)-success_count
        if not success_count or not failure_count:
            atomic_json(run.report/'baseline_training.json',dict(status='INCONCLUSIVE',
                real_blocks=len(bundle['rows']),successful_attempts=success_count,failed_attempts=failure_count,
                critic_updates=0,replay_sha256=digest(replay_path),accounting=ledger.record()))
            return {'status':'INCONCLUSIVE','reason':'Shared real attempts did not span both success and failure; critic calibration not admitted'}
        progress('critic_training',completed_updates=0,requested_updates=run.config['baseline']['critic_initial_updates'])
        fitted = fit_online_critic(bundle, policy, updates=run.config['baseline']['critic_initial_updates'],
            batch_size=run.config['algorithm']['batch_size'],progress=lambda **v:progress('critic_training',**v))
        torch.save(fitted, checkpoint.parent / 'critic0.pt')
        atomic_json(run.report / 'baseline_training.json', dict(status='PASSED',
            real_blocks=len(bundle['rows']), real_complete_episodes=bundle['complete_episodes'],
            critic_updates=len(fitted['losses']), return_scale=fitted['scale'].tolist(),
            replay_sha256=digest(replay_path), policy_sha256=digest(checkpoint), accounting=ledger.record()))
    finally:
        adapter.close()
        ledger.close()
    return {'status':'PASSED', 'reason':'ACT0 has measured insertions; ordinary critic fitted on actual chunk transitions'}
