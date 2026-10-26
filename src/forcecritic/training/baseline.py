"""P1 driver: accepted demonstrations -> ACT -> measured validation -> real TD."""
from pathlib import Path
import json
import numpy as np
import torch
from ..runtime import atomic_json, digest
from ..data.demonstrations import Demonstrations
from .pretrain import baseline_actor, pretrain
from .live import LivePolicy, evaluate, collect_shared, fit_online_critic


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


def baseline(run, manifest, make_adapter, *, device='cuda:0'):
    limits = accepted_limits(run.report / 'p0_acceptance.json')
    dataset = Demonstrations(manifest, episode_limit=200)
    if len(dataset.rows) < 200:
        raise ValueError('P1 requires 200 accepted successful training episodes; collect more before pretraining')
    checkpoint = Path('checkpoints') / run.run_id / 'act0.pt'
    actor = baseline_actor(run.config['seed']).to(device)
    optimizer = torch.optim.AdamW([p for p in actor.parameters() if p.requires_grad], lr=1e-4)
    pretrain(actor, dataset, optimizer, checkpoint, config_hash=run.config_hash, device=device,
             updates=run.config['baseline']['act_pretrain_updates'],
             batch_size=run.config['algorithm']['batch_size'], seed=run.config['seed'])
    policy = LivePolicy(actor, dataset.normalizer, device)
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
    adapter, ledger = make_adapter(8, limits)
    try:
        bundle = collect_shared(adapter, policy, actor, ledger, policy_sha=digest(checkpoint),
            episode_cap=run.config['baseline']['shared_collection_episode_cap'],
            transition_cap=run.config['baseline']['shared_collection_transition_cap'])
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
        fitted = fit_online_critic(bundle, policy, updates=run.config['baseline']['critic_initial_updates'],
            batch_size=run.config['algorithm']['batch_size'])
        torch.save(fitted, checkpoint.parent / 'critic0.pt')
        atomic_json(run.report / 'baseline_training.json', dict(status='PASSED',
            real_blocks=len(bundle['rows']), real_complete_episodes=bundle['complete_episodes'],
            critic_updates=len(fitted['losses']), return_scale=fitted['scale'].tolist(),
            replay_sha256=digest(replay_path), policy_sha256=digest(checkpoint), accounting=ledger.record()))
    finally:
        adapter.close()
        ledger.close()
    return {'status':'PASSED', 'reason':'ACT0 has measured insertions; ordinary critic fitted on actual chunk transitions'}
