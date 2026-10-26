"""B1/B2/B3 same-data optimizer execution followed by real frozen evaluation."""
import copy
import time
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from ..runtime import atomic_json, digest
from ..data.demonstrations import Demonstrations
from ..models.critic import TwinCritic
from .baseline import accepted_limits
from .pretrain import baseline_actor
from .live import LivePolicy, replay_batch, evaluate
from .replay import ReturnScale
from .ablation import short_updates


class DeterministicBatches:
    """Lazily materialize matched batches without retaining thousands of RGB batches."""
    def __init__(self, length, builder, seed):
        self.length, self.builder, self.seed = length, builder, seed
    def __len__(self): return self.length
    def __getitem__(self, index): return self.builder(np.random.default_rng(self.seed + index))


def short_ablation(run, manifest, paired_bundle, make_adapter, device='cuda:0'):
    telemetry={'arms':{a:{'critic_completed':0,'critic_target':2000,'actor_completed':0,'actor_target':200}
                       for a in ('B1','B2','B3')}}
    def progress(**record):
        arm=record.get('arm');phase=record.get('phase')
        if arm in telemetry['arms'] and phase in ('critic','actor'):
            telemetry['arms'][arm][phase+'_completed']=record['completed_updates']
            telemetry['arms'][arm][phase+'_target']=record['requested_updates']
        telemetry.update(current=record,time=time.time())
        atomic_json(run.report/'short_live_progress.json',telemetry)
        print('SHORT_PROGRESS '+json.dumps(record),flush=True)
    progress(phase='prepare')
    limits = accepted_limits(run.report / 'p0_acceptance.json')
    directory = Path('checkpoints') / run.run_id
    act_path = directory / 'act0.pt'
    policy_sha = digest(act_path)
    # These are trusted local artifacts created by the live collector, not remote pickle inputs.
    act_state = torch.load(act_path, map_location='cpu', weights_only=False)
    replay = torch.load(directory / 'shared_replay.pt', map_location='cpu', weights_only=False)
    replay['rows'] = [r for r in replay['rows'] if r['split']=='train']
    if not replay['rows']:raise ValueError('No legal shared training parents')
    fitted = torch.load(directory / 'critic0.pt', map_location='cpu', weights_only=False)
    paired = torch.load(paired_bundle, map_location='cpu', weights_only=False)
    if any(x['policy_sha'] != policy_sha for x in (replay, fitted, paired)):
        raise ValueError('Shared data/MC continuation policy version changed')
    if paired.get('status') != 'INDEPENDENTLY_CONFIRMED' or not paired.get('d2_measured_passed'):
        raise ValueError('D1 independent confirmation and real D2 transfer must pass first')
    whitelist = paired['whitelist']
    if not whitelist or any(r['split'] != 'train' or r['continuation_sha'] != policy_sha for r in whitelist):
        raise ValueError('Only confirmed training parents may select actor states')
    actor = baseline_actor().to(device)
    actor.load_state_dict(act_state['actor'])
    policy = LivePolicy(actor, act_state['identity']['normalizer'], device)
    dataset = Demonstrations(manifest, normalizer=policy.normalizer)
    critic = TwinCritic(fitted['condition_dim']).to(device)
    critic.load_state_dict(fitted['critic'])
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    critic_optimizer.load_state_dict(fitted['optimizer'])
    scale = ReturnScale()
    scale.scale = np.asarray(fitted['scale'])
    train_pairs = [p for p in paired['pairs'] if p['split']=='train']
    endpoints = [p for p in paired['endpoints'] if p['split']=='train']
    states = [p for p in paired['checkpoints'] if p['split']=='train']
    branch_rows=[r for r in paired['branch_rows'] if r['split']=='train']
    if not branch_rows or any(r['continuation_sha']!=policy_sha for r in branch_rows):
        raise ValueError('All arms require identical real branch transitions for this frozen actor')
    td_rows=replay['rows']+branch_rows
    if not train_pairs or not endpoints or not states:
        raise ValueError('Real training pairs, endpoints and parent states required')
    size = run.config['algorithm']['batch_size']
    tensor = lambda value: torch.as_tensor(np.asarray(value), dtype=torch.float32, device=device)
    condition = lambda rows: torch.stack([r['condition'] for r in rows]).to(device)
    def td(rng):
        return replay_batch(td_rows, rng.integers(len(td_rows),size=size), policy, scale)
    def point(rng):
        rows = [endpoints[i] for i in rng.integers(len(endpoints),size=size)]
        return dict(condition=condition(rows), action=tensor([r['action'] for r in rows]),
                    returns=tensor(scale.transform([r['returns'] for r in rows])),
                    splits=['train']*size, policy_sha=policy_sha)
    def pair(rng):
        rows = [train_pairs[i] for i in rng.integers(len(train_pairs),size=size)]
        return dict(condition=condition(rows), plus=tensor([r['plus'] for r in rows]),
                    minus=tensor([r['minus'] for r in rows]),
                    delta=tensor(scale.transform([r['delta'] for r in rows])),
                    accepted=tensor([r['accepted'] for r in rows]), splits=['train']*size, policy_sha=policy_sha)
    def actor_batch(rng):
        # Identical mixture in all arms. B2/B3 apply the same mask to this pool.
        rows = [states[i] for i in rng.integers(len(states),size=size//2)]
        rows += [dict(checkpoint_id=f"replay:{r['parent_id']}", observation=r['observation'], condition=r['condition'])
                 for r in [replay['rows'][i] for i in rng.integers(len(replay['rows']),size=size-len(rows))]]
        obs, actions, valid = default_collate([dataset[int(i)] for i in rng.integers(len(dataset),size=size)])
        demo = ({k:v.to(device) for k,v in obs.items()}, actions.to(device), valid.to(device))
        return dict(obs=policy.batch([r['observation'] for r in rows]), condition=condition(rows),
                    splits=['train']*size, checkpoint_ids=[r['checkpoint_id'] for r in rows], demo=demo)
    streams = [DeterministicBatches(2000, fn, seed) for fn,seed in
               ((td,110000),(point,120000),(pair,130000),(actor_batch,140000))]
    atomic_json(run.report / 'short_sampling.json', {'actor_state_pool':'50% training paired checkpoints / 50% legal shared replay',
        'matched_stream_seeds':[110000,120000,130000,140000], 'batch_size':size,
        'critic_updates_per_arm':2000, 'actor_updates_per_arm':200,
        'policy_sha':policy_sha, 'paired_bundle_sha256':digest(paired_bundle)})
    result = short_updates(actor, critic, actor_optimizer, critic_optimizer,
        td_batches=streams[0], endpoint_batches=streams[1], pair_batches=streams[2], actor_batches=streams[3],
        whitelist=whitelist, data_hash=digest(paired_bundle), policy_sha=policy_sha,progress=progress)
    # Held-out labels are read only after every optimizer arm has completed.
    prediction_rows=[]
    with torch.no_grad():
        for arm in ('B1','B2','B3'):
            critic.load_state_dict(result[arm]['critic'])
            for pair_row in paired['pairs']:
                if pair_row['split']=='train':continue
                c=pair_row['condition'][None].to(device)
                predicted=(critic(c,tensor(pair_row['plus'][None])).mean(1)-
                           critic(c,tensor(pair_row['minus'][None])).mean(1))[0].cpu().numpy()
                observed=np.asarray(pair_row['delta'])/scale.scale
                prediction_rows.append(dict(arm=arm,parent_id=pair_row['parent_id'],split=pair_row['split'],
                    direction=pair_row['direction'],epsilon=pair_row['epsilon'],
                    predicted=predicted.tolist(),observed=observed.tolist(),
                    contrast_mse=float(np.mean((predicted-observed)**2))))
    atomic_json(run.report/'short_prediction_diagnostics.json',prediction_rows)
    outcomes = {}
    # Keep one simulator for all matched-reset arms; never recreate Kit/PhysX
    # inside this process. evaluate() resets each arm to the same seed schedule.
    adapter, ledger = make_adapter(1, limits)
    from ..diagnostics.live import NoiseSchedule
    noise=NoiseSchedule(adapter,21000)
    try:
        for arm in ('B0','B1','B2','B3'):
            progress(phase='evaluation',arm=arm,completed_episodes=0,requested_episodes=32)
            actor.load_state_dict(act_state['actor'] if arm=='B0' else result[arm]['actor'])
            path = directory / f'short_{arm}.pt'
            torch.save({'actor':actor.state_dict(),'identity':act_state['identity'],
                        'training':None if arm=='B0' else result[arm]}, path)
            before=ledger.record()
            outcomes[arm] = evaluate(adapter, LivePolicy(actor,policy.normalizer,device), range(21000,21032), ledger,
                run.result / f'short_{arm}_validation.csv', arm=arm, policy_sha=digest(path),noise=noise)
            progress(phase='evaluation',arm=arm,completed_episodes=len(outcomes[arm]),requested_episodes=32)
            after=ledger.record()
            atomic_json(run.report / f'short_{arm}_accounting.json',dict(before=before,after=after,
                individual_control_step_equivalents={k:after['individual_control_step_equivalents'][k]-v
                    for k,v in before['individual_control_step_equivalents'].items()}))
    finally:
        noise.close()
        adapter.close()
        ledger.close()
    atomic_json(run.report / 'short_ablation_outcomes.json', {'status':'MEASURED',
        'outcomes':outcomes, 'decision':'Requires pre-registered evidence gate; no automatic GO from training losses'})
    from ..evaluation.export import export_episodes
    export_episodes(run,outcomes,'short_same_data_32_validation',
        [run.result/f'short_{arm}_validation.csv' for arm in outcomes])
    from ..evaluation.short_gate import short_decision
    decision=short_decision(run,outcomes,prediction_rows,paired)
    return {'status':'PASSED' if decision['status']=='GO_PILOT' else decision['status'],
            'reason':decision['reason']}
