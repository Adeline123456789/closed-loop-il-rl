"""Real FORGE rollout, frozen-policy evaluation and online TD critic fitting.

All entry points receive a live ForgeAdapter; no fabricated transitions or CPU
gradient outcome can enter these collectors. Models are saved separately from
measured CSV outcomes.
"""
import copy
import csv
from pathlib import Path
import numpy as np
import torch
from ..models.critic import condition_features, TwinCritic
from .updates import td_target, critic_step, polyak
from .replay import ReturnScale


def observation_batch(observations, normalizer, device):
    rows = list(observations.values()) if isinstance(observations, dict) else observations
    rgb = torch.as_tensor(np.stack([r['rgb'] for r in rows]), device=device).permute(0,3,1,2).float()/255
    out = {'rgb': rgb}
    for key in ('state', 'history'):
        values = np.stack([r[key] for r in rows]).astype(np.float32)
        stats = normalizer[key]
        values = (values - np.asarray(stats['mean'])) / np.asarray(stats['std'])
        out[key] = torch.as_tensor(values, dtype=torch.float32, device=device)
    return out


class LivePolicy:
    def __init__(self, actor, normalizer, device):
        self.actor = actor.eval()
        self.normalizer = normalizer
        self.device = device

    def batch(self, observations):
        return observation_batch(observations, self.normalizer, self.device)

    @torch.no_grad()
    def __call__(self, observations):
        return self.actor(self.batch(observations)).cpu().numpy()


def evaluate(adapter, policy, seeds, ledger, csv_path, *, arm, policy_sha, phase='validation', noise=None):
    """Independent reset evaluations, including raw substep load metrics."""
    if adapter.n != 1:
        raise ValueError('Independent seeded evaluation uses one stock synchronous-reset environment')
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if csv_path.exists():
        raise FileExistsError('Use an immutable evaluation attempt path')
    fields = ['arm','policy_sha','split','seed','parent_id','stable_success','ever_success',
              'final_success','return','cost','overload_steps','peak_force_N','peak_torque_Nm','steps','control_dt','termination_reason','source']
    results = []
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for seed in seeds:
            if noise is not None:
                noise.seed,noise.tick = int(seed),0
            adapter.reset(seed=int(seed))
            success = ever = final = False
            reward = cost = 0.
            steps = overload = 0
            peak_force = peak_torque = 0.
            done = False
            while not done:
                plan = policy(adapter.observation)
                # The whole chunk is computed once; no replan inside the block.
                for k in range(4):
                    with ledger.using('evaluation'):
                        row = adapter.step(plan[:, k])
                    if row['cost'] is None:
                        raise ValueError('Evaluation cost requires locked physical limits')
                    reward += .99**steps * float(row['reward'][0])
                    cost += .99**steps * float(row['cost'][0])
                    steps += 1
                    overload += int(row['cost'][0] > 0)
                    raw = row['substep_wrench'][:,0]
                    peak_force = max(peak_force, float(np.linalg.norm(raw[:,:3], axis=-1).max()))
                    peak_torque = max(peak_torque, float(np.linalg.norm(raw[:,3:], axis=-1).max()))
                    success |= bool(row['success'][0])
                    ever |= bool(row['geometric_success'][0])
                    final = bool(row['geometric_success'][0])
                    done = success or bool(row['terminated'][0] or row['truncated'][0] or row['unsafe'][0])
                    if done:
                        break
            result = dict(arm=arm, policy_sha=policy_sha, split=phase, seed=int(seed),
                parent_id=f'{phase}-{seed}', stable_success=int(success), ever_success=int(ever),
                final_success=int(final), return_=reward, cost=cost, overload_steps=overload,
                peak_force_N=peak_force, peak_torque_Nm=peak_torque, steps=steps, source='real_forge',
                control_dt=float(adapter.raw.step_dt),
                termination_reason='unsafe' if row['unsafe'][0] else 'stable_success' if success else 'finite_horizon')
            result['return'] = result.pop('return_')
            writer.writerow(result)
            stream.flush()
            results.append(result)
    return results


def collect_shared(adapter, policy, frozen_features, ledger, *, policy_sha, seed=0,
                   episode_cap=100, transition_cap=50000):
    """Complete ACT0 chunks, with preassigned whole-parent 70/15/15 splits."""
    rng = np.random.default_rng(seed)
    adapter.reset(seed=seed)
    rows = []
    episodes = 0
    transitions = 0
    parent_numbers = np.arange(adapter.n)
    next_parent = adapter.n
    episode_active = np.ones(adapter.n, dtype=bool)
    while episodes + adapter.n <= episode_cap and transitions + adapter.n * 4 <= transition_cap:
        obs = copy.deepcopy(adapter.observation)
        tensor_obs = policy.batch(obs)
        remaining = (adapter.raw.max_episode_length - 1 - adapter.raw.episode_length_buf).clamp_min(0)
        controller = torch.as_tensor(adapter.controller(), device=policy.device)
        condition = condition_features(frozen_features, tensor_obs, remaining, controller).cpu()
        plan = policy(obs)
        # Exploration acts on full native chunks; no independent residual model.
        if rng.random() < .5:
            plan = np.clip(plan + rng.normal(0, .02, plan.shape), -1, 1).astype(np.float32)
        active = episode_active.copy()
        totals = np.zeros((adapter.n, 2), dtype=np.float32)
        valid = np.zeros((adapter.n,4), dtype=bool)
        last = {}
        ended = np.zeros(adapter.n, dtype=bool)
        for k in range(4):
            try:
                with ledger.using('interaction'):
                    row = adapter.step(plan[:,k])
            except RuntimeError as error:
                if 'BUDGET_EXHAUSTED' not in str(error):
                    raise
                return {'rows':rows,'individual_transitions':transitions,'complete_episodes':episodes,
                    'source':'real_forge','policy_sha':policy_sha,'accounting':ledger.record(),
                    'stop_reason':'BUDGET_EXHAUSTED','incomplete_block_discarded':True}
            transitions += adapter.n
            if row['cost'] is None:
                raise ValueError('Uncalibrated cost cannot enter critic replay')
            valid[:,k] = active
            totals[active] += .99**k * np.stack([row['reward'], row['cost']], -1)[active]
            for i in np.flatnonzero(active):
                last[i] = (row['observation'][i], row['remaining'][i], row['controller'][i],
                           bool(row['terminated'][i] or row['success'][i] or row['unsafe'][i]),
                           bool(row['success'][i]),bool(row['unsafe'][i]))
            ended = row['terminated'] | row['truncated']
            episode_active &= ~(ended | row['success'] | row['unsafe'])
            # Success is a terminal chunk. Remaining vector rows are still charged.
            active &= ~(ended | row['success'] | row['unsafe'])
            if ended.any() or not active.any():
                break
        for i in range(adapter.n):
            if not valid[i].any():
                continue
            nxt, time_left, ctrl, terminal, success, unsafe = last[i]
            next_condition = condition_features(frozen_features, policy.batch([nxt]),
                torch.tensor([time_left], device=policy.device),
                torch.as_tensor(ctrl[None], device=policy.device)).cpu()[0]
            split = 'train' if parent_numbers[i]%20<14 else 'validation' if parent_numbers[i]%20<17 else 'test'
            rows.append(dict(parent_id=f'shared-{seed}-{parent_numbers[i]}', split=split,
                behavior_sha=policy_sha, source='real_forge', observation=obs[i], action=plan[i],
                valid=valid[i], returns=totals[i], condition=condition[i], next_observation=nxt,
                next_condition=next_condition, bootstrap=not terminal, gamma_k=.99**int(valid[i].sum())))
            rows[-1].update(success=success,unsafe=unsafe)
        if ended.any() or not episode_active.any():
            if not ended.all():
                adapter.reset()
            episodes += adapter.n
            episode_active[:] = True
            for i in range(adapter.n):
                parent_numbers[i] = next_parent
                next_parent += 1
    return {'rows': rows, 'individual_transitions': transitions, 'complete_episodes': episodes,
            'source': 'real_forge', 'policy_sha': policy_sha, 'accounting': ledger.record()}


def replay_batch(rows, indices, policy, scale):
    batch = [rows[int(i)] for i in indices]
    device = policy.device
    tensor = lambda x: torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
    missing = [r for r in batch if 'next_action' not in r]
    if missing:
        actions = policy([r['next_observation'] for r in missing])
        for row, action in zip(missing, actions):row['next_action']=action
    return dict(condition=torch.stack([r['condition'] for r in batch]).to(device),
        action=tensor([r['action'] for r in batch]),
        next_obs=None, next_action=tensor([r['next_action'] for r in batch]),
        next_condition=torch.stack([r['next_condition'] for r in batch]).to(device),
        returns=tensor(scale.transform([r['returns'] for r in batch])),
        gamma_k=tensor([r['gamma_k'] for r in batch]), bootstrap=tensor([r['bootstrap'] for r in batch]),
        splits=[r['split'] for r in batch])


def fit_online_critic(bundle, policy, *, updates=10000, batch_size=128, seed=0):
    if bundle['source'] != 'real_forge' or not 1 <= updates <= 10000:
        raise ValueError('Real bounded online collection required')
    rows = [r for r in bundle['rows'] if r['split']=='train']
    if not rows or any(r['source'] != 'real_forge' for r in rows):
        raise ValueError('Real training parents only')
    # Normalize full finite-horizon values, not the much smaller one-block
    # variation. These behavior returns set units only; they are NOT relabeled
    # as Monte Carlo supervision for a different continuation policy.
    totals, discounts, completed = {}, {}, set()
    for row in rows:
        parent = row['parent_id']
        totals.setdefault(parent,np.zeros(2))
        discounts.setdefault(parent,1.)
        totals[parent] += discounts[parent]*np.asarray(row['returns'])
        discounts[parent] *= row['gamma_k']
        if not row['bootstrap']:
            completed.add(parent)
    if len(completed)<2:
        raise ValueError('NO_REWARD_VARIATION: fewer than two complete training returns')
    scale = ReturnScale().fit([totals[p] for p in sorted(completed)], ['train']*len(completed))
    torch.manual_seed(seed)
    critic = TwinCritic(len(rows[0]['condition'])).to(policy.device)
    target = copy.deepcopy(critic).eval().requires_grad_(False)
    frozen_actor = copy.deepcopy(policy.actor).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)
    rng = np.random.default_rng(seed)
    losses = []
    for _ in range(updates):
        batch = replay_batch(rows, rng.integers(len(rows), size=batch_size), policy, scale)
        labels = td_target(frozen_actor, target, batch['next_obs'], batch['next_condition'],
                           batch['returns'], batch['gamma_k'], batch['bootstrap'],batch['next_action'])
        losses.append(critic_step(critic, optimizer, batch['condition'], batch['action'], labels))
        polyak(critic, target)
    validation = [r for r in bundle['rows'] if r['split']=='validation']
    residuals=[]
    with torch.no_grad():
        for start in range(0,len(validation),batch_size):
            batch=replay_batch(validation,np.arange(start,min(start+batch_size,len(validation))),policy,scale)
            labels=td_target(frozen_actor,target,batch['next_obs'],batch['next_condition'],
                             batch['returns'],batch['gamma_k'],batch['bootstrap'],batch['next_action'])
            predictions=critic(batch['condition'],batch['action'])
            residuals.extend((predictions.mean(1)-labels).square().cpu().tolist())
    return {'critic': critic.state_dict(), 'target_critic': target.state_dict(),
            'optimizer': optimizer.state_dict(), 'scale': scale.scale, 'losses': losses,
            'scale_source':'completed training behavior-episode returns; normalization only',
            'scale_training_parents':sorted(completed),
            'validation_parents':sorted({r['parent_id'] for r in validation}),
            'validation_td_mse':None if not residuals else np.mean(residuals,axis=0).tolist(),
            'validation_note':'Held-out TD consistency, not a Monte Carlo accuracy claim; test rows unused',
            'condition_dim': len(rows[0]['condition']), 'policy_sha': bundle['policy_sha']}
