"""Real simulator branching with measured, never assumed, replay fidelity."""
import copy
import hashlib
import json
from contextlib import contextmanager
import numpy as np
import torch
from .progress import progress


class NoiseSchedule:
    """Align FORGE observation and interval-event RNG by physics time and channel.

    The pinned task draws fixed-size position/rotation/force noise tensors on each
    intermediate-value update. The channel has an isolated RNG stream, so an
    interval event does not shift observation-noise draws. This does not assert
    PhysX determinism: O/R/P trajectory comparisons still decide admission.
    """
    def __init__(self, adapter, seed):
        self.raw = adapter.raw
        self.ledger = adapter.physics_ledger
        if self.ledger is None:
            raise ValueError('An actual physics-step clock is required for paired noise')
        self.seed = seed
        self.origin = self._clock()
        self.intermediate = self.raw._compute_intermediate_values
        self.event_apply = self.raw.event_manager.apply
        self.adapter = adapter
        self.adapter_reset = adapter.reset
        self.raw._compute_intermediate_values = self._intermediate
        self.raw.event_manager.apply = self._event
        adapter.reset = self._reset

    def _reset(self, seed=None):
        # reset_to writes measured joint velocities into actuator velocity targets.
        # Stock FORGE reset never clears those targets, contaminating later P runs.
        robot = self.raw._robot
        robot.set_joint_velocity_target(torch.zeros_like(robot.data.joint_vel_target))
        robot.set_joint_effort_target(torch.zeros_like(robot.data.joint_effort_target))
        if seed is not None:
            self.seed, self.tick = int(seed), 0
        result = self.adapter_reset(seed=seed)
        # DirectRLEnv._reset_idx does not reset the interval-event countdown.
        # Same seed alone therefore does not align the dead-zone event schedule.
        with self.stream('interval_reset'):
            self.raw.event_manager.reset()
        return result

    def _clock(self):
        return sum(self.ledger.counts.values()) // self.ledger.num_envs

    @property
    def tick(self):
        return self._clock() - self.origin

    @tick.setter
    def tick(self, value):
        self.origin = self._clock() - value

    @contextmanager
    def stream(self, channel):
        payload = f'{self.seed}:{self.tick}:{channel}'.encode()
        seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], 'little') % (2**63-1)
        device = torch.device(self.raw.device)
        devices = [device.index or 0] if device.type == 'cuda' else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(seed)
            yield

    def _intermediate(self, *args, **kwargs):
        with self.stream('observation'):
            result = self.intermediate(*args, **kwargs)
        return result

    def _event(self, *args, **kwargs):
        mode = args[0] if args else kwargs.get('mode')
        if mode != 'interval':
            return self.event_apply(*args, **kwargs)
        with self.stream('interval'):
            return self.event_apply(*args, **kwargs)

    def close(self):
        self.raw._compute_intermediate_values = self.intermediate
        self.raw.event_manager.apply = self.event_apply
        self.adapter.reset = self.adapter_reset


def snapshot(adapter, noise):
    if adapter.n != 1:
        raise ValueError('Paired restore is single-environment only')
    r = adapter.raw
    tensors = {k: v.clone() for k,v in vars(r).items() if torch.is_tensor(v)}
    properties = {}
    for name, asset in {'robot': r._robot, 'held': r._held_asset, 'fixed': r._fixed_asset}.items():
        view = asset.root_physx_view
        properties[name] = {key: getattr(view, f'get_{key}')().clone()
                            for key in ('masses', 'inertias', 'material_properties')}
    targets = {name: {kind: getattr(asset.data, f'joint_{kind}_target').clone()
                     for kind in ('pos', 'vel', 'effort')}
               for name, asset in r.scene.articulations.items()}
    return dict(scene=copy.deepcopy(r.scene.get_state()), tensors=tensors, properties=properties, targets=targets,
        counters={k: getattr(r,k) for k in ('common_step_counter','_sim_step_counter')},
        intervals=copy.deepcopy(r.event_manager._interval_term_time_left),
        adapter={k: copy.deepcopy(getattr(adapter,k)) for k in
                 ('history','previous','success_count','success_paid','ever_success','protected','observation','current_frame')},
        noise_seed=noise.seed, noise_tick=noise.tick,
        limitation='PhysX contact warm-start/solver internals are not exposed; snapshot admission requires O/R comparison')


def restore(adapter, noise, state, ledger):
    r = adapter.raw
    with ledger.using('restore'):
        for name, asset in {'robot':r._robot, 'held':r._held_asset, 'fixed':r._fixed_asset}.items():
            view = asset.root_physx_view
            for key, value in state['properties'][name].items():
                indices = torch.arange(len(value), device='cpu', dtype=torch.int32)
                getattr(view, f'set_{key}')(value, indices)
        r.scene.reset_to(state['scene'])
        # Isaac Lab reset_to assumes PD position control and replaces targets
        # with measured joint states. FORGE is effort controlled: restore the
        # actual actuator commands, including finger position/velocity targets.
        for name, targets in state['targets'].items():
            asset = r.scene.articulations[name]
            asset.set_joint_position_target(targets['pos'])
            asset.set_joint_velocity_target(targets['vel'])
            asset.set_joint_effort_target(targets['effort'])
        r.scene.write_data_to_sim()
        r.sim.forward()
        r.scene.update(0.)
        for key, value in state['tensors'].items():
            setattr(r, key, value.clone())
        for key, value in state['counters'].items():
            setattr(r, key, value)
        r.event_manager._interval_term_time_left = copy.deepcopy(state['intervals'])
        r.last_update_timestamp = r._robot._data._sim_timestamp
        for key, value in state['adapter'].items():
            setattr(adapter, key, copy.deepcopy(value))
        noise.seed, noise.tick = state['noise_seed'], state['noise_tick']
        r.sim.render()


def prefix_replay(adapter, noise, seed, actions, ledger):
    from collections import deque
    rewards=deque(maxlen=8)
    noise.seed, noise.tick = seed, 0
    adapter.reset(seed=seed)
    for action in actions:
        with ledger.using('prefix'):
            row = adapter.step(np.asarray(action)[None])
        rewards.append(float(row['reward'][0]))
        if row['terminated'][0] or row['truncated'][0] or row['success'][0] or row['unsafe'][0]:
            raise RuntimeError('Prefix ended before the checkpoint; not the same branch state')
    return {'last':row if len(actions) else None,'recent_rewards':list(rewards)}


def continuation(adapter, policy, first_plan, ledger, *, category='branch', collect_td=False):
    """Execute candidate then the frozen ACT to true success/finite horizon."""
    if adapter.n != 1 or adapter.limits is None or not adapter.diagnostic:
        raise ValueError('Single real environment and accepted costs required')
    observer = getattr(ledger, 'progress', None)
    if observer is not None: observer.begin()
    plan = np.asarray(first_plan, dtype=np.float32).reshape(1,4,4)
    trajectory = {k: [] for k in ('position_m','rotation_matrix','joint_rad','gripper_m','gripper_velocity_m_s','linear_velocity_m_s',
        'angular_velocity_rad_s','joint_velocity_rad_s','force_N','torque_Nm','controller')}
    returns = np.zeros(2)
    step = 0
    done = False
    success = False
    transitions = []
    while not done:
        if collect_td:
            from ..models.critic import condition_features
            condition = condition_features(policy.actor,policy.batch(adapter.observation),
                (adapter.raw.max_episode_length-1-adapter.raw.episode_length_buf).clamp_min(0),
                torch.as_tensor(adapter.controller(),device=policy.device)).cpu()[0]
            block_return = np.zeros(2)
        for k in range(4):
            with ledger.using(category):
                row = adapter.step(plan[:,k])
            returns += .99**step * np.array([row['reward'][0], row['cost'][0]])
            if collect_td:block_return += .99**k * np.array([row['reward'][0],row['cost'][0]])
            step += 1
            if observer is not None: observer.data['trajectory_step'] = step
            for key,value in row['physical'].items():trajectory[key].append(value[0].copy())
            trajectory['force_N'].append(row['substep_wrench'][:,0,:3].copy())
            trajectory['torque_Nm'].append(row['substep_wrench'][:,0,3:].copy())
            trajectory['controller'].append(row['controller'][0].copy())
            success = bool(row['success'][0])
            done = success or bool(row['terminated'][0] or row['truncated'][0] or row['unsafe'][0])
            if done:
                break
        if collect_td:
            next_obs=policy.batch([row['observation'][0]])
            next_condition=condition_features(policy.actor,next_obs,
                torch.as_tensor(row['remaining'],device=policy.device),
                torch.as_tensor(row['controller'],device=policy.device)).cpu()[0]
            with torch.no_grad():next_action=policy.actor(next_obs).cpu().numpy()[0]
            transitions.append(dict(condition=condition,action=plan[0].copy(),
                returns=block_return,next_condition=next_condition,next_action=next_action,
                valid=np.arange(4)<=k,gamma_k=.99**(k+1),bootstrap=not done,source='real_forge'))
        if not done:
            plan = policy(adapter.observation)
    result = {'trajectory': {k: np.stack(v) for k,v in trajectory.items()}, 'returns':returns,
            'success':success, 'steps':step, 'source':'real_forge_full_remaining_horizon',
            'unsafe':bool(row['unsafe'][0]),'transitions':transitions}
    if observer is not None: observer.finish(result)
    return result


def compare(original, replayed, tolerances, return_scale):
    physical={'position_m','rotation_matrix','joint_rad','gripper_m','gripper_velocity_m_s','linear_velocity_m_s',
        'angular_velocity_rad_s','joint_velocity_rad_s','force_N','torque_Nm','controller'}
    required = physical | {'scaled_return','scaled_cost','success','steps'}
    if set(tolerances) != required:
        raise ValueError('Pre-registered physical and return fidelity tolerances required')
    errors = {}
    for key in physical:
        a, b = original['trajectory'][key], replayed['trajectory'][key]
        errors[key] = None if a.shape != b.shape else float(np.abs(a-b).max())
    delta = np.abs(original['returns']-replayed['returns']) / np.asarray(return_scale)
    errors.update(scaled_return=float(delta[0]), scaled_cost=float(delta[1]),
                  success=float(original['success'] != replayed['success']), steps=abs(original['steps']-replayed['steps']))
    passed = all(v is not None and np.isfinite(v) and v <= tolerances[k] for k,v in errors.items())
    return {'passed':passed, 'errors':errors}


def fidelity_checkpoint(adapter, policy, noise, seed, prefix, candidates, ledger, tolerances, return_scale,
                        *, first_at_checkpoint=False):
    """O then R/P for each candidate, no reset between original checkpoint and O."""
    records = []
    for candidate_id, action in enumerate(candidates):
        progress(ledger, candidate=candidate_id, method='original', repeat=0)
        original_before=sum(ledger.counts.values())
        if candidate_id!=0 or not first_at_checkpoint:
            prefix_replay(adapter, noise, seed, prefix, ledger)
        state = snapshot(adapter, noise)
        original = continuation(adapter, policy, action, ledger)
        original_steps=(sum(ledger.counts.values())-original_before)/ledger.decimation
        for method in ('snapshot_restore', 'prefix_replay'):
            for repeat in range(3):
                progress(ledger, candidate=candidate_id, method=method, repeat=repeat)
                before=sum(ledger.counts.values())
                if method == 'snapshot_restore':
                    restore(adapter, noise, state, ledger)
                else:
                    prefix_replay(adapter, noise, seed, prefix, ledger)
                actual = continuation(adapter, policy, action, ledger)
                records.append(dict(method=method, repeat=repeat, candidate=candidate_id,
                    **compare(original, actual, tolerances, return_scale),
                    env_steps=(sum(ledger.counts.values())-before)/ledger.decimation,
                    original_env_steps=original_steps if method=='snapshot_restore' and repeat==0 else 0.,
                    original=original, repeated=actual))
        if all(any(not r['passed'] for r in records if r['method']==method)
               for method in ('snapshot_restore','prefix_replay')):
            # Both methods already violate a required comparison. Further
            # candidates cannot rescue an all-comparisons equivalence gate.
            break
    accepted = [method for method in ('snapshot_restore','prefix_replay')
                if all(r['passed'] for r in records if r['method'] == method)]
    return {'accepted_methods':accepted, 'records':records,
            'continuations_used':len(records)+candidate_id+1,
            'candidates_measured':candidate_id+1, 'candidates_planned':len(candidates),
            'status':'PASSED' if accepted else 'NO_GO_REPLAY'}


def collect_pairs(adapter, policy, noise, checkpoints, method, ledger, csv_path,
                  *, policy_sha, continuation_cap=1500, already_used=0):
    """D1 screening: 24 parents x (3 A0 + 2 directions x 2 eps x 2 signs x 3)."""
    import csv
    from pathlib import Path
    from .pairs import candidates as perturb
    if len(checkpoints) != 24 or sum(c['hard'] for c in checkpoints) != 16:
        raise ValueError('D1 needs 24 independent parents, including 16 predefined hard states')
    if len({c['parent_id'] for c in checkpoints}) != 24:
        raise ValueError('Repeated parents do not increase independent checkpoint count')
    if [sum(c['split']==s for c in checkpoints) for s in ('train','validation','test')] != [12,6,6]:
        raise ValueError('D1 parent split must be 12/6/6')
    if method not in ('snapshot_restore','prefix_replay'):
        raise ValueError('An empirically admitted restore method is required')
    if any(method not in c['accepted_methods'] for c in checkpoints):
        raise ValueError('Unverified checkpoint cannot produce accepted paired labels')
    directions = np.zeros((2,4,4), dtype=np.float32)
    directions[0,:,2] = .5  # unit-norm native upward/unload template
    directions[1,:,0] = .5  # unit-norm lateral template
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError('Do not overwrite diagnostic outcomes')
    fields = ['parent_id','split','hard','policy_sha','method','noise_id','candidate','direction',
              'epsilon','symmetric','action','physical_units','return','cost','success','unsafe',
              'steps','prefix_steps','restore_steps','branch_steps','source']
    results, branch_rows = [], []
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for checkpoint in checkpoints:
            variants = [('base', -1, 0., True, checkpoint['a0'])]
            for direction_id, direction in enumerate(directions):
                for epsilon in (.02,.05):
                    plus, minus, symmetric = perturb(checkpoint['a0'], direction, epsilon)
                    variants += [('plus', direction_id, epsilon, symmetric, plus),
                                 ('minus', direction_id, epsilon, symmetric, minus)]
            for repeat in range(3):
                noise_id = 300000 + checkpoint['seed']*10 + repeat
                for name, direction, epsilon, symmetric, action in variants:
                    if already_used + len(results) >= continuation_cap:
                        raise RuntimeError('BUDGET_EXHAUSTED before diagnostic continuation')
                    before=dict(ledger.counts)
                    if method == 'snapshot_restore':
                        restore(adapter, noise, checkpoint['snapshot'], ledger)
                    else:
                        prefix_replay(adapter, noise, checkpoint['seed'], checkpoint['prefix'], ledger)
                    # Same physical-time/channel schedule for A0/A+/A- within a noise block.
                    noise.seed = noise_id
                    actual = continuation(adapter, policy, action, ledger, collect_td=True)
                    for transition in actual['transitions']:
                        transition.update(parent_id=checkpoint['parent_id'],split=checkpoint['split'],
                            continuation_sha=policy_sha,behavior_sha=policy_sha,noise_id=noise_id)
                        branch_rows.append(transition)
                    row = dict(parent_id=checkpoint['parent_id'], split=checkpoint['split'],
                        hard=int(checkpoint['hard']), policy_sha=policy_sha, method=method, noise_id=noise_id,
                        candidate=name, direction=direction, epsilon=epsilon, symmetric=symmetric,
                        cost=float(actual['returns'][1]), success=int(actual['success']),
                        unsafe=int(actual['unsafe']),steps=actual['steps'], source=actual['source'],
                        action=json.dumps(np.asarray(action).tolist()),
                        physical_units='normalized pre-EMA x/y/z/yaw; .05 m/unit; 3pi/4 rad/unit bolt yaw',
                        **{k+'_steps':(ledger.counts[k]-before[k])/ledger.decimation for k in ('prefix','restore','branch')})
                    row['return'] = float(actual['returns'][0])
                    writer.writerow(row)
                    f.flush()
                    results.append(dict(row, action=np.asarray(action).copy(),
                                        condition=checkpoint['condition'], observation=checkpoint['observation']))
    return {'rows':results, 'branch_rows':branch_rows,'continuations':len(results), 'policy_sha':policy_sha,
            'directions':directions, 'accounting':ledger.record(), 'status':'SCREENING_ONLY'}


def diagnose_transfer(adapter, policy, critic, checkpoint, improving_action, demo, noise, ledger, *, method):
    """D2 shadow actors: direct candidate versus actual Q-only/Q+BC updates."""
    from ..training.live import LivePolicy
    from ..training.updates import actor_step
    if checkpoint['split'] != 'train' or not checkpoint.get('independently_confirmed'):
        raise ValueError('D2 selection requires an independently confirmed training checkpoint')
    if method not in checkpoint['accepted_methods']:
        raise ValueError('Replay admission required before transfer claims')
    def reset_branch():
        if method == 'snapshot_restore':
            restore(adapter, noise, checkpoint['snapshot'], ledger)
        else:
            prefix_replay(adapter, noise, checkpoint['seed'], checkpoint['prefix'], ledger)
        noise.seed = 700000 + checkpoint['seed']
    outcomes = {'parent_id':checkpoint['parent_id']}
    reset_branch()
    outcomes['baseline'] = continuation(adapter, policy, checkpoint['a0'], ledger)
    reset_branch()
    outcomes['direct_candidate'] = continuation(adapter, policy, improving_action, ledger)
    original = policy.actor.state_dict()
    obs = policy.batch([checkpoint['observation']])
    condition = checkpoint['condition'].reshape(1,-1).to(policy.device)
    with torch.no_grad():
        old = policy.actor(obs)
    direction = torch.as_tensor(improving_action, device=policy.device).reshape(1,4,4)-old
    for name, bc_weight, step_weight in [('q_only',0.,0.), ('q_bc_step',1.,1.)]:
        actor = copy.deepcopy(policy.actor)
        actor.load_state_dict(original)
        optimizer = torch.optim.Adam(actor.parameters(), lr=1e-5)
        update = actor_step(actor, critic, optimizer, obs, condition,
            torch.ones(1, dtype=torch.bool, device=policy.device), old,
            demo=demo if bc_weight else None, alpha_bc=bc_weight, eta_step=step_weight)
        changed = LivePolicy(actor, policy.normalizer, policy.device)
        reset_branch()
        actual = continuation(adapter, changed, update['after'].cpu().numpy(), ledger)
        delta = update['delta_action']
        cosine = torch.nn.functional.cosine_similarity(delta.flatten(1), direction.flatten(1))
        outcomes[name] = dict(actual=actual, action_before=update['before'].cpu(),
            action_after=update['after'].cpu(), direction_cosine=float(cosine.item()),
            q_grad_norm=update['q_grad_norm'], bc_grad_norm=update['bc_grad_norm'],
            q_bc_dot=update['q_bc_dot'], saturation=update['saturation'],
            actor=copy.deepcopy(actor.state_dict()),
            normal_validation='REQUIRED_BEFORE_TRANSFER_ACCEPTANCE')
    return outcomes
