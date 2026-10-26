"""Bounded engineering probe; never admitted as D0/D1 or insertion results."""
import argparse
import faulthandler
import json
from pathlib import Path
import time

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', required=True)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(p)
args = p.parse_args()
app = AppLauncher(args).app
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import numpy as np
import torch
from forcecritic.envs.forge import ForgeAdapter, configure_camera, TASK
from forcecritic.envs.accounting import PhysicsLedger
from forcecritic.training.baseline import accepted_limits
from forcecritic.training.pretrain import baseline_actor
from forcecritic.training.live import LivePolicy
from forcecritic.diagnostics.live import NoiseSchedule, snapshot, restore, prefix_replay
from forcecritic.diagnostics.artifacts import atomic_json
from forcecritic.runtime import source_hashes

out = Path(args.output)
report = Path('reports/20261023T104759Z')
cfg = configure_camera(parse_env_cfg(TASK, device=args.device, num_envs=1))
cfg.seed = 0
ledger = PhysicsLedger(1, cfg.decimation, 5000)
result = dict(status='RUNNING', purpose='Engineering reset/restore probe, not scientific admission',
              source_hashes=source_hashes(), observations=[], comparisons=[])
def save(phase):
    result.update(phase=phase, time=time.time(), accounting=ledger.record())
    atomic_json(out, result)
    print('REPLAY_PROBE', phase, flush=True)

def capture(adapter, noise):
    r = adapter.raw
    keys = ('actions', 'ema_factor', 'task_prop_gains', 'pos_threshold', 'rot_threshold',
            'dead_zone_thresholds', 'init_fixed_pos_obs_noise', 'fingertip_midpoint_pos',
            'joint_pos', 'joint_vel', 'held_pos', 'held_quat', 'force_sensor_world_smooth')
    data = {k:getattr(r,k).detach().cpu().numpy().copy() for k in keys}
    data.update(noise_tick=np.array(noise.tick),
                interval=np.array([t.item() for t in r.event_manager._interval_term_time_left]),
                mass=r._held_asset.root_physx_view.get_masses().cpu().numpy().copy(),
                joint_position_target=r._robot.data.joint_pos_target.cpu().numpy().copy(),
                joint_velocity_target=r._robot.data.joint_vel_target.cpu().numpy().copy(),
                joint_effort_target=r._robot.data.joint_effort_target.cpu().numpy().copy())
    return data

try:
    save('environment_initialization')
    env = gym.make(TASK, cfg=cfg)
    adapter = ForgeAdapter(env, limits=accepted_limits(report/'p0_acceptance.json'),
        hard_limits=json.loads((report/'p0_acceptance.json').read_text())['physical_protection_limits'],
        physics_ledger=ledger, diagnostic=True)
    noise = NoiseSchedule(adapter, 60000)
    state = torch.load('checkpoints/20261023T104759Z/act0.pt',map_location='cpu',weights_only=False)
    actor = baseline_actor().to(args.device)
    actor.load_state_dict(state['actor'])
    policy = LivePolicy(actor,state['identity']['normalizer'],args.device)
    reference = None
    prefix = []
    for repeat in range(3):
        noise.seed,noise.tick = 60000,0
        adapter.reset(seed=60000)
        data = capture(adapter,noise)
        result['observations'].append(data)
        if reference is None: reference = data
        result['comparisons'].append(dict(kind='reset',repeat=repeat,
            max_errors={k:float(np.max(np.abs(v-reference[k]))) for k,v in data.items()}))
        save(f'reset_{repeat}_measured')
        if repeat == 0:
            for _ in range(2):
                plan=policy(adapter.observation)
                for k in range(4):
                    with ledger.using('prefix'): adapter.step(plan[:,k])
                    prefix.append(plan[0,k].copy())
        else:
            for action in prefix:
                with ledger.using('prefix'):adapter.step(action[None])
        reached = capture(adapter,noise)
        if repeat == 0: reached_ref = reached
        result['comparisons'].append(dict(kind='prefix',repeat=repeat,
            max_errors={k:float(np.max(np.abs(v-reached_ref[k]))) for k,v in reached.items()}))
        saved = snapshot(adapter,noise)
        action = policy(adapter.observation)[:,0]
        with ledger.using('branch'): original=adapter.step(action)
        restore(adapter,noise,saved,ledger)
        restored = capture(adapter,noise)
        result['comparisons'].append(dict(kind='restore_before_step',repeat=repeat,
            max_errors={k:float(np.max(np.abs(v-reached[k]))) for k,v in restored.items()}))
        with ledger.using('branch'): actual=adapter.step(action)
        result['comparisons'].append(dict(kind='restore_one_step',repeat=repeat,
            max_errors={k:float(np.max(np.abs(v-actual['physical'][k]))) for k,v in original['physical'].items()}))
        save(f'prefix_and_restore_{repeat}_measured')
    result['status']='MEASURED'
except BaseException as error:
    result.update(status='FAILED_IMPLEMENTATION',error=repr(error))
    raise
finally:
    save('finished')
    if 'noise' in globals(): noise.close()
    if 'adapter' in globals(): adapter.close()
    ledger.close()
    faulthandler.cancel_dump_traceback_later()
    app.close()
