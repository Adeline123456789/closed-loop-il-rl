"""Bounded engineering probe; never admitted as D0/D1 or insertion results."""
import argparse
import faulthandler
import json
from pathlib import Path
import time
import os

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--output', required=True)
p.add_argument('--fresh-full', action='store_true', help='One fresh-process prefix and full ACT continuation, engineering evidence only')
p.add_argument('--only-prefix', action='store_true', help='Inspect real input determinism before an expensive continuation')
p.add_argument('--render-aa', choices=['FXAA','Off','DLSS'], help='Engineering-only renderer variation; never silently alter the recorded baseline')
p.add_argument('--reference', help='Replay this measured fixed prefix and first candidate action in a fresh process')
p.add_argument('--reset-render-history', action='store_true', help='Engineering probe for full temporal renderer reset before each render')
p.add_argument('--spatial-render', action='store_true', help='Engineering-only non-temporal, non-sampled lighting settings')
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
from forcecritic.diagnostics.live import NoiseSchedule, snapshot, restore, prefix_replay, continuation
from forcecritic.diagnostics.artifacts import atomic_json
from forcecritic.runtime import source_hashes

out = Path(args.output)
artifact_name=out.parent.name+'_'+out.stem
report = Path('reports/20261023T104759Z')
cfg = configure_camera(parse_env_cfg(TASK, device=args.device, num_envs=1))
cfg.seed = 0
if args.render_aa: cfg.sim.render.antialiasing_mode=args.render_aa
if args.spatial_render:
    cfg.sim.render.rendering_mode='performance'
    cfg.sim.render.enable_dl_denoiser=False
    cfg.sim.render.carb_settings={'/rtx/directLighting/sampledLighting/enabled':False,
                                 '/rtx/raytracing/cached/enabled':False,
                                 '/rtx/post/tonemap/dither':0.0}
ledger = PhysicsLedger(1, cfg.decimation, 5000)
result = dict(status='RUNNING', purpose='Engineering reset/restore probe, not scientific admission',
              source_hashes=source_hashes(), observations=[], comparisons=[],render_aa=args.render_aa,
              reference=args.reference)
result['reset_render_history']=args.reset_render_history
result['spatial_render']=args.spatial_render
def save(phase):
    result.update(phase=phase, time=time.time(), accounting=ledger.record())
    atomic_json(out, result)
    atomic_json(report/'engineering_job.json',dict(pid=os.getpid(),status=result['status'],
        phase=phase,output=str(out),progress=str(out.parent/out.stem/'diagnostic_live_progress.json'),
        time=time.time(),log=str(out.with_suffix('.log'))))
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
    from forcecritic.diagnostics.progress import DiagnosticProgress
    observer=DiagnosticProgress(ledger,out.parent/out.stem,Path('checkpoints/20261023T104759Z')/artifact_name)
    observer.mark(phase='engineering_initialization')
    save('environment_initialization')
    env = gym.make(TASK, cfg=cfg)
    if args.reset_render_history:
        import omni.usd
        context=omni.usd.get_context()
        if not hasattr(context,'reset_renderer_accumulation'):
            raise RuntimeError('Installed omni.usd lacks reset_renderer_accumulation; no unsupported API assumed')
        render=env.unwrapped.sim.render
        def cold_render(*a,**kw):
            context.reset_renderer_accumulation()
            return render(*a,**kw)
        env.unwrapped.sim.render=cold_render
    import carb.settings
    settings=carb.settings.get_settings()
    result['actual_render_settings']={key:settings.get(key) for key in (
        '/rtx/post/aa/op','/rtx/post/dlss/execMode','/rtx-transient/dldenoiser/enabled',
        '/rtx/directLighting/sampledLighting/enabled','/rtx/rendermode')}
    result['actual_render_settings']['/rtx/domeLight/upperLowerStrategy']=settings.get('/rtx/domeLight/upperLowerStrategy')
    adapter = ForgeAdapter(env, limits=accepted_limits(report/'p0_acceptance.json'),
        hard_limits=json.loads((report/'p0_acceptance.json').read_text())['physical_protection_limits'],
        physics_ledger=ledger, diagnostic=True)
    noise = NoiseSchedule(adapter, 60000)
    state = torch.load('checkpoints/20261023T104759Z/act0.pt',map_location='cpu',weights_only=False)
    actor = baseline_actor().to(args.device)
    actor.load_state_dict(state['actor'])
    policy = LivePolicy(actor,state['identity']['normalizer'],args.device)
    observer.mark(phase='engineering_fresh_prefix')
    reference = None
    prefix = []
    for repeat in range(3):
        noise.seed,noise.tick = 60000,0
        adapter.reset(seed=60000)
        data = capture(adapter,noise)
        if repeat==0:
            raw_inputs=Path('checkpoints/20261023T104759Z')/(artifact_name+'_inputs.pt')
            plan_a=policy(adapter.observation)
            plan_b=policy(adapter.observation)
            torch.save(dict(observation=adapter.observation,plan=plan_a),raw_inputs)
            result.update(raw_inputs=str(raw_inputs),same_input_action_max_error=float(np.max(abs(plan_a-plan_b))))
        result['observations'].append(data)
        if reference is None: reference = data
        result['comparisons'].append(dict(kind='reset',repeat=repeat,
            max_errors={k:float(np.max(np.abs(v-reference[k]))) for k,v in data.items()}))
        save(f'reset_{repeat}_measured')
        if repeat==0 and args.reference:
            ref=json.loads(Path(args.reference).read_text())
            prefix=[np.asarray(a,dtype=np.float32) for a in ref['prefix_actions']]
            for action in prefix:
                with ledger.using('prefix'):adapter.step(action[None])
        elif repeat == 0:
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
        if args.only_prefix:
            result['prefix_actions']=prefix
            break
        if args.fresh_full:
            result['prefix_actions']=prefix
            result['checkpoint_observation']={k:v for k,v in reached.items()}
            candidate=(np.asarray(ref['candidate_action'],dtype=np.float32) if args.reference else policy(adapter.observation)[0])
            result['candidate_action']=candidate
            actual=continuation(adapter,policy,candidate,ledger)
            raw=Path('checkpoints/20261023T104759Z')/(artifact_name+'.pt')
            torch.save(actual,raw)
            result.update(raw_trajectory=str(raw),returns=actual['returns'],
                          success=actual['success'],steps=actual['steps'])
            break
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
    if 'observer' in globals(): observer.close()
    if 'noise' in globals(): noise.close()
    if 'adapter' in globals(): adapter.close()
    ledger.close()
    faulthandler.cancel_dump_traceback_later()
    app.close()
