"""Fixed-command, substep-level real FORGE evidence; not scientific D0 admission."""
import argparse
import copy
import json
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output', required=True)
parser.add_argument('--variant', choices=['legacy', 'skip_properties', 'fresh'], default='legacy')
parser.add_argument('--steps', type=int, default=16)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--render-profile', choices=['stock','rtpt','rtpt_forced'], default='stock')
parser.add_argument('--control-step-cap', type=int)
parser.add_argument('--mutation', default='none', choices=['none','forward','render','robot_position',
    'robot_velocity','held_pose','held_velocity','fixed_pose','robot_root','scene','scene_no_forward',
    'mass_clone','tensor_clones'])
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app

import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import numpy as np
import torch
from forcecritic.envs.forge import ForgeAdapter, configure_camera, TASK, numpy
from forcecritic.envs.accounting import PhysicsLedger
from forcecritic.training.baseline import accepted_limits
from forcecritic.diagnostics.live import NoiseSchedule, snapshot, restore
from forcecritic.diagnostics.progress import DiagnosticProgress
from forcecritic.diagnostics.artifacts import atomic_json
from forcecritic.runtime import digest, source_hashes

out = Path(args.output)
out.mkdir(parents=True, exist_ok=False)
plan_path = Path('configs/replay_root_cause_20261027.json')
plan = json.loads(plan_path.read_text())
source = Path(plan['fixed_action_source'])
reference = json.loads(source.read_text())
prefix = np.asarray(reference['prefix_actions'], dtype=np.float32)
candidate = np.asarray(reference.get('candidate_action', prefix[-4:]), dtype=np.float32)
commands = np.tile(candidate, (int(np.ceil(args.steps / len(candidate))), 1))[:args.steps]
cfg = configure_camera(parse_env_cfg(TASK, device=args.device, num_envs=1))
cfg.seed = 0
render_revision = None
if args.render_profile != 'stock':
    render_revision = json.loads(Path('configs/replay_render_rtpt_v1.json').read_text())
    cfg.sim.render.carb_settings = render_revision['settings']
cap = args.control_step_cap or plan['per_process_control_step_cap']
if not 1 <= cap <= plan['per_process_control_step_cap']:
    raise ValueError('Process cap exceeds registered budget')
ledger = PhysicsLedger(1, cfg.decimation, cap)
raw_dir = Path('checkpoints/replay_root_cause_20261027') / out.name
observer = DiagnosticProgress(ledger, out, raw_dir)
result = dict(status='RUNNING', protocol=plan['protocol'], plan_sha256=digest(plan_path),
              variant=args.variant, render_profile=args.render_profile, render_revision=render_revision,
              mutation=args.mutation,
              seed=plan['seed'], source=str(source), source_sha256=digest(source),
              source_hashes=source_hashes(), fixed_commands=commands.tolist(), comparisons=[])
records = {}
active = None
traces = []

def save():
    result['accounting'] = ledger.record()
    atomic_json(out/'result.json', result)

def capture():
    r = adapter.raw
    values = {}
    for name, asset in [('robot', r._robot), ('held', r._held_asset), ('fixed', r._fixed_asset)]:
        view = asset.root_physx_view
        for key in ('root_transforms', 'root_velocities', 'dof_positions', 'dof_velocities',
                    'link_transforms', 'link_velocities', 'jacobians', 'generalized_mass_matrices',
                    'link_incoming_joint_force'):
            values[name+'/'+key] = numpy(getattr(view, 'get_'+key)())
        for key in ('pos', 'vel', 'effort'):
            values[name+'/target_'+key] = numpy(getattr(asset.data, 'joint_'+key+'_target'))
    for key in ('joint_pos', 'joint_vel', 'fingertip_midpoint_pos', 'fingertip_midpoint_quat',
                'fingertip_midpoint_linvel', 'fingertip_midpoint_angvel', 'arm_mass_matrix',
                'fingertip_midpoint_jacobian', 'actions', 'task_prop_gains', 'task_deriv_gains',
                'dead_zone_thresholds', 'force_sensor_world_smooth', 'prev_fingertip_pos',
                'prev_fingertip_quat', 'prev_joint_pos'):
        values['controller/'+key] = numpy(getattr(r, key))
    values['noise_tick'] = np.asarray(noise.tick)
    return values

def errors(a, b):
    return {k: float(np.max(np.abs(v.astype(float)-b[k].astype(float)))) if v.size else 0.
            for k, v in a.items()}

def run_commands(name):
    global active, traces
    active, traces = name, []
    before = capture()
    rows = []
    for action in commands:
        with ledger.using('branch'):
            row = adapter.step(action[None])
        rows.append({k:copy.deepcopy(row[k]) for k in ('physical','substep_wrench','controller','reward','cost','success')})
        if row['terminated'][0] or row['truncated'][0] or row['success'][0] or row['unsafe'][0]:
            break
    active = None
    data = dict(before=before, traces=traces, rows=rows, observation=copy.deepcopy(adapter.observation))
    records[name] = data
    torch.save(data, raw_dir/(name+'.pt'))
    if name != 'original':
        orig = records['original']
        comparison = dict(name=name, before=errors(orig['before'], before),
                          physics_steps=[dict(before=errors(x['before'],y['before']),
                                              after=errors(x['after'],y['after']))
                                         for x,y in zip(orig['traces'],traces)],
                          original_substeps=len(orig['traces']), replay_substeps=len(traces))
        result['comparisons'].append(comparison)
    save()
    print('MEASURED', name, len(traces), flush=True)

try:
    save()
    env = gym.make(TASK, cfg=cfg)
    import carb.settings
    settings = carb.settings.get_settings()
    if args.render_profile == 'rtpt_forced':
        for key,value in render_revision['settings'].items(): settings.set(key,value)
    result['actual_render_settings'] = {k:settings.get(k) for k in
        ('/rtx/rendermode','/rtx/rtpt/cached/enabled','/rtx/rtpt/lightcache/cached/enabled')}
    acceptance = Path('reports/20261023T104759Z/p0_acceptance.json')
    adapter = ForgeAdapter(env, limits=accepted_limits(acceptance),
                           hard_limits=json.loads(acceptance.read_text())['physical_protection_limits'],
                           physics_ledger=ledger, diagnostic=True)
    result['setup_audit'] = {
        'physics_dt': cfg.sim.dt, 'decimation': cfg.decimation,
        'solver_type': cfg.sim.physx.solver_type,
        'max_position_iterations': cfg.sim.physx.max_position_iteration_count,
        'max_velocity_iterations': cfg.sim.physx.max_velocity_iteration_count,
        'gpu_max_num_partitions': cfg.sim.physx.gpu_max_num_partitions,
        'assets': {name: {'fixed_tendons': asset.num_fixed_tendons,
                          'spatial_tendons': asset.num_spatial_tendons,
                          'joints': asset.joint_names,
                          'dof_stiffness': numpy(asset.root_physx_view.get_dof_stiffnesses()).tolist(),
                          'dof_damping': numpy(asset.root_physx_view.get_dof_dampings()).tolist()}
                   for name,asset in adapter.raw.scene.articulations.items()}}
    noise = NoiseSchedule(adapter, plan['seed'])
    adapter.reset(seed=plan['seed'])
    for action in prefix:
        with ledger.using('prefix'):
            adapter.step(action[None])
    state = snapshot(adapter, noise)
    wrapped_step = ledger.original
    def measured_step(sim, *a, **kw):
        before = capture() if active else None
        value = wrapped_step(sim, *a, **kw)
        if active:
            traces.append(dict(before=before, after=capture()))
        return value
    ledger.original = measured_step
    # No elapsed physics: locate the first destructive component of a state
    # round trip, independently of rolling a future contact trajectory back.
    before_mutation = capture()
    result['tensor_layouts_before'] = {k:list(v.stride()) for k,v in vars(adapter.raw).items() if torch.is_tensor(v)}
    if args.mutation == 'forward':
        adapter.raw.sim.forward()
    elif args.mutation == 'render':
        adapter.raw.sim.render()
    elif args.mutation == 'mass_clone':
        adapter.raw.arm_mass_matrix = adapter.raw.arm_mass_matrix.clone()
    elif args.mutation == 'tensor_clones':
        for key,value in state['tensors'].items(): setattr(adapter.raw,key,value.clone())
    elif args.mutation.startswith('scene'):
        restore_state = {**state, 'properties':{k:{} for k in state['properties']}}
        forward = adapter.raw.sim.forward
        render = adapter.raw.sim.render
        if args.mutation == 'scene_no_forward':
            adapter.raw.sim.forward = lambda: None
            adapter.raw.sim.render = lambda: None
        try:
            result['restore_receipt'] = restore(adapter, noise, restore_state, ledger)
        finally:
            adapter.raw.sim.forward = forward
            adapter.raw.sim.render = render
    elif args.mutation != 'none':
        asset_name, field = args.mutation.split('_')
        asset = {'robot':adapter.raw._robot,'held':adapter.raw._held_asset,'fixed':adapter.raw._fixed_asset}[asset_name]
        if field == 'position': asset.write_joint_position_to_sim(asset.data.joint_pos.clone())
        elif field == 'velocity' and asset_name == 'robot': asset.write_joint_velocity_to_sim(asset.data.joint_vel.clone())
        elif field == 'velocity': asset.write_root_velocity_to_sim(asset.data.root_vel_w.clone())
        else: asset.write_root_pose_to_sim(asset.data.root_pose_w.clone())
    result['immediate_mutation_errors'] = errors(before_mutation,capture())
    result['tensor_layouts_after'] = {k:list(v.stride()) for k,v in vars(adapter.raw).items() if torch.is_tensor(v)}
    run_commands('original')
    if args.variant != 'fresh':
        for repeat in range(args.repeats):
            restored = state
            if args.variant == 'skip_properties':
                restored = {**state, 'properties':{k:{} for k in state['properties']}}
            restore(adapter, noise, restored, ledger)
            run_commands('restore_'+str(repeat))
        for repeat in range(args.repeats):
            adapter.reset(seed=plan['seed'])
            for action in prefix:
                with ledger.using('prefix'):
                    adapter.step(action[None])
            run_commands('prefix_'+str(repeat))
    result['status'] = 'MEASURED'
except BaseException as error:
    result.update(status='FAILED_IMPLEMENTATION', error=repr(error))
    raise
finally:
    save()
    if 'noise' in globals(): noise.close()
    observer.close()
    if 'adapter' in globals(): adapter.close()
    ledger.close()
    app.close()
