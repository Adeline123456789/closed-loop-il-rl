"""Measured post-launch action, camera and six-axis sensor contract audit."""
import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
parser.add_argument('--seed', type=int, default=1000)
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.headless or not args.enable_cameras:
    parser.error('--headless --enable_cameras required')
app = AppLauncher(args).app
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
from forcecritic.runtime import atomic_json
result = {'status': 'RUNNING', 'source': 'real_forge', 'seed': args.seed}
atomic_json(out / 'contract_audit.json', result)
try:
    import gymnasium as gym
    import numpy as np
    import torch
    from PIL import Image
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from forcecritic.envs.forge import ForgeAdapter, configure_camera, TASK, CAMERA
    from forcecritic.envs.accounting import PhysicsLedger
    cfg = configure_camera(parse_env_cfg(TASK, device=args.device, num_envs=1))
    cfg.seed = args.seed
    ledger = PhysicsLedger(1, cfg.decimation, 5000)
    env = gym.make(TASK, cfg=cfg)
    adapter = ForgeAdapter(env, physics_ledger=ledger)
    adapter.reset(seed=args.seed)
    hold = adapter.raw.actions[:, [0, 1, 2, 5]].detach().cpu().numpy().clip(-1, 1)
    traces = []
    for axis in range(4):
        for sign in (-1, 1):
            for repeat in range(3):
                action = hold.copy()
                action[:, axis] = np.clip(action[:, axis] + sign * .02, -1, 1)
                before_wrench = adapter.current_frame['wrench'].copy()
                with ledger.using('evaluation'):
                    row = adapter.step(action)
                traces.append(dict(axis=axis, sign=sign, repeat=repeat, before_wrench=before_wrench, **row))
    first = traces[0]['frame']['rgb'][0]
    Image.fromarray(first).save(out / 'camera.png')
    arrays = {key: np.stack([r[key] for r in traces]) for key in
              ('submitted', 'previous_ema', 'ema_action', 'ema_factor', 'raw_wrench',
               'smoothed_raw_wrench', 'stock_transformed_wrench', 'substep_wrench')}
    for key in ('state', 'motion', 'wrench', 'previous_command'):
        arrays[key] = np.stack([r['frame'][key] for r in traces])
    arrays['target_position'] = np.stack([
        r['targets'][-1]['ctrl_target_fingertip_midpoint_pos'] for r in traces])
    arrays['target_quaternion'] = np.stack([
        r['targets'][-1]['ctrl_target_fingertip_midpoint_quat'] for r in traces])
    position_errors, yaw_errors, force_filter_errors = [], [], []
    def yaw(q):
        w,x,y,z = np.moveaxis(q,-1,0)
        return np.mod(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)),2*np.pi)
    def controller_wrap(angle):
        return np.where(angle>np.deg2rad(235),angle-2*np.pi,angle)
    for row in traces:
        if len(row['substep_wrench']) != cfg.decimation+1:
            raise RuntimeError('Unexpected number of raw force samples at control-step boundaries')
        expected_wrench = row['before_wrench'].copy()
        for raw in row['substep_wrench'][1:]:
            expected_wrench = cfg.ft_smoothing_factor*raw+(1-cfg.ft_smoothing_factor)*expected_wrench
        force_filter_errors.append(float(np.abs(expected_wrench-row['smoothed_raw_wrench']).max()))
        for target in row['targets']:
            unclipped = target['noisy_fixed_origin'] + target['ema_action'][:,:3]*np.asarray(cfg.ctrl.pos_action_bounds)
            expected = target['measured_eef_pos'] + np.clip(unclipped-target['measured_eef_pos'],
                                                          -target['pos_threshold'],target['pos_threshold'])
            position_errors.append(float(np.abs(expected-target['ctrl_target_fingertip_midpoint_pos']).max()))
            bolt_yaw = -np.pi + (3*np.pi/2)*(target['ema_action'][:,5]*cfg.ctrl.rot_action_bounds[2]+1)/2
            # RotX(pi) * RotZ(bolt_yaw) gives world Euler yaw -bolt_yaw.
            desired_yaw = controller_wrap(np.mod(-bolt_yaw,2*np.pi))
            current_yaw = controller_wrap(yaw(target['measured_eef_quat']))
            expected_yaw = current_yaw+np.clip(desired_yaw-current_yaw,
                -target['rot_threshold'][:,2],target['rot_threshold'][:,2])
            error = yaw(target['ctrl_target_fingertip_midpoint_quat'])-expected_yaw
            yaw_errors.append(float(np.abs((error+np.pi)%(2*np.pi)-np.pi).max()))
    np.savez_compressed(out / 'action_sensor_trace.npz', **arrays)
    # Inspect authored sensor-joint anchors, without inferring them from a name.
    from pxr import UsdPhysics
    joints = []
    for prim in adapter.raw.sim.stage.Traverse():
        if prim.IsA(UsdPhysics.Joint):
            joint = UsdPhysics.Joint(prim)
            children = [str(x) for x in joint.GetBody1Rel().GetTargets()]
            if any(x.endswith('/force_sensor') for x in children):
                joints.append({'path': str(prim.GetPath()), 'body0': [str(x) for x in joint.GetBody0Rel().GetTargets()],
                    'body1': children, 'local_pos1': str(joint.GetLocalPos1Attr().Get()),
                    'local_rot1': str(joint.GetLocalRot1Attr().Get())})
    result.update(status='MEASURED_NOT_YET_ACCEPTED', task=TASK, camera=CAMERA,
        control_dt=adapter.raw.step_dt, physics_dt=adapter.raw.physics_dt,
        decimation=cfg.decimation, individual_control_steps=adapter.control_steps,
        max_ema_error=max(r['ema_error'] for r in traces),
        action_alignment=all(r['ema_error'] < 1e-6 for r in traces) and max(position_errors)<1e-6 and max(yaw_errors)<1e-5,
        max_position_target_error_m=max(position_errors),
        max_yaw_target_error_rad=max(yaw_errors),
        native_axes=['x', 'y', 'z', 'yaw'], action_stage='submitted_pre_ema',
        position_bounds=list(cfg.ctrl.pos_action_bounds), bolt_frame_yaw_range_radians=[-np.pi, np.pi/2],
        target_rotation_order='RotX(pi) * RotZ(bolt_frame_yaw)',
        camera_nonconstant=bool(first.max() > first.min()),
        sensor_finite=all(np.isfinite(arrays[k]).all() for k in ('raw_wrench','wrench')),
        raw_wrench_max_abs_per_axis=np.abs(arrays['raw_wrench']).max(axis=(0,1)).tolist(),
        force_filter_alpha=cfg.ft_smoothing_factor, force_filter_rate_hz=1/adapter.raw.physics_dt,
        max_force_filter_error=max(force_filter_errors),force_filter_alignment=max(force_filter_errors)<1e-4,
        wrench_dim=6, actor_wrench_source='force_sensor_world_smooth (variable name is misleading)',
        actor_wrench_frame='incoming joint frame at joint anchor; PhysX 5.5.1 API',
        sensor_joints=joints, physical_limits_status='UNRESOLVED',
        observation_timing='before_action', causal_observations=True,
        note='Requires sensor-frame/anchor review and physical limit calibration before P0 acceptance.')
    result['accounting'] = ledger.record()
    atomic_json(out / 'contract_audit.json', result)
    adapter.close()
except BaseException as error:
    import traceback
    traceback.print_exc()
    result.update(status='FAILED', error=repr(error))
    if 'ledger' in globals():
        result['accounting'] = ledger.record()
    atomic_json(out / 'contract_audit.json', result)
    raise
finally:
    app.close()
