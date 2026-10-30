"""Live FORGE adapter. Import/create only AFTER Isaac's AppLauncher.

The stock controller, physics and randomizations are retained. Instrumentation
captures observations before DirectRLEnv's automatic reset. The geometry reward
is separated from the stock success-prediction and contact penalties.
"""
from collections import deque
from contextlib import nullcontext
import numpy as np
import torch

from .contracts import native_to_stock, contact_cost
from .observations import ObservationHistory

TASK = 'Isaac-Forge-PegInsert-Direct-v0'
CAMERA = dict(eye=[.95, .45, .45], target=[.6, 0., .12], width=256, height=256,
              focal_length=24., horizontal_aperture=20.955, clipping_range=[.1, 10.])


def numpy(value):
    return value.detach().cpu().numpy().copy()


def configure_camera(cfg):
    from isaaclab.sensors import CameraCfg
    import isaaclab.sim as sim_utils
    cfg.sim.render_interval = cfg.decimation
    cfg.scene.clone_in_fabric = False  # Camera enumerates USD, not Fabric-only clones.
    cfg.scene.diagnostic_camera = CameraCfg(
        prim_path='{ENV_REGEX_NS}/DiagnosticCamera', update_period=0.,
        height=CAMERA['height'], width=CAMERA['width'], data_types=['rgb'],
        spawn=sim_utils.PinholeCameraCfg(focal_length=CAMERA['focal_length'],
            horizontal_aperture=CAMERA['horizontal_aperture'],
            clipping_range=tuple(CAMERA['clipping_range'])))
    return cfg


class ForgeAdapter:
    """Batched native four-axis interface with explicit pre-reset final frames.

    Without accepted physical limits this is an audit/collection interface only;
    cost remains None, so it cannot silently produce legal critic replay.
    """
    def __init__(self, env, *, limits=None, hard_limits=None, physics_ledger=None, diagnostic=False):
        self.env = env
        self.raw = env.unwrapped
        self.n = self.raw.num_envs
        self.camera = self.raw.scene.sensors['diagnostic_camera']
        self.limits = None if limits is None else np.asarray(limits, dtype=float)
        self.hard_limits = None if hard_limits is None else np.asarray(hard_limits, dtype=float)
        if self.hard_limits is not None:
            contact_cost(np.zeros(6), self.hard_limits)
            if self.limits is None or (self.hard_limits < self.limits).any():
                raise ValueError('Hard protection limits must be no lower than contact-cost thresholds')
        self.protected = np.zeros(self.n, dtype=bool)
        self._stepping = False
        self.physics_ledger = physics_ledger
        self.diagnostic = diagnostic
        if limits is not None:
            contact_cost(np.zeros(6), self.limits)  # validate before any interaction
        self.history = [ObservationHistory() for _ in range(self.n)]
        self.previous = np.zeros((self.n, 4), dtype=np.float32)
        self.success_count = np.zeros(self.n, dtype=int)
        self.success_paid = np.zeros(self.n, dtype=bool)
        self.ever_success = np.zeros(self.n, dtype=bool)
        self.control_steps = 0
        self.capture = None
        self.targets = deque(maxlen=int(self.raw.cfg.decimation))
        self.substep_wrench = deque(maxlen=int(self.raw.cfg.decimation) + 1)
        self._reward = self.raw._get_rewards
        self._control = self.raw.generate_ctrl_signals
        self._reset_idx = self.raw._reset_idx if hasattr(self.raw, '_reset_idx') else None
        self.raw._get_rewards = self._before_reset
        self.raw.generate_ctrl_signals = self._capture_control
        if self._reset_idx is not None:
            self.raw._reset_idx = self._counted_reset

    def _counted_reset(self, *args, **kwargs):
        ids = args[0] if args else kwargs['env_ids']
        if len(ids) not in (0, self.n):
            raise RuntimeError('Pinned FORGE reset mutates whole-batch controller memory; partial reset rejected')
        context = self.physics_ledger.using('restore') if self.physics_ledger else nullcontext()
        stepping = self._stepping
        self._stepping = False
        try:
            with context:
                return self._reset_idx(*args, **kwargs)
        finally:
            self._stepping = stepping

    def _capture_control(self, *args, **kwargs):
        if hasattr(self.raw, 'force_sensor_world'):
            wrench = numpy(self.raw.force_sensor_world)
            self.substep_wrench.append(wrench)
            if self._stepping and self.hard_limits is not None:
                self.protected |= (~np.isfinite(wrench).all(-1) | (np.abs(wrench)>self.hard_limits).any(-1))
                mask = torch.as_tensor(self.protected, device=self.raw.device)[:,None]
                for key, actual in [('ctrl_target_fingertip_midpoint_pos', self.raw.fingertip_midpoint_pos),
                                    ('ctrl_target_fingertip_midpoint_quat', self.raw.fingertip_midpoint_quat)]:
                    if key in kwargs:
                        kwargs[key] = torch.where(mask, actual, kwargs[key])
        trace = {k: numpy(v) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        if hasattr(self.raw, 'fixed_pos_obs_frame'):
            trace.update(measured_eef_pos=numpy(self.raw.fingertip_midpoint_pos),
                measured_eef_quat=numpy(self.raw.fingertip_midpoint_quat),
                noisy_fixed_origin=numpy(self.raw.fixed_pos_obs_frame+self.raw.init_fixed_pos_obs_noise),
                ema_action=numpy(self.raw.actions), pos_threshold=numpy(self.raw.pos_threshold),
                rot_threshold=numpy(self.raw.rot_threshold))
        self.targets.append(trace)
        return self._control(*args, **kwargs)

    def frame(self):
        r = self.raw
        self.camera.update(r.step_dt, force_recompute=True)
        # No true hole/peg pose, task geometry or stock privileged critic input.
        gripper = r._robot.data.joint_pos[:, -2:].sum(-1, keepdim=True)
        return dict(rgb=numpy(self.camera.data.output['rgb'][..., :3]),
            state=numpy(torch.cat((r.noisy_fingertip_pos, r.noisy_fingertip_quat, gripper), -1)),
            # The PhysX 5.5.1 API reports the incoming JOINT frame, despite
            # FORGE's variable name "world". Keep the actual sensor frame;
            # do not apply FORGE's hole-origin translation to this quantity.
            wrench=numpy(r.force_sensor_world_smooth),
            motion=numpy(torch.cat((r.ee_linvel_fd, r.ee_angvel_fd), -1)),
            previous_command=self.previous.copy())

    def controller(self):
        """Known controller memory only; excludes true task/asset geometry."""
        fields = ('actions', 'ema_factor', 'pos_threshold', 'rot_threshold',
                  'task_prop_gains', 'dead_zone_thresholds')
        return np.concatenate([numpy(getattr(self.raw, k)).reshape(self.n, -1) for k in fields], -1)

    def _observations(self, frame, ids=None):
        ids = range(self.n) if ids is None else ids
        observations = {}
        for i in ids:
            self.history[i].append(frame['wrench'][i], frame['motion'][i], frame['previous_command'][i])
            observations[i] = self.history[i].observation(frame['rgb'][i], frame['state'][i])
        return observations

    def _before_reset(self):
        # DirectRLEnv invokes this after _get_dones and before _reset_idx.
        stock_reward = self._reward()
        r = self.raw
        geometric = numpy(r._get_curr_successes(r.cfg_task.success_threshold, check_rot=False)).astype(bool)
        self.success_count = np.where(geometric, self.success_count + 1, 0)
        self.ever_success |= geometric
        stable = self.success_count >= 3
        terms, _ = r._get_factory_rew_dict(torch.as_tensor(geometric, device=r.device))
        reward = sum(numpy(terms[k]) for k in ('kp_baseline', 'kp_coarse', 'kp_fine', 'curr_engaged'))
        reward += (stable & ~self.success_paid).astype(float)
        self.success_paid |= stable
        frame = self.frame()
        self.substep_wrench.append(numpy(r.force_sensor_world))
        samples = np.stack(self.substep_wrench)
        if not np.isfinite(samples).all():
            raise FloatingPointError('Nonfinite six-axis physical wrench')
        if self.hard_limits is not None:
            self.protected |= (np.abs(samples)>self.hard_limits).any(axis=(0,2))
        self.capture = dict(frame=frame, observation=self._observations(frame), reward=reward,
            stock_reward_audit_only=numpy(stock_reward),
            cost=None if self.limits is None else contact_cost(samples, self.limits).max(0)+self.protected,
            unsafe=self.protected.copy(),
            geometric_success=geometric, success=stable, ever_success=self.ever_success.copy(),
            terminated=numpy(r.reset_terminated), truncated=numpy(r.reset_time_outs),
            ema_action=numpy(r.actions), ema_factor=numpy(r.ema_factor),
            raw_wrench=numpy(r.force_sensor_world), smoothed_raw_wrench=numpy(r.force_sensor_world_smooth),
            stock_transformed_wrench=numpy(r.force_sensor_smooth), substep_wrench=samples,
            targets=list(self.targets))
        if hasattr(r, 'episode_length_buf'):
            self.capture['remaining'] = numpy((r.max_episode_length - 1 - r.episode_length_buf).clamp_min(0))
            self.capture['controller'] = self.controller()
        if self.diagnostic:
            from isaaclab.utils.math import matrix_from_quat
            # Keep units separate. Rotation matrices are invariant to q vs -q.
            self.capture['physical'] = {
                'position_m': numpy(torch.cat((r._robot.data.body_pos_w.flatten(1),
                    r._held_asset.data.root_pos_w,r._fixed_asset.data.root_pos_w),-1)),
                'rotation_matrix': numpy(torch.cat((matrix_from_quat(r._robot.data.body_quat_w).flatten(1),
                    matrix_from_quat(r._held_asset.data.root_quat_w).flatten(1),
                    matrix_from_quat(r._fixed_asset.data.root_quat_w).flatten(1)),-1)),
                'joint_rad': numpy(r._robot.data.joint_pos[:,:7]),
                'gripper_m': numpy(r._robot.data.joint_pos[:,7:]),
                'linear_velocity_m_s': numpy(torch.cat((r._robot.data.body_lin_vel_w.flatten(1),
                    r._held_asset.data.root_lin_vel_w,r._fixed_asset.data.root_lin_vel_w),-1)),
                'angular_velocity_rad_s': numpy(torch.cat((r._robot.data.body_ang_vel_w.flatten(1),
                    r._held_asset.data.root_ang_vel_w,r._fixed_asset.data.root_ang_vel_w),-1)),
                'joint_velocity_rad_s': numpy(r._robot.data.joint_vel[:,:7]),
                'gripper_velocity_m_s': numpy(r._robot.data.joint_vel[:,7:])}
        # The returned environment reward also excludes action-7 prediction and
        # stock contact penalties; the original value is retained only for audit.
        return torch.as_tensor(reward, dtype=stock_reward.dtype, device=r.device)

    def reset(self, seed=None):
        self.protected[:] = False
        self.previous[:] = 0
        self.success_count[:] = 0
        self.success_paid[:] = False
        self.ever_success[:] = False
        for h in self.history:
            h.reset()
        expert_obs, _ = self.env.reset(seed=seed)
        origins = self.raw.scene.env_origins
        self.camera.set_world_poses_from_view(
            origins + torch.tensor(CAMERA['eye'], device=self.raw.device),
            origins + torch.tensor(CAMERA['target'], device=self.raw.device))
        self.raw.sim.render()
        self.current_frame = self.frame()
        self.observation = self._observations(self.current_frame)
        self.expert_observation = expert_obs
        return self.observation

    def step(self, native):
        native = np.asarray(native, dtype=np.float32)
        if native.shape != (self.n, 4):
            raise ValueError('one native four-axis command per environment required')
        stock = native_to_stock(native)
        old_ema = numpy(self.raw.actions)
        alpha = numpy(self.raw.ema_factor)
        self.previous = stock[:, [0, 1, 2, 5]].copy()
        self.targets.clear()
        self.substep_wrench.clear()
        self.capture = None
        self._stepping = True
        try:
            expert_obs, _, _, _, _ = self.env.step(torch.as_tensor(stock, device=self.raw.device))
        finally:
            self._stepping = False
        self.control_steps += self.n
        if self.capture is None:
            raise RuntimeError('FORGE pre-reset capture hook was not called')
        batch = self.capture
        batch.update(submitted=stock, previous_ema=old_ema,
                     ema_error=float(np.max(np.abs(batch['ema_action'] - (alpha * stock + (1-alpha) * old_ema)))))
        ended = batch['terminated'] | batch['truncated']
        if 'controller' in batch and not ended.all():
            batch['controller'][~ended] = self.controller()[~ended]
        self.observation = batch['observation'].copy()
        self.current_frame = batch['frame']
        if ended.any():
            ids = np.flatnonzero(ended)
            self.previous[ids] = 0
            self.success_count[ids] = 0
            self.success_paid[ids] = False
            self.ever_success[ids] = False
            self.protected[ids] = False
            for i in ids:
                self.history[i].reset()
            reset_frame = self.frame()
            self.observation.update(self._observations(reset_frame, ids))
            # Before-action recording must match the reset state for these rows.
            self.current_frame = {k: v.copy() for k, v in batch['frame'].items()}
            for k in self.current_frame:
                self.current_frame[k][ids] = reset_frame[k][ids]
        self.expert_observation = expert_obs
        return batch

    def chunk(self, plan, ledger, category='interaction', gamma=.99):
        if self.limits is None:
            raise ValueError('critic collection requires accepted physical limits before interaction')
        plan = np.asarray(plan, dtype=np.float32)
        if plan.shape != (self.n, 4, 4):
            raise ValueError('expected N x K=4 x native=4 complete chunks')
        valid = np.zeros((self.n, 4), dtype=bool)
        returns = np.zeros((self.n, 2))
        active = np.ones(self.n, dtype=bool)
        next_obs = {}
        terminal = np.zeros(self.n, dtype=bool)
        records = []
        for k in range(4):
            ledger.charge(category, self.n)
            row = self.step(plan[:, k])
            if row['cost'] is None:
                raise ValueError('critic collection requires accepted physical limits')
            valid[:, k] = active
            returns[active] += gamma**k * np.stack((row['reward'], row['cost']), -1)[active]
            for i in np.flatnonzero(active):
                next_obs[i] = row['observation'][i]  # pre-reset, including terminal RGB
            done = row['terminated'] | row['truncated'] | row['success'] | row['unsafe']
            terminal |= active & (row['terminated'] | row['success'] | row['unsafe'])
            active &= ~done
            records.append(row)
            if not active.any():
                break
        lengths = valid.sum(-1)
        return dict(plan=plan.copy(), valid=valid, k=lengths, returns=returns,
                    gamma_k=gamma**lengths, bootstrap=~terminal, next_observation=next_obs, records=records)

    def close(self):
        self.raw._get_rewards = self._reward
        self.raw.generate_ctrl_signals = self._control
        if self._reset_idx is not None:
            self.raw._reset_idx = self._reset_idx
        self.env.close()
