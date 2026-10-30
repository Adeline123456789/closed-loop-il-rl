"""Lifecycle regression tests; these are NOT robot experiments."""
from types import SimpleNamespace
import numpy as np
import torch
import pytest
from forcecritic.envs.forge import ForgeAdapter


class Camera:
    def __init__(self):
        self.data = SimpleNamespace(output={'rgb': torch.zeros(1, 4, 4, 3, dtype=torch.uint8)})
    def update(self, *args, **kwargs): pass
    def set_world_poses_from_view(self, *args): pass


class LifecycleFixture:
    def __init__(self):
        self.unwrapped = self
        self.num_envs = 1
        self.device = 'cpu'
        self.step_dt = 1 / 15
        self.cfg = SimpleNamespace(decimation=8)
        self.cfg_task = SimpleNamespace(success_threshold=.01)
        self.camera = Camera()
        self.scene = SimpleNamespace(sensors={'diagnostic_camera': self.camera}, env_origins=torch.zeros(1, 3))
        self.sim = SimpleNamespace(render=lambda: None)
        self._robot = SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1, 9)))
        self.noisy_fingertip_pos = torch.zeros(1, 3)
        self.noisy_fingertip_quat = torch.tensor([[1., 0, 0, 0]])
        self.ee_linvel_fd = self.ee_angvel_fd = torch.zeros(1, 3)
        self.force_sensor_world = torch.zeros(1, 6)
        self.force_sensor_world_smooth = torch.zeros(1, 6)
        self.force_sensor_smooth = torch.ones(1, 6) * 999
        self.actions = torch.zeros(1, 7)
        self.ema_factor = torch.ones(1, 1) * .1
        self.reset_terminated = torch.tensor([False])
        self.reset_time_outs = torch.tensor([False])
        self.ending = True
    def _get_rewards(self): return torch.tensor([999.])
    def generate_ctrl_signals(self, **kwargs): pass
    def _get_curr_successes(self, *args, **kwargs): return torch.tensor([False])
    def _get_factory_rew_dict(self, _):
        return {k: torch.ones(1) for k in ('kp_baseline', 'kp_coarse', 'kp_fine', 'curr_engaged')}, {}
    def reset(self, seed=None): return {'policy': torch.zeros(1, 24)}, {}
    def step(self, action):
        self.actions = .1 * action + .9 * self.actions
        self.force_sensor_world[:] = 2
        self.force_sensor_world_smooth[:] = 1
        self.generate_ctrl_signals(ctrl_target_fingertip_midpoint_pos=torch.ones(1, 3))
        self.reset_time_outs[:] = self.ending
        self.camera.data.output['rgb'][:] = 17
        reward = self._get_rewards()
        if self.ending:
            self.force_sensor_world_smooth[:] = 100
            self.camera.data.output['rgb'][:] = 222
        return {'policy': torch.zeros(1, 24)}, reward, self.reset_terminated, self.reset_time_outs, {}
    def close(self): pass


def test_final_observation_is_before_reset_and_geometry_reward_is_separate():
    env = LifecycleFixture()
    adapter = ForgeAdapter(env, limits=[1] * 6)
    adapter.reset()
    row = adapter.step(np.ones((1, 4)) * .5)
    assert row['observation'][0]['rgb'].max() == 17
    assert adapter.observation[0]['rgb'].min() == 222
    assert row['frame']['wrench'].max() == 1
    assert adapter.current_frame['wrench'].min() == 100
    assert row['reward'][0] == 4  # not success-prediction reward 999
    assert row['cost'][0] == 6  # raw six-axis substep cost, not smoothed force
    assert row['ema_error'] < 1e-7
    assert np.allclose(row['submitted'][0], [.5, .5, .5, 0, 0, .5, -1])
    assert np.all(adapter.current_frame['previous_command'] == 0)
    adapter.close()


def test_history_contains_submitted_command_and_real_six_axis_sensor():
    env = LifecycleFixture()
    env.ending = False
    adapter = ForgeAdapter(env)
    adapter.reset()
    row = adapter.step(np.ones((1, 4)) * .3)
    assert row['cost'] is None  # uncalibrated is never silently zero-cost
    assert np.allclose(row['observation'][0]['history'][-1, -4:], .3)
    assert row['observation'][0]['history'][-1, :6].tolist() == [1.] * 6
    assert len(row['targets']) == 1


def test_uncalibrated_chunk_rejected_before_any_physics_charge():
    from forcecritic.envs.contracts import StepLedger
    env = LifecycleFixture()
    adapter = ForgeAdapter(env)
    ledger = StepLedger(10)
    with pytest.raises(ValueError, match='before interaction'):
        adapter.chunk(np.zeros((1,4,4)), ledger)
    assert ledger.total == 0
    assert adapter.control_steps == 0


def test_hard_protection_overrides_target_using_raw_substep_load():
    env = LifecycleFixture()
    env.ending = False
    env.fingertip_midpoint_pos = torch.tensor([[.1,.2,.3]])
    env.fingertip_midpoint_quat = torch.tensor([[1.,0,0,0]])
    adapter = ForgeAdapter(env, limits=[1]*6, hard_limits=[1.5]*6)
    adapter.reset()
    row = adapter.step(np.ones((1,4)))
    assert row['unsafe'].tolist() == [True]
    assert row['cost'].tolist() == [7.]  # six raw overload components + unsafe penalty
    assert np.allclose(row['targets'][0]['ctrl_target_fingertip_midpoint_pos'], [[.1,.2,.3]])
    adapter.reset()
    assert not adapter.protected.any()
