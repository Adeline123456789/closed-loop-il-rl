"""Pure fidelity-decision tests; no simulated task outcome is claimed."""
import numpy as np
from forcecritic.diagnostics.live import compare


def test_repeatable_but_biased_restore_fails_original_comparison():
    original = {'trajectory':{k:np.zeros((4,8)) for k in ('position_m','rotation_matrix','joint_rad','gripper_m','gripper_velocity_m_s','linear_velocity_m_s','angular_velocity_rad_s','joint_velocity_rad_s','force_N','torque_Nm','controller')},
                'returns':np.array([1.,2.]), 'success':False, 'steps':4}
    repeated = {'trajectory':{k:np.ones((4,8))*.02 for k in ('position_m','rotation_matrix','joint_rad','gripper_m','gripper_velocity_m_s','linear_velocity_m_s','angular_velocity_rad_s','joint_velocity_rad_s','force_N','torque_Nm','controller')},
                'returns':np.array([1.,2.]), 'success':False, 'steps':4}
    tolerance = {k:.01 for k in (*original['trajectory'],'scaled_return','scaled_cost','success','steps')}
    assert compare(repeated,repeated,tolerance,[1,1])['passed']
    assert not compare(original,repeated,tolerance,[1,1])['passed']


def test_noise_schedule_keys_physical_time_not_number_of_observation_calls():
    import torch
    from types import SimpleNamespace
    from forcecritic.diagnostics.live import NoiseSchedule
    ledger=SimpleNamespace(counts={'prefix':0},num_envs=1)
    raw=SimpleNamespace(device='cpu',_compute_intermediate_values=lambda:torch.randn(3),
        event_manager=SimpleNamespace(apply=lambda mode:torch.randn(17)))
    noise=NoiseSchedule(SimpleNamespace(raw=raw,physics_ledger=ledger,reset=lambda seed=None:None),17)
    first=raw._compute_intermediate_values()
    raw.event_manager.apply(mode='interval')
    assert torch.equal(first,raw._compute_intermediate_values())
    ledger.counts['prefix']+=1
    assert not torch.equal(first,raw._compute_intermediate_values())
    noise.tick=0  # Restore logical physical time, not the global execution counter.
    assert torch.equal(first,raw._compute_intermediate_values())
    noise.close()


def test_diagnostic_reset_clears_actuator_velocity_and_restarts_event_clock():
    import torch
    from types import SimpleNamespace
    from forcecritic.diagnostics.live import NoiseSchedule
    calls=[]
    robot=SimpleNamespace(data=SimpleNamespace(joint_vel_target=torch.ones(1,9),
        joint_effort_target=torch.ones(1,9)),
        set_joint_velocity_target=lambda v:calls.append(('velocity',v.clone())),
        set_joint_effort_target=lambda v:calls.append(('effort',v.clone())))
    raw=SimpleNamespace(device='cpu',_robot=robot,_compute_intermediate_values=lambda:None,
        event_manager=SimpleNamespace(apply=lambda **kwargs:None,reset=lambda:calls.append(('event_reset',None))))
    adapter=SimpleNamespace(raw=raw,physics_ledger=SimpleNamespace(counts={'prefix':8},num_envs=1),
        reset=lambda seed=None:calls.append(('reset',seed)))
    old=adapter.reset
    noise=NoiseSchedule(adapter,1)
    adapter.reset(seed=7)
    assert [c[0] for c in calls]==['velocity','effort','reset','event_reset']
    assert not calls[0][1].any() and not calls[1][1].any()
    assert noise.seed==7 and noise.tick==0
    noise.close()
    assert adapter.reset is old
