"""State-write regression tests; real contact evidence lives in the probe reports."""
from types import SimpleNamespace
import pytest
import torch
from forcecritic.diagnostics.live import write_changed_scene_state, clone_tensor_layout, restore


def setup_scene():
    current = {k:torch.ones(1,n) for k,n in
               [('root_pose',7),('root_velocity',6),('joint_position',9),('joint_velocity',9)]}
    calls = []
    asset = SimpleNamespace()
    for key,method in [('root_pose','write_root_pose_to_sim'),('root_velocity','write_root_velocity_to_sim'),
                       ('joint_position','write_joint_position_to_sim'),('joint_velocity','write_joint_velocity_to_sim')]:
        setattr(asset,method,lambda value,k=key:calls.append((k,value.clone())))
    scene = SimpleNamespace(articulations={'robot':asset},get_state=lambda:{'articulation':{'robot':current}})
    saved = {'articulation':{'robot':{k:v.clone() for k,v in current.items()}}}
    return scene,saved,calls


def test_identity_restore_does_not_reset_any_physics_state():
    scene,saved,calls=setup_scene()
    assert write_changed_scene_state(scene,saved)==[]
    assert calls==[]


def test_velocity_only_restore_does_not_touch_pose_or_other_state():
    scene,saved,calls=setup_scene()
    saved['articulation']['robot']['joint_velocity'][0,0]=2
    assert write_changed_scene_state(scene,saved)==['robot/joint_velocity']
    assert len(calls)==1 and calls[0][0]=='joint_velocity'
    assert torch.equal(calls[0][1],saved['articulation']['robot']['joint_velocity'])


def test_unknown_entities_fail_before_state_writes():
    scene,saved,calls=setup_scene()
    saved['rigid_object']={'extra':{}}
    with pytest.raises(ValueError,match='Unimplemented'): write_changed_scene_state(scene,saved)
    assert not calls


def test_legacy_layout_losing_snapshot_is_rejected_before_touching_simulator():
    with pytest.raises(ValueError,match='Legacy snapshot lost tensor layouts'):
        restore(None,None,{},None)


@pytest.mark.parametrize('layout',['mass_matrix','transpose','broadcast','scalar','empty'])
def test_snapshot_preserves_values_and_controller_tensor_layout(layout):
    base=torch.arange(81,dtype=torch.float32).reshape(1,9,9)
    values={'mass_matrix':base[:,:7,:7], 'transpose':base.transpose(1,2),
            'broadcast':base[:,0:1,:].expand(3,4,9), 'scalar':torch.tensor(2.),
            'empty':torch.empty(1,0)}
    original=values[layout]
    saved=clone_tensor_layout(original)
    restored=clone_tensor_layout(saved)
    assert saved.stride()==restored.stride()==original.stride()
    assert torch.equal(original,saved) and torch.equal(original,restored)
    if original.numel():
        assert saved.untyped_storage().data_ptr()!=original.untyped_storage().data_ptr()
        assert restored.untyped_storage().data_ptr()!=saved.untyped_storage().data_ptr()
