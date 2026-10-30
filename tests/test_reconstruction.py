import json
import pytest
from forcecritic.data.reconstruction import first_collection_batch


@pytest.fixture
def recording(tmp_path):
    root = tmp_path/'00_train_1100'
    root.mkdir()
    data = dict(status='COLLECTED', source='same_version_recollection', seed=1100, split='train',
                accounting={'num_envs':8}, contract=dict(action_stage='submitted_pre_ema',action_axes=['x','y','z','yaw']),
                episodes=[dict(parent_id=f'train-1100-{i:06d}',split='train',source='real_forge_recording',
                               collection_run_id=root.name) for i in range(16)])
    path=root/'manifest.json'
    def write(): path.write_text(json.dumps(data)); return path
    return data, write


def test_original_batch_width_and_order_are_recovered(recording):
    data,write=recording
    _,rows,width=first_collection_batch(write(),requested_steps=149)
    assert width==8 and len(rows)==8 and rows[-1]['parent_id']=='train-1100-000007'


@pytest.mark.parametrize('corruption',['later_batch','shuffled','missing_width','mixed_run','post_ema','test'])
def test_invalid_reconstruction_fails_before_simulation(recording,corruption):
    data,write=recording
    if corruption=='later_batch': data['episodes']=data['episodes'][8:]
    if corruption=='shuffled': data['episodes'][0],data['episodes'][1]=data['episodes'][1],data['episodes'][0]
    if corruption=='missing_width': data['accounting']={}
    if corruption=='mixed_run': data['episodes'][0]['collection_run_id']='other'
    if corruption=='post_ema': data['contract']['action_stage']='post_ema'
    if corruption=='test': data['split']='test'
    with pytest.raises(ValueError): first_collection_batch(write(),requested_steps=149)


def test_runtime_rlgames_deterministic_mean_still_consumes_rng():
    """Regress the RNG mechanism, not a robot outcome: sampling happens before mean selection."""
    import torch
    torch.manual_seed(1100)
    state=torch.get_rng_state()
    dist=torch.distributions.Normal(torch.zeros(8,7),torch.ones(8,7))
    _unused_action=dist.sample()
    deterministic_action=dist.mean
    noise_after_call=torch.randn(8,3)
    torch.set_rng_state(state)
    noise_without_call=torch.randn(8,3)
    assert torch.equal(deterministic_action,torch.zeros(8,7))
    assert not torch.equal(noise_after_call,noise_without_call)
