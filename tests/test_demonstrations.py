"""Temporary software fixtures; never admitted as measured FORGE performance."""
import copy
import json
import numpy as np
import pytest
import torch
from torch import nn
from forcecritic.data.demonstrations import Demonstrations,file_hash
from forcecritic.training.pretrain import pretrain


def fixture_manifest(tmp_path):
    acceptance=tmp_path/'acceptance.json'
    acceptance.write_text(json.dumps(dict(status='PASSED',action_alignment=True,sensor_alignment=True,
                                         camera_alignment=True,causal_observations=True,scope='SOFTWARE_FIXTURE')))
    episodes=[]
    for parent,split,successful,offset in [('a','train',True,0),('b','test',True,100),('c','train',False,1000)]:
        action=np.arange(20,dtype=np.float32).reshape(5,4)/40
        data=dict(rgb=np.zeros((5,8,8,3),np.uint8),state=np.ones((5,8),np.float32)*offset,
                  wrench=np.repeat(np.arange(5)[:,None],6,axis=1).astype('float32'),
                  motion=np.zeros((5,6),np.float32),previous_command=np.vstack([np.zeros((1,4)),action[:-1]]).astype('float32'),action=action)
        files={}
        for key,value in data.items():
            path=tmp_path/f'{parent}_{key}.npy';np.save(path,value)
            files[key]={'path':path.name,'sha256':file_hash(path)}
        episodes.append(dict(parent_id=parent,split=split,successful=successful,source='real_forge_recording',files=files))
    contract=dict(task='Isaac-Forge-PegInsert-Direct-v0',action_stage='submitted_pre_ema',action_axes=['x','y','z','yaw'],
                  observation_timing='before_action',wrench_dim=6,history_steps=4,chunk_length=4,
                  acceptance_evidence=acceptance.name,acceptance_sha256=file_hash(acceptance))
    path=tmp_path/'manifest.json';path.write_text(json.dumps(dict(contract=contract,episodes=episodes)))
    return path


def test_demonstration_windows_are_causal_and_train_only(tmp_path):
    path=fixture_manifest(tmp_path);ds=Demonstrations(path)
    assert len(ds)==5 and [r['parent_id'] for r in ds.rows]==['a']
    assert ds.normalizer['state']['mean']==[0]*8  # excludes test and failure data
    obs,actions,valid=ds[1]
    mean=ds.normalizer['history']['mean'][0];std=ds.normalizer['history']['std'][0]
    torch.testing.assert_close(obs['history'][:,0],torch.tensor([(0-mean)/std]*3+[(1-mean)/std],dtype=torch.float32))
    _,a,mask=ds[4];assert mask.tolist()==[True,False,False,False]
    assert torch.count_nonzero(a[1:])==0
    with pytest.raises(ValueError,match='reuse training'):Demonstrations(path,split='test')
    val=Demonstrations(path,split='test',normalizer=ds.normalizer)
    assert val.rows[0]['parent_id']=='b'
    broken=copy.deepcopy(ds.normalizer);broken['training_parents']=['b']
    with pytest.raises(ValueError,match='provenance'):Demonstrations(path,split='test',normalizer=broken)


class SmallBC(nn.Module):
    def __init__(self):super().__init__();self.value=nn.Parameter(torch.zeros(4,4))
    def bc_loss(self,obs,actions,valid):
        return ((self.value+torch.randn_like(self.value)*.01-actions).square()*valid[:,:,None]).mean()


def test_expert_post_success_frames_never_enter_bc_or_normalization(tmp_path):
    path=fixture_manifest(tmp_path);manifest=json.loads(path.read_text())
    marker=tmp_path/'a_success.npy';np.save(marker,np.array([False,False,True,True,True]))
    manifest['episodes'][0]['files']['stable_success']={'path':marker.name,'sha256':file_hash(marker)}
    path.write_text(json.dumps(manifest))
    ds=Demonstrations(path)
    assert len(ds)==3
    assert ds.normalizer['history']['mean'][0]==1  # only force samples 0,1,2
    assert ds[2][2].tolist()==[True,False,False,False]


def test_pretrain_restart_matches_uninterrupted_updates(tmp_path):
    ds=Demonstrations(fixture_manifest(tmp_path));base=SmallBC()
    full=copy.deepcopy(base);part=copy.deepcopy(base)
    opt_full=torch.optim.Adam(full.parameters(),lr=.001);opt_part=torch.optim.Adam(part.parameters(),lr=.001)
    common=dict(dataset=ds,batch_size=2,seed=4,config_hash='fixture',device='cpu',save_every=1)
    pretrain(full,optimizer=opt_full,checkpoint=tmp_path/'full.pt',updates=4,**common)
    pretrain(part,optimizer=opt_part,checkpoint=tmp_path/'part.pt',updates=2,**common)
    resumed=SmallBC();opt=torch.optim.Adam(resumed.parameters(),lr=.001)
    result=pretrain(resumed,optimizer=opt,checkpoint=tmp_path/'part.pt',updates=4,**common)
    torch.testing.assert_close(full.value,resumed.value,rtol=0,atol=0)
    assert result['robot_baseline_acceptance']=='NOT_RUN'
