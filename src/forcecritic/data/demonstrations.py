"""Causal demonstration windows from an accepted, hashed recording manifest.

The recording format keeps observed motion separate from submitted command labels.
Unknown HF pre/post-EMA actions cannot enter this loader by merely changing shape.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset

FIELDS=('rgb','state','wrench','motion','previous_command','action')


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


class Demonstrations(Dataset):
    def __init__(self,manifest,split='train',episode_limit=200,normalizer=None):
        self.path=Path(manifest);self.manifest=json.loads(self.path.read_text())
        self.manifest_hash=file_hash(self.path)
        contract=self.manifest['contract']
        required={'task':'Isaac-Forge-PegInsert-Direct-v0','action_stage':'submitted_pre_ema',
                  'action_axes':['x','y','z','yaw'],'observation_timing':'before_action',
                  'wrench_dim':6,'history_steps':4,'chunk_length':4}
        for k,v in required.items():
            if contract.get(k)!=v:raise ValueError(f'unaccepted demonstration contract: {k}')
        evidence=self.path.parent/contract['acceptance_evidence']
        if file_hash(evidence)!=contract['acceptance_sha256']:raise ValueError('acceptance evidence changed')
        acceptance=json.loads(evidence.read_text())
        if acceptance.get('status')!='PASSED' or not all(acceptance.get(k) is True for k in (
            'action_alignment','sensor_alignment','camera_alignment','causal_observations')):
            raise ValueError('measured acceptance not passed')
        self.rows=[r for r in self.manifest['episodes'] if r['split']==split and r['successful']]
        self.rows=sorted(self.rows,key=lambda r:r['parent_id'])
        if episode_limit is not None:self.rows=self.rows[:episode_limit]
        if not self.rows:raise ValueError('no accepted successful episodes for requested split')
        if len({r['parent_id'] for r in self.manifest['episodes']})!=len(self.manifest['episodes']):
            raise ValueError('duplicate parent across demonstration splits')
        run_splits={}
        for row in self.manifest['episodes']:
            group=row.get('collection_run_id')
            if group and run_splits.setdefault(group,row['split'])!=row['split']:
                raise ValueError('collection run leaks across splits')
        self.arrays=[];self.windows=[]
        for ei,row in enumerate(self.rows):
            if row['source'] not in ('real_forge_recording','aligned_hf'):
                raise ValueError('unverified demonstration source')
            arrays={}
            for field in FIELDS:
                entry=row['files'][field];path=self.path.parent/entry['path']
                if file_hash(path)!=entry['sha256']:raise ValueError('recording content changed')
                arrays[field]=np.load(path,mmap_mode='r',allow_pickle=False)
            t=len(arrays['action'])
            shapes={'state':(t,8),'wrench':(t,6),'motion':(t,6),'previous_command':(t,4),'action':(t,4)}
            for field,shape in shapes.items():
                if arrays[field].shape!=shape or not np.isfinite(arrays[field]).all():raise ValueError(f'invalid {field}')
            rgb=arrays['rgb']
            if rgb.ndim!=4 or rgb.shape[0]!=t or rgb.shape[-1]!=3 or rgb.dtype!=np.uint8:
                raise ValueError('expected time-aligned uint8 RGB')
            if t==0 or np.abs(arrays['action']).max()>1:raise ValueError('native normalized commands required')
            if not np.allclose(arrays['previous_command'][1:],arrays['action'][:-1],atol=1e-6):
                raise ValueError('historical command alignment failed')
            self.arrays.append(arrays);self.windows.extend((ei,i) for i in range(t))
        if normalizer is None:
            if split!='train':raise ValueError('validation/test must reuse training normalization')
            self.normalizer=self.fit_normalizer()
        else:
            legal={r['parent_id'] for r in self.manifest['episodes'] if r['split']=='train' and r['successful']}
            if normalizer['manifest_sha256']!=self.manifest_hash or not set(normalizer['training_parents'])<=legal:
                raise ValueError('normalizer provenance is not this training split')
            self.normalizer=normalizer

    def fit_normalizer(self):
        def stats(key):
            n=0;total=None;squares=None
            for arrays in self.arrays:
                x=np.asarray(arrays['state'],dtype=np.float64) if key=='state' else np.concatenate(
                    [arrays[f] for f in ('wrench','motion','previous_command')],1).astype(np.float64)
                n+=len(x);total=x.sum(0) if total is None else total+x.sum(0)
                squares=(x*x).sum(0) if squares is None else squares+(x*x).sum(0)
            mean=total/n;std=np.sqrt(np.maximum(squares/n-mean*mean,0)).clip(1e-6)
            return {'mean':mean.tolist(),'std':std.tolist()}
        return {'state':stats('state'),'history':stats('history'),
                'training_parents':[r['parent_id'] for r in self.rows], 'manifest_sha256':self.manifest_hash}

    def __len__(self):return len(self.windows)

    def __getitem__(self,index):
        episode,t=self.windows[index];a=self.arrays[episode]
        times=np.maximum(np.arange(t-3,t+1),0)
        history=np.concatenate([a[f][times] for f in ('wrench','motion','previous_command')],1).astype(np.float32)
        state=np.array(a['state'][t],dtype=np.float32,copy=True)
        for key,x in [('state',state),('history',history)]:
            stats=self.normalizer[key]
            x-=np.array(stats['mean'],dtype=np.float32);x/=np.array(stats['std'],dtype=np.float32)
        k=min(4,len(a['action'])-t);actions=np.zeros((4,4),dtype=np.float32);actions[:k]=a['action'][t:t+k]
        obs={'rgb':torch.from_numpy(np.array(a['rgb'][t],copy=True)).permute(2,0,1).float()/255,
             'state':torch.from_numpy(state),'history':torch.from_numpy(history)}
        return obs,torch.from_numpy(actions),torch.arange(4)<k
