"""Resumable CVAE BC optimizer loop, admitted only after demonstration acceptance.

Software unit tests may use injected small networks. This module does not turn a
BC loss into a robot baseline acceptance; that requires 32 independent real starts.
"""
import json
import os
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import default_collate
from ..models.act import ACT
from ..runtime import atomic_json


def baseline_actor(seed=0):
    # Standard-capacity ACT backbone/transformer for the real baseline, not the tiny test fixture.
    torch.manual_seed(seed)
    return ACT(width=512,layers=4,decoder_layers=7)


def pretrain(actor,dataset,optimizer,checkpoint,*,updates=10000,batch_size=128,seed=0,
             config_hash,device='cuda:0',save_every=100,resume=True,after_interface_audit=False):
    if not 1<=updates<=20000 or (updates>10000 and not after_interface_audit):
        raise ValueError('pretraining above 10000 requires the recorded interface audit; absolute cap 20000')
    if batch_size<1 or save_every<1:raise ValueError('positive batch size/checkpoint interval required')
    if not dataset.rows or any(r['split']!='train' or not r['successful'] for r in dataset.rows):
        raise ValueError('positive training episodes only')
    expected=torch.device(device)
    if expected.type=='cuda' and expected.index is None:expected=torch.device('cuda',torch.cuda.current_device())
    if any(p.device!=expected for p in actor.parameters()):
        raise ValueError('move ACT to the requested device before constructing its optimizer')
    checkpoint=Path(checkpoint);checkpoint.parent.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator().manual_seed(seed);start=0;history=[]
    identity={'config_hash':config_hash,'dataset_sha256':dataset.manifest_hash,
              'training_parents':[r['parent_id'] for r in dataset.rows],
              'normalizer':dataset.normalizer,'batch_size':batch_size}
    if checkpoint.exists() and not resume:raise FileExistsError('use a fresh checkpoint path for a new run')
    if resume and checkpoint.exists():
        state=torch.load(checkpoint,map_location='cpu',weights_only=False)  # trusted local artifact only
        if state['identity']!=identity:raise ValueError('pretraining resume identity changed')
        actor.load_state_dict(state['actor']);optimizer.load_state_dict(state['optimizer'])
        start=state['update'];history=state['history'];generator.set_state(state['sampler_rng'])
        if start>updates:raise ValueError('requested update limit precedes existing checkpoint')
        torch.set_rng_state(state['torch_rng']);random.setstate(state['python_rng']);np.random.set_state(state['numpy_rng'])
        if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
    else:
        torch.manual_seed(seed);random.seed(seed);np.random.seed(seed)
    actor.train()
    def save(update):
        payload={'identity':identity,'actor':actor.state_dict(),'optimizer':optimizer.state_dict(),'update':update,
                 'history':history,'sampler_rng':generator.get_state(),'torch_rng':torch.get_rng_state(),
                 'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                 'python_rng':random.getstate(),'numpy_rng':np.random.get_state()}
        temp=checkpoint.with_suffix('.tmp');torch.save(payload,temp);os.replace(temp,checkpoint)
        atomic_json(checkpoint.with_suffix('.json'),{'status':'COMPLETED' if update>=updates else 'RUNNING',
                    'completed_updates':update,'requested_updates':updates,'identity':identity,
                    'robot_baseline_acceptance':'NOT_RUN','checkpoint':str(checkpoint)})
    completed=start
    if not checkpoint.exists() or not resume:save(start)
    try:
        for update in range(start,updates):
            indices=torch.randint(len(dataset),(batch_size,),generator=generator).tolist()
            obs,actions,valid=default_collate([dataset[i] for i in indices])
            obs={k:v.to(device) for k,v in obs.items()};actions=actions.to(device);valid=valid.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss=actor.bc_loss(obs,actions,valid)
            if not torch.isfinite(loss):raise FloatingPointError('nonfinite CVAE loss')
            loss.backward();optimizer.step();completed=update+1
            history.append({'update':completed,'bc_cvae_loss':float(loss.detach())})
            if completed%save_every==0 or completed==updates:save(completed)
    except BaseException as error:
        # Never overwrite a valid restart with potentially half-applied optimizer state.
        # Replay from the last atomic boundary; record lost work instead of hiding it.
        last=json.loads(checkpoint.with_suffix('.json').read_text())['completed_updates']
        atomic_json(checkpoint.with_suffix('.interruption.json'),{
            'status':'INTERRUPTED','completed_in_memory':completed,'restart_update':last,
            'completed_updates_to_replay':completed-last,'error':repr(error)})
        raise
    return {'completed_updates':completed,'checkpoint':str(checkpoint),'history':history,
            'robot_baseline_acceptance':'NOT_RUN'}
