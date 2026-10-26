"""Reproduce the historical A2 first step, then train only the missing matched A1."""
import copy
import csv
import json
import os
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch.utils.data import default_collate
from torch.nn import functional as F
from forcecritic.staged.common import locked_plan,training_indices,CONFIG
from forcecritic.runtime import atomic_json,digest,source_hashes
from forcecritic.models.critic import TwinCritic
from forcecritic.training.pretrain import baseline_actor
from forcecritic.training.live import LivePolicy
from forcecritic.training.updates import actor_step
from forcecritic.data.demonstrations import Demonstrations

p=locked_plan();report=Path(p['report']);result=Path(p['results']);result.mkdir(parents=True,exist_ok=True)
directory=Path(p['checkpoints']);directory.mkdir(parents=True,exist_ok=True)
if (directory/'A1.pt').exists() or (result/'A1_updates.csv').exists():raise FileExistsError('Preserve existing training attempt')
def progress(phase,**kw):
    atomic_json(report/'training_progress.json',dict(status='PASSED' if phase=='A1_training_complete' else 'RUNNING',phase=phase,pid=os.getpid(),updated=time.time(),**kw))
    print('STAGE_V3',phase,json.dumps(kw),flush=True)
progress('loading',actor_completed=0)
old=Path('checkpoints/20261023T104759Z');device='cuda:0'
act=torch.load(old/'act0.pt',weights_only=False,map_location='cpu')
critic0=torch.load(old/'critic0.pt',weights_only=False,map_location='cpu')
a2=torch.load(old/'direct_Q.pt',weights_only=False,map_location='cpu')
bundle=torch.load(old/'shared_replay.pt',weights_only=False,map_location='cpu')
rows=[r for r in bundle['rows'] if r['split']=='train']
if any(r['source']!='real_forge' for r in rows):raise ValueError('Non-real replay')
torch.manual_seed(p['torch_seed'])
actor=baseline_actor().to(device);actor.load_state_dict(act['actor'])
policy=LivePolicy(actor,act['identity']['normalizer'],device)
critic=TwinCritic(critic0['condition_dim']).to(device);critic.load_state_dict(a2['critic'])
dataset=Demonstrations(p['demonstration_manifest'],normalizer=policy.normalizer)
reference=copy.deepcopy(actor).eval().requires_grad_(False)
initial_cpu=torch.get_rng_state();initial_cuda=torch.cuda.get_rng_state_all()
def batch(i):
    ri,di=training_indices(len(rows),len(dataset),i,p['batch_size'],p['actor_numpy_seed_base'])
    selected=[rows[int(j)] for j in ri]
    obs=policy.batch([r['observation'] for r in selected])
    condition=torch.stack([r['condition'] for r in selected]).to(device)
    demo_obs,actions,valid=default_collate([dataset[int(j)] for j in di])
    return obs,condition,({k:v.to(device) for k,v in demo_obs.items()},actions.to(device),valid.to(device)),ri,di
obs,condition,demo,_,_=batch(0)
probe=copy.deepcopy(actor)
with torch.no_grad():old_action=reference(obs)
measured=actor_step(probe,critic,torch.optim.Adam(probe.parameters(),lr=p['actor_lr']),obs,condition,
    torch.ones(p['batch_size'],device=device,dtype=torch.bool),old_action,demo=demo)
observed={k:measured[k] for k in ('loss','q_grad_norm','bc_grad_norm','q_bc_dot','saturation')}
observed.update(mean_abs_delta_action=float(measured['delta_action'].abs().mean()),max_abs_delta_action=float(measured['delta_action'].abs().max()))
historical=next(csv.DictReader(Path('results/20261023T104759Z/direct_Q_actor_updates.csv').open()))
checks={k:dict(expected=float(historical[k]),actual=v,passed=bool(np.isclose(v,float(historical[k]),rtol=p['reproduction_rtol'],atol=p['reproduction_atol']))) for k,v in observed.items()}
audit_pass=all(v['passed'] for v in checks.values())
atomic_json(report/'a2_reproduction_audit.json',dict(status='PASSED' if audit_pass else 'BLOCKED',checks=checks,
    scope='One GPU optimizer-step implementation audit, not a robot success result; model discarded',
    audited_updates=1,source_hashes=source_hashes(),plan_sha=digest(CONFIG),
    rng_reason='ACT/Critic initialization reproduced; frozen z=0 actor and deterministic MLP TD loop do not consume RNG; CVAE draws begin at Actor step 1'))
if not audit_pass:raise RuntimeError('Historical A2 not reproducible; do not silently relabel or train unmatched A1')
del probe,measured,obs,condition,demo
torch.cuda.empty_cache()
torch.set_rng_state(initial_cpu);torch.cuda.set_rng_state_all(initial_cuda)
optimizer=torch.optim.Adam(actor.parameters(),lr=p['actor_lr'])
trace=[]
with (result/'A1_updates.csv').open('x',newline='') as stream:
    fields=['update','loss','bc_loss','step_loss','bc_grad_norm','step_grad_norm','mean_abs_delta_action','max_abs_delta_action']
    writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
    for i in range(p['actor_updates']):
        obs,condition,demo,ri,di=batch(i)
        with torch.no_grad():old_action=reference(obs)
        optimizer.zero_grad(set_to_none=True)
        before=actor(obs)
        bc=actor.bc_loss(*demo);step=F.mse_loss(before,old_action)
        params=[v for v in actor.parameters() if v.requires_grad]
        norm=lambda gs:float(sum(g.detach().square().sum() for g in gs if g is not None).sqrt())
        bc_norm=norm(torch.autograd.grad(bc,params,retain_graph=True,allow_unused=True))
        step_norm=norm(torch.autograd.grad(step,params,retain_graph=True,allow_unused=True))
        loss=p['alpha_bc']*bc+p['eta_step']*step
        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite A1 loss')
        loss.backward();optimizer.step()
        with torch.no_grad():delta=actor(obs)-before
        row=dict(update=i+1,loss=float(loss.detach()),bc_loss=float(bc.detach()),step_loss=float(step.detach()),
            bc_grad_norm=bc_norm,step_grad_norm=step_norm,mean_abs_delta_action=float(delta.abs().mean()),max_abs_delta_action=float(delta.abs().max()))
        writer.writerow(row);stream.flush()
        trace.append(dict(update=i+1,replay_indices=ri.tolist(),demo_indices=di.tolist()))
        if (i+1)%10==0:
            temp=directory/'A1.tmp'
            torch.save(dict(actor=actor.state_dict(),identity=act['identity'],actor_optimizer=optimizer.state_dict(),
                actor_updates=i+1,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),
                python_rng=random.getstate(),numpy_rng=np.random.get_state(),plan_sha=digest(CONFIG)),temp)
            temp.replace(directory/'A1.pt');progress('A1_training',actor_completed=i+1,last_update=row)
atomic_json(report/'A1_batch_order.json',trace)
atomic_json(report/'A1_training.json',dict(status='PASSED',actor_updates=p['actor_updates'],checkpoint=str(directory/'A1.pt'),
    checkpoint_sha=digest(directory/'A1.pt'),plan_sha=digest(CONFIG),batch_order_sha=digest(report/'A1_batch_order.json'),
    only_objective_difference_from_A2='Q coefficient is zero; same BC, reference ACT0, step coefficient, batch seeds and CVAE noise sequence',
    source_hashes=source_hashes(),new_environment_interactions=0,robot_evaluation='NOT_RUN'))
progress('A1_training_complete',actor_completed=200)
