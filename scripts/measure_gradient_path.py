"""Numerical SOFTWARE-ONLY evidence; this generates no robot result rows."""
import copy
import os
os.environ["CUDA_VISIBLE_DEVICES"]=""
import json
from pathlib import Path
import sys
sys.path.insert(0,'src')
import torch
from torch import nn
from forcecritic.models.act import ACT
from forcecritic.training.updates import actor_step
from forcecritic.runtime import atomic_json

torch.set_num_threads(2);torch.manual_seed(7)
a=ACT(width=32,layers=1,backbone=nn.Conv2d(3,8,3,padding=1),feature_dim=8)
obs={'state':torch.randn(3,8),'history':torch.randn(3,4,16),'rgb':torch.rand(3,3,16,16)}
class Q(nn.Module):
    def __init__(self,s):super().__init__();self.slope=nn.Parameter(torch.tensor(s))
    def forward(self,h,action):
        r=self.slope*action[:,:,0].mean(1)
        return torch.stack((r,r*0),-1)[:,None].expand(-1,2,-1)
records=[]
for sign in [1.,-1.]:
    actor=copy.deepcopy(a);critic=Q(sign)
    original=critic.slope.detach().clone()
    r=actor_step(actor,critic,torch.optim.SGD(actor.parameters(),lr=1e-5),obs,torch.zeros(3,2),
                 torch.ones(3,dtype=torch.bool),actor(obs).detach(),alpha_bc=0,eta_step=0)
    records.append({'critic_slope':sign,'q_gradient_norm':r['q_grad_norm'],
        'mean_x_action_change':float(r['delta_action'][:,:,0].mean()),
        'critic_unchanged':torch.equal(original,critic.slope.detach()),
        'before':r['before'].tolist(),'after':r['after'].tolist()})
run_id=json.loads(Path('reports/PROJECT_STATUS.json').read_text())['run_id']
atomic_json(Path('reports')/run_id/'software_gradient_evidence.json',{
    'scope':'CPU software fixture, NOT FORGE, NOT a performance experiment',
    'actor':'ACT-CVAE with injected small visual backbone','bc_weight':0,'seed':7,'records':records})
print(json.dumps([{k:v for k,v in r.items() if k not in ('before','after')} for r in records],indent=2))
