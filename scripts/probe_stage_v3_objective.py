"""Read-only final-policy objective decomposition; never an insertion result."""
import json
from pathlib import Path
import time
import torch
from torch.nn import functional as F
from torch.utils.data import default_collate
from forcecritic.runtime import atomic_json,digest
from forcecritic.staged.common import locked_plan,training_indices,CONFIG
from forcecritic.training.pretrain import baseline_actor
from forcecritic.training.live import LivePolicy
from forcecritic.models.critic import TwinCritic
from forcecritic.data.demonstrations import Demonstrations

p=locked_plan();folder=Path(p['report'])
plan_path=folder/'objective_probe_plan.json';output=folder/'objective_probe.json'
if plan_path.exists() or output.exists():raise FileExistsError('Preserve read-only probe attempts')
plan=dict(scope='Read-only final A2 gradients on training data; zero optimizer and simulator steps',
    parent_config_sha=digest(CONFIG),batch_update_index=199,torch_cvae_probe_seed=0,device='cpu',
    policy_sha=p['inputs']['checkpoints/20261023T104759Z/direct_Q.pt'],
    limitation='A fixed final-policy check, not reconstruction of every historical step or a robot success metric')
atomic_json(plan_path,plan);started=time.time()
torch.set_num_threads(2)
old=Path('checkpoints/20261023T104759Z')
load=lambda name:torch.load(old/name,map_location='cpu',weights_only=False,mmap=True)
act0=load('act0.pt');a2=load('direct_Q.pt');fitted=load('critic0.pt');bundle=load('shared_replay.pt')
rows=[r for r in bundle['rows'] if r['split']=='train']
actor=baseline_actor();actor.load_state_dict(a2['actor']);actor.eval()
reference=baseline_actor();reference.load_state_dict(act0['actor']);reference.eval().requires_grad_(False)
critic=TwinCritic(fitted['condition_dim']);critic.load_state_dict(a2['critic']);critic.eval().requires_grad_(False)
policy=LivePolicy(actor,act0['identity']['normalizer'],'cpu')
dataset=Demonstrations(p['demonstration_manifest'],normalizer=policy.normalizer)
ri,di=training_indices(len(rows),len(dataset),199,p['batch_size'],p['actor_numpy_seed_base'])
selected=[rows[int(i)] for i in ri];obs=policy.batch([r['observation'] for r in selected])
condition=torch.stack([r['condition'] for r in selected])
demo=default_collate([dataset[int(i)] for i in di])
with torch.no_grad():old_action=reference(obs)
params=[v for v in actor.parameters() if v.requires_grad]
action=actor(obs);values=critic(condition.detach(),action)
q_loss=-(values[:,:,0].min(1).values-p['lambda_cost']*values[:,:,1].max(1).values).mean()
action_gradient=torch.autograd.grad(q_loss,action,retain_graph=True)[0]
qg=torch.autograd.grad(q_loss,params,retain_graph=True,allow_unused=True)
torch.manual_seed(plan['torch_cvae_probe_seed'])
bc=actor.bc_loss(*demo);step=F.mse_loss(action,old_action)
bcg=torch.autograd.grad(p['alpha_bc']*bc,params,retain_graph=True,allow_unused=True)
sg=torch.autograd.grad(p['eta_step']*step,params,allow_unused=True)
def dot(left,right):
    return sum(float((a.double()*b.double()).sum()) for a,b in zip(left,right) if a is not None and b is not None)
q2=dot(qg,qg);b2=dot(bcg,bcg);s2=dot(sg,sg);qb=dot(qg,bcg);qs=dot(qg,sg);bs=dot(bcg,sg)
total2=q2+b2+s2+2*(qb+qs+bs)
out=dict(status='PASSED',scope=plan['scope'],plan_sha=digest(plan_path),started=started,finished=time.time(),
    losses=dict(q=float(q_loss.detach()),bc=float(bc.detach()),step=float(step.detach())),
    weighted_parameter_gradient_norm=dict(q=q2**.5,bc=b2**.5,step=s2**.5,total=max(total2,0)**.5),
    action_q_gradient_norm=float(action_gradient.norm()),
    q_bc_cosine=qb/(q2*b2)**.5,q_step_cosine=qs/(q2*s2)**.5,
    total_gradient_projection_onto_q_relative_to_q=(q2+qb+qs)/q2,
    optimizer_steps=0,simulator_steps=0,checkpoint_after_sha=digest(old/'direct_Q.pt'),
    interpretation='Positive total-gradient projection preserves a local descent component for Q loss; this does not establish a correct physical action direction or improvement')
if out['checkpoint_after_sha']!=plan['policy_sha']:raise ValueError('Read-only probe changed policy')
atomic_json(output,out);print(json.dumps(out,indent=2))
