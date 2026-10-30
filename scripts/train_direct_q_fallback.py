"""Ordinary direct-Q baseline explicitly permitted after D0 no-go by CODEX_TASK.

This is NOT the paired-endpoint B1/B2/B3 ablation and does not admit calibration.
"""
import copy
import csv
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from forcecritic.runtime import Run,digest,atomic_json,source_hashes
from forcecritic.models.critic import TwinCritic
from forcecritic.data.demonstrations import Demonstrations
from forcecritic.training.pretrain import baseline_actor
from forcecritic.training.live import LivePolicy,replay_batch
from forcecritic.training.replay import ReturnScale
from forcecritic.training.updates import actor_step,critic_step,td_target,polyak

run=Run('configs/minimal_validation.yaml','20261023T104759Z')
if not run.verified('baseline'):raise RuntimeError('Verified real baseline required')
directory=Path('checkpoints')/run.run_id
output=directory/'direct_Q.pt'
if output.exists():raise FileExistsError('Do not retrain/overwrite a completed direct-Q attempt')
device='cuda:0'
plan_path=run.report/'direct_q_plan.json'
plan=dict(status='LOCKED',scope='Ordinary direct Q permitted by CODEX_TASK D0; not paired B1/B2/B3',
    act_sha=digest(directory/'act0.pt'),critic_sha=digest(directory/'critic0.pt'),
    replay_sha=digest(directory/'shared_replay.pt'),critic_updates=2000,actor_updates=200,
    actor_lr=1e-5,critic_lr=3e-4,alpha_bc=1.,eta_step=1.,lambda_cost=1.,batch_size=128,
    replay_split='train',demo_split='train',td_seed=110000,actor_seed=140000,
    evaluation_seeds=list(range(21000,21032)),arms=['direct_B0','direct_Q'],
    renderer='Original baseline configuration; render repair probes are separate',
    no_mc_or_pair_labels=True,calibration_gate_bypassed=False,
    attribution='Q gradient reaches ACT; B0 vs Q+BC includes additional BC, so it does not isolate Q benefit alone',
    source_hashes=source_hashes())
atomic_json(plan_path,plan)
def progress(phase,**kw):
    atomic_json(run.report/'direct_q_progress.json',dict(phase=phase,updated=time.time(),**kw))
    print('DIRECT_Q',phase,json.dumps(kw),flush=True)
progress('loading',critic_completed=0,actor_completed=0)
act=torch.load(directory/'act0.pt',weights_only=False,map_location='cpu')
fitted=torch.load(directory/'critic0.pt',weights_only=False,map_location='cpu')
bundle=torch.load(directory/'shared_replay.pt',weights_only=False,map_location='cpu')
if any(x['policy_sha']!=plan['act_sha'] for x in (fitted,bundle)):raise ValueError('Frozen continuation version mismatch')
rows=[r for r in bundle['rows'] if r['split']=='train']
if not rows or any(r['source']!='real_forge' for r in rows):raise ValueError('Actual training replay required')
torch.manual_seed(0)
actor=baseline_actor().to(device);actor.load_state_dict(act['actor'])
policy=LivePolicy(actor,act['identity']['normalizer'],device)
critic=TwinCritic(fitted['condition_dim']).to(device);critic.load_state_dict(fitted['critic'])
target=copy.deepcopy(critic).eval().requires_grad_(False)
optimizer=torch.optim.Adam(critic.parameters(),lr=3e-4);optimizer.load_state_dict(fitted['optimizer'])
scale=ReturnScale();scale.scale=np.asarray(fitted['scale'])
size=plan['batch_size'];losses=[]
for i in range(2000):
    rng=np.random.default_rng(plan['td_seed']+i)
    batch=replay_batch(rows,rng.integers(len(rows),size=size),policy,scale)
    labels=td_target(actor,target,batch['next_obs'],batch['next_condition'],batch['returns'],batch['gamma_k'],batch['bootstrap'],batch['next_action'])
    loss=critic_step(critic,optimizer,batch['condition'],batch['action'],labels)
    if not np.isfinite(loss):raise FloatingPointError('Nonfinite critic loss')
    losses.append(loss);polyak(critic,target)
    if (i+1)%20==0:progress('critic',critic_completed=i+1,actor_completed=0,loss=loss)
dataset=Demonstrations('data/act_demonstrations/manifest.json',normalizer=policy.normalizer)
reference=copy.deepcopy(actor).eval().requires_grad_(False)
actor_optimizer=torch.optim.Adam(actor.parameters(),lr=1e-5)
records=[]
path=run.result/'direct_Q_actor_updates.csv'
if path.exists():raise FileExistsError('Do not overwrite an optimizer attempt')
with path.open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=['update','loss','q_grad_norm','bc_grad_norm','q_bc_dot','saturation','mean_abs_delta_action','max_abs_delta_action'])
    writer.writeheader()
    for i in range(200):
        rng=np.random.default_rng(plan['actor_seed']+i)
        selected=[rows[int(j)] for j in rng.integers(len(rows),size=size)]
        obs=policy.batch([r['observation'] for r in selected])
        condition=torch.stack([r['condition'] for r in selected]).to(device)
        demo_obs,actions,valid=default_collate([dataset[int(j)] for j in rng.integers(len(dataset),size=size)])
        demo=({k:v.to(device) for k,v in demo_obs.items()},actions.to(device),valid.to(device))
        with torch.no_grad():old=reference(obs)
        value=actor_step(actor,critic,actor_optimizer,obs,condition,torch.ones(size,device=device,dtype=torch.bool),old,demo=demo)
        record={k:value[k] for k in ('loss','q_grad_norm','bc_grad_norm','q_bc_dot','saturation')}
        record.update(update=i+1,mean_abs_delta_action=float(value['delta_action'].abs().mean()),max_abs_delta_action=float(value['delta_action'].abs().max()))
        if not all(np.isfinite(v) for v in record.values()):raise FloatingPointError('Nonfinite actor update')
        if record['q_grad_norm']<=0:raise ValueError('Missing Critic-to-ACT action gradient')
        writer.writerow(record);f.flush();records.append(record)
        if (i+1)%10==0:
            progress('actor',critic_completed=2000,actor_completed=i+1,last_update=record)
            temporary=output.with_suffix('.tmp')
            torch.save(dict(actor=actor.state_dict(),identity=act['identity'],critic=critic.state_dict(),
                actor_optimizer=actor_optimizer.state_dict(),actor_updates=i+1,plan_sha=digest(plan_path)),temporary)
            temporary.replace(output)
atomic_json(run.report/'direct_q_training.json',dict(status='TRAINED_NOT_EVALUATED',checkpoint=str(output),
    sha256=digest(output),critic_updates=2000,actor_updates=200,critic_final_loss=losses[-1],
    actor_update_records=str(path),robot_success_rate=None,plan_sha=digest(plan_path)))
progress('training_complete',critic_completed=2000,actor_completed=200)
