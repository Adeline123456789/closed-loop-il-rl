"""Software fixtures only: target progression and same-data attribution."""
import torch
import pytest
from torch import nn
import forcecritic.training.ablation as ablation
from forcecritic.models.critic import TwinCritic
from forcecritic.training.updates import critic_step
from forcecritic.diagnostics.pairs import training_whitelist


def test_short_updates_recompute_targets_and_reset_each_arm(monkeypatch):
    torch.manual_seed(12)
    class Actor(nn.Module):
        def __init__(self):
            super().__init__();self.layer=nn.Linear(2,16)
        def forward(self,obs):return self.layer(obs).reshape(-1,4,4).tanh()
    a=Actor();q=TwinCritic(2,16)
    condition=torch.ones(2,2)
    td=dict(condition=condition,action=torch.zeros(2,4,4),next_obs=condition,next_condition=condition,
            returns=torch.tensor([[1.,2.],[2.,3.]]),gamma_k=torch.ones(2)*.99**4,
            bootstrap=torch.ones(2),splits=['train','train'])
    end=dict(condition=condition,action=td['action'],returns=td['returns'],policy_sha='fixed',splits=['train','train'])
    pairs=dict(condition=condition,plus=torch.ones(2,4,4),minus=-torch.ones(2,4,4),
               delta=torch.ones(2,2),accepted=torch.ones(2,dtype=torch.bool),policy_sha='fixed',splits=['train','train'])
    captured=[]
    original=ablation.critic_step
    def record(*args,**kwargs):
        captured.append(args[4].clone())
        return original(*args,**kwargs)
    monkeypatch.setattr(ablation,'critic_step',record)
    progress=[]
    result=ablation.short_updates(a,q,torch.optim.SGD(a.parameters(),lr=.001),torch.optim.SGD(q.parameters(),lr=.01),
        td_batches=[td],endpoint_batches=[end],pair_batches=[pairs],actor_batches=[],whitelist=[],
        data_hash='same',policy_sha='fixed',critic_updates=2,actor_updates=0,target_tau=.5,
        progress=lambda **record:progress.append(record))
    assert [(r['arm'],r['completed_updates']) for r in progress
            if r['phase']=='critic' and r['completed_updates']>0]==[('B1',2),('B2',2),('B3',2)]
    assert not torch.equal(captured[0],captured[1])
    torch.testing.assert_close(captured[0],captured[2])
    torch.testing.assert_close(captured[0],captured[4])
    # B1/B2 have identical critic supervision; only B3 adds contrast loss.
    for key in result['B1']['critic']:
        torch.testing.assert_close(result['B1']['critic'][key],result['B2']['critic'][key])
    assert any(not torch.equal(result['B2']['critic'][k],result['B3']['critic'][k]) for k in result['B2']['critic'])
    assert all(r['data_hash']=='same' and r['whitelist']==[] for r in result.values())


def test_endpoint_labels_reject_test_split_and_detach_observation():
    q=TwinCritic(2,16)
    optimizer=torch.optim.SGD(q.parameters(),lr=.01)
    condition=torch.ones(2,2,requires_grad=True)
    action=torch.zeros(2,4,4,requires_grad=True)
    returns=torch.ones(2,2)
    endpoint=dict(condition=condition,action=action,returns=returns,
                  policy_sha='fixed',splits=['train','test'])
    before={k:v.clone() for k,v in q.state_dict().items()}
    with pytest.raises(ValueError,match='non-training MC'):
        critic_step(q,optimizer,condition,action,returns,endpoints=endpoint,policy_sha='fixed')
    assert all(torch.equal(v,q.state_dict()[k]) for k,v in before.items())
    endpoint['splits']=['train','train']
    critic_step(q,optimizer,condition,action,returns,endpoints=endpoint,policy_sha='fixed')
    assert condition.grad is None and action.grad is None


def test_generated_whitelist_retains_training_policy_identity():
    row=dict(checkpoint_id='p1/c0',split='train',continuation_sha='v1',replay_passed=True,
             independent_confirmation=True,improvement_lower=.1,cost_worse=False)
    white=training_whitelist([row],'v1')
    assert white==[dict(checkpoint_id='p1/c0',split='train',continuation_sha='v1')]
    # Reject stale filters even before any optimizer or sampling operation.
    with pytest.raises(ValueError,match='stale actor whitelist'):
        ablation.short_updates(None,None,None,None,td_batches=[],endpoint_batches=[],pair_batches=[],
            actor_batches=[],whitelist=white,data_hash='x',policy_sha='v2',critic_updates=0,actor_updates=0)
