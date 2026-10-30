import copy
import torch
from torch import nn
import pytest
from forcecritic.models.act import ACT
from forcecritic.models.critic import TwinCritic
from forcecritic.training.updates import actor_step, critic_step, td_target, clone_arms

torch.set_num_threads(2)


@pytest.fixture
def setup():
    torch.manual_seed(7)
    actor=ACT(width=32,layers=1,backbone=nn.Conv2d(3,8,3,padding=1),feature_dim=8)
    obs={'state':torch.randn(3,8),'history':torch.randn(3,4,16),'rgb':torch.randn(3,3,16,16)}
    return actor,obs


class LinearCritic(nn.Module):
    def __init__(self,sign):
        super().__init__();self.slope=nn.Parameter(torch.tensor(float(sign)))
    def forward(self,h,a):
        r=self.slope*a[:,:,0].mean(1)
        return torch.stack((r,r*0),-1)[:,None].expand(-1,2,-1)


def test_fixed_act_critic_gradient_changes_actual_update(setup):
    actor,obs=setup
    results=[]
    for sign in [1,-1]:
        a=copy.deepcopy(actor);q=LinearCritic(sign)
        before=copy.deepcopy(q.state_dict())
        r=actor_step(a,q,torch.optim.SGD(a.parameters(),lr=.005),obs,torch.zeros(3,2),
                     torch.ones(3,dtype=torch.bool),a(obs).detach(),alpha_bc=0,eta_step=0)
        assert r['q_grad_norm']>0
        assert r['delta_action'][:,:,0].mean()*sign>0
        assert all(torch.equal(before[k],q.state_dict()[k]) for k in before)
        assert q.slope.grad is None
        results.append(r['after'])
    assert not torch.equal(*results)


def test_critic_step_does_not_update_actor(setup):
    a,obs=setup; q=TwinCritic(2,16)
    old=copy.deepcopy(a.state_dict())
    critic_step(q,torch.optim.Adam(q.parameters()),torch.zeros(3,2),a(obs),torch.ones(3,2))
    assert all(p.grad is None for p in a.parameters())
    assert all(torch.equal(old[k],a.state_dict()[k]) for k in old)


def test_condition_cannot_be_changed_to_game_q(setup):
    a,obs=setup;q=TwinCritic(2,16)
    condition=nn.Parameter(torch.randn(3,2)); opt=torch.optim.SGD(list(a.parameters())+[condition],lr=.01)
    before=condition.detach().clone()
    actor_step(a,q,opt,obs,condition,torch.ones(3,dtype=torch.bool),a(obs).detach(),alpha_bc=0)
    assert condition.grad is None
    assert torch.equal(before,condition)


def test_zero_latent_does_not_read_posterior(setup):
    a,obs=setup
    before=a(obs).detach()
    def forbidden(*args,**kwargs):raise AssertionError('posterior invoked by deployment')
    a.posterior.forward=forbidden
    obs['future_actions']=torch.full((3,4,4),float('nan'))
    assert torch.equal(before,a(obs).detach())


def test_empty_mask_bc_finite_and_critic_frozen(setup):
    a,obs=setup;q=TwinCritic(2,16)
    r=actor_step(a,q,torch.optim.Adam(a.parameters(),lr=1e-5),obs,torch.zeros(3,2),
        torch.zeros(3,dtype=torch.bool),a(obs).detach(),demo=(obs,torch.zeros(3,4,4),torch.ones(3,4,dtype=torch.bool)))
    assert r['q_grad_norm']==0
    assert r['bc_grad_norm']>0
    assert torch.isfinite(torch.tensor(r['loss']))


def test_pair_supervision_policy_and_split_guard(setup):
    a,obs=setup;q=TwinCritic(2,16);o=torch.optim.Adam(q.parameters())
    pairs=dict(policy_sha='old',splits=['train']*3,condition=torch.zeros(3,2),
               plus=torch.ones(3,4,4),minus=-torch.ones(3,4,4),delta=torch.ones(3,2),accepted=torch.ones(3,dtype=torch.bool))
    with pytest.raises(ValueError,match='stale'):
        critic_step(q,o,torch.zeros(3,2),a(obs),torch.ones(3,2),pairs=pairs,policy_sha='new')
    pairs.update(policy_sha='new',splits=['train','test','train'])
    with pytest.raises(ValueError,match='non-training'):
        critic_step(q,o,torch.zeros(3,2),a(obs),torch.ones(3,2),pairs=pairs,policy_sha='new')


def test_same_data_same_mask_independent_copies(setup):
    a,obs=setup;q=TwinCritic(2,16)
    arms=clone_arms(a,q,torch.optim.Adam(a.parameters()),torch.optim.Adam(q.parameters()),'abc',[{'split':'train','id':1}])
    assert arms['B2']['data_hash']==arms['B3']['data_hash']
    assert arms['B2']['whitelist']==arms['B3']['whitelist']
    assert torch.equal(arms['B2']['rng'],arms['B3']['rng'])
    arms['B2']['whitelist'].clear()
    assert len(arms['B3']['whitelist'])==1


def test_frozen_next_action_cache_keeps_dynamic_critic_targets(setup):
    actor,obs=setup
    q=LinearCritic(1)
    with torch.no_grad():action=actor(obs)
    args=(torch.zeros(3,2),torch.ones(3,2),torch.tensor([.99**4]*3),torch.tensor([1.,0.,1.]))
    direct=td_target(actor,q,obs,*args)
    def forbidden(_):raise AssertionError('cached frozen actor should not run')
    cached=td_target(forbidden,q,None,*args,next_action=action)
    assert torch.equal(direct,cached)
    with torch.no_grad():q.slope.mul_(2)
    updated=td_target(forbidden,q,None,*args,next_action=action)
    assert not torch.equal(cached[[0,2]],updated[[0,2]])
    assert torch.equal(updated[1],torch.ones(2))  # terminal does not bootstrap
