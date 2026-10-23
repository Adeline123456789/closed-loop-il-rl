import numpy as np
import pytest
from forcecritic.envs.contracts import native_to_stock,stock_to_native,transform_wrench,contact_cost,StepLedger,execute_chunk
from forcecritic.data.splits import group_splits,inherit_split,train_statistics
from forcecritic.diagnostics.pairs import candidates,paired_effect,fidelity,training_whitelist


def test_action_seventh_axis_ema_contract():
    a=np.array([.3,-.5,2.,.1]);stock=native_to_stock(a)
    np.testing.assert_array_equal(stock,[.3,-.5,1,0,0,.1,-1])
    # Adapter is stateless: it does not apply EMA, including on repeated calls.
    np.testing.assert_array_equal(stock,native_to_stock(a))
    with pytest.raises(ValueError):stock_to_native(stock,ema_stage='unknown')
    np.testing.assert_array_equal(stock_to_native(stock,ema_stage='pre'),np.clip(a,-1,1))


def test_wrench_translation_rotation_and_physical_cost():
    rot=np.array([[0,-1,0],[1,0,0],[0,0,1]])
    np.testing.assert_array_equal(transform_wrench([1,0,0,0,0,0],rot,[1,0,0]),[0,1,0,0,0,1])
    with pytest.raises(ValueError):transform_wrench([1,2,3],np.eye(3),[0,0,0])
    assert contact_cost([2,0,0,0,0,0],[1]*6,True)==2


@pytest.mark.parametrize('end', ['success','unsafe','terminated','truncated','budget_exhausted','none'])
def test_chunk_prefix_discount_and_final_observation(end):
    calls=[];ledger=StepLedger(20)
    def step(command):
        calls.append(command)
        return dict(reward=1.,cost=2.,observation='reset state' if end!='none' else 'next',
                    final_observation='pre-reset',**{end:len(calls)==2})
    r=execute_chunk(np.zeros((4,4)),step,ledger,gamma=.9)
    k=4 if end=='none' else 2
    assert len(calls)==k and r['k']==k and r['valid'].sum()==k
    assert r['gamma_k']==pytest.approx(.9**k)
    assert r['returns'][0]==pytest.approx(sum(.9**i for i in range(k)))
    assert r['bootstrap']==(end in ['none','truncated'])
    assert r['next_observation']==('next' if end=='none' else 'pre-reset')
    assert ledger.total==k


def test_no_reset_bootstrap_without_final_obs():
    with pytest.raises(ValueError,match='pre-reset'):
        execute_chunk(np.zeros((4,4)),lambda _:dict(reward=0,cost=0,truncated=True,observation='reset'),StepLedger(4))


def test_all_cost_categories_and_parallel_budget():
    ledger=StepLedger(40)
    for cat in ledger.counts:ledger.charge(cat,num_envs=8)
    assert ledger.total==40
    with pytest.raises(RuntimeError):ledger.charge('branch')
    assert ledger.total==40


def test_splits_derived_branches_and_normalization():
    rows=[{'parent_id':str(i),'collection_run_id':str(i//2)} for i in range(100)]
    splits=group_splits(rows)
    for i in range(0,100,2):assert splits[str(i)]==splits[str(i+1)]
    assert inherit_split('1',splits)==splits['1']
    stats=train_statistics([{'split':'train','values':[[1],[3]]},{'split':'test','values':[[999]]}])
    assert stats['mean'][0]==2
    assert len([x for x in splits.values() if x=='train'])==70


def test_fidelity_compares_original_not_repeated_wrong_state():
    assert not fidelity({'pose':[0]},[{'pose':[1]}]*3,{'pose':.025})['passed']


def test_paired_finite_effect_not_improvement_over_base():
    r=paired_effect([2,2,2],[1,1,1],[3,3,3],['a','b','c'])
    assert r['delta']==1 and r['i_plus']==-1
    assert paired_effect([2]*3,[1]*3,[3]*3,['same']*3)['ci'] is None
    plus,minus,symmetric=candidates(np.ones((4,4)),np.ones((4,4)),.05)
    assert not symmetric


def test_test_labels_and_stale_mc_cannot_release_actor_mask():
    row=dict(split='test',continuation_sha='v1')
    with pytest.raises(ValueError):training_whitelist([row],'v1')
    row['split']='train'
    with pytest.raises(ValueError):training_whitelist([row],'v2')


def test_causal_history_and_geometric_success_isolation():
    from forcecritic.envs.observations import ObservationHistory,StableSuccess
    h=ObservationHistory();h.append(np.ones(6),np.ones(6)*2,np.ones(4)*3)
    obs=h.observation(np.zeros((8,8,3)),np.zeros(8))
    assert set(obs)=={'rgb','state','history'}
    h.append(np.ones(6)*4,np.ones(6)*5,np.ones(4)*6)
    assert obs['history'][0,0]==1  # no alias to later writes
    assert h.observation(np.zeros((8,8,3)),np.zeros(8))['history'][-1,0]==4
    s=StableSuccess()
    assert not s.step(True) and not s.step(False)
    assert not s.step(True) and not s.step(True) and s.step(True)


def test_reward_cannot_use_seventh_action_or_duplicate_success():
    from forcecritic.envs.reward import GeometricReward
    r=GeometricReward(1)
    terms={k:np.zeros(1) for k in r.TERMS}
    terms.update(success_pred_error=np.array([999.]),contact_penalty=np.array([999.]))
    assert r.compute(terms,[True])[0]==1
    assert r.compute(terms,[True])[0]==0
