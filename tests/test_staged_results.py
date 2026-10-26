import numpy as np
import pytest
from forcecritic.staged.common import training_indices
from forcecritic.staged.results import matched_decision,arm_metrics

PLAN=dict(bootstrap_seed=450000,bootstrap_resamples=20000,look_alpha=.025,
          minimum_success_difference=.05,paired_interpretation='seed pairing only')

def rows(successes):
    return [dict(seed=i,source='real_forge',stable_success=v) for i,v in enumerate(successes)]

def decision(a,b,look=1):return matched_decision(rows(a),rows(b),list(range(len(a))),look,PLAN)

def test_exact_historical_batch_rng_order():
    for update in (0,1,199):
        rng=np.random.default_rng(140000+update)
        expected_replay=rng.integers(1234,size=128)
        expected_demo=rng.integers(23456,size=128)
        actual=training_indices(1234,23456,update)
        np.testing.assert_array_equal(actual[0],expected_replay)
        np.testing.assert_array_equal(actual[1],expected_demo)

def test_nonresults_are_not_zero_success():
    assert arm_metrics([])==dict(status='NOT_RUN',episodes=0,success_rate=None)
    with pytest.raises(ValueError):matched_decision([],[],[],1,PLAN)

def test_one_more_success_is_not_evidence_of_improvement():
    result=decision([0]*32,[1]+[0]*31)
    assert result['status']=='INCONCLUSIVE' and result['extension_allowed']
    assert result['exact_discordant_p']==1
    assert not decision([0]*64,[1]+[0]*63,2)['extension_allowed']

def test_go_and_no_go_are_symmetric():
    assert decision([0]*32,[1]*16+[0]*16)['status']=='GO'
    assert decision([1]*16+[0]*16,[0]*32)['status']=='NO_GO'

def test_same_successes_and_changed_or_fake_starts_rejected():
    assert decision([1]*8+[0]*24,[1]*8+[0]*24)['status']=='INCONCLUSIVE'
    good=rows([0]*32)
    for key,value in [('seed',999),('source','cpu_test'),('stable_success',2)]:
        bad=[dict(r) for r in good];bad[0][key]=value
        with pytest.raises(ValueError):matched_decision(good,bad,list(range(32)),1,PLAN)
    bad=[dict(r) for r in good];bad[0]['seed']=1
    with pytest.raises(ValueError):matched_decision(good,bad,list(range(32)),1,PLAN)
    with pytest.raises(ValueError):decision([0]*32,[0]*32,3)
