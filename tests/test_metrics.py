import pytest
from forcecritic.evaluation.metrics import summary_rows,paired_comparisons


def row(arm,parent,success,**extra):
    return dict(arm=arm,parent_episode=parent,seed='0',split='test',policy_sha=arm+'-fixed',status='COMPLETED',
                stable_success=str(success),ever_success=str(success),final_success=str(success),
                overload='0',force_peak='20',torque_peak='2',completion_time='1',env_steps='10',**extra)


def test_failed_episodes_remain_in_load_summary_and_repeats_not_independent():
    a=row('B1','p1',1);b=row('B1','p2',0);b['force_peak']='100'
    rows=summary_rows([a,a,b]);r=next(x for x in rows if x['arm']=='B1')
    assert r['episodes']==3 and r['independent_parents']==2
    assert r['force_peak_mean_all_episodes']==pytest.approx(140/3)
    assert r['successful_completion_time_denominator']==2


def test_versions_and_splits_not_silently_pooled():
    a=row('B1','p1',1);b=row('B1','p2',0);b['policy_sha']='different'
    assert len([r for r in summary_rows([a,b]) if r['arm']=='B1'])==2
    a['stable_success']='nan'
    with pytest.raises(ValueError,match='nonfinite'):summary_rows([a])


def test_pairing_rejects_selective_overlap_and_reports_true_difference():
    a=row('B2','p1',0);b=row('B3','p2',1)
    r=next(r for r in paired_comparisons([a,b]) if r['comparison']=='B3-B2')
    assert r['status']=='INCONCLUSIVE' and r['success_difference_pp']=='NA'
    b['parent_episode']='p1'
    r=next(r for r in paired_comparisons([a,b]) if r['comparison']=='B3-B2')
    assert r['success_difference_pp']==100 and r['paired_parents']==1
    assert r['success_ci_low_pp']=='NA'
