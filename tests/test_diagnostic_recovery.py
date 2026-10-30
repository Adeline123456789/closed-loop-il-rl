import json
from forcecritic.diagnostics.recovery import recover_accounting


def test_native_crash_charges_before_step_even_without_a_finished_trajectory(tmp_path):
    journal=tmp_path/'physics_ledger.jsonl'
    journal.write_text(json.dumps(dict(event='before',counts={'branch':17,'restore':8}))+'\n')
    result=recover_accounting(dict(journal_path=str(journal),accounting=dict(decimation=8,num_envs=1)))
    assert result['individual_control_step_equivalents']=={'branch':3,'restore':1}
    assert result['pending_physics_substeps']==1
    assert not result['unresolved_unflushed_steps']


def test_absent_native_accounting_stays_unknown():
    result=recover_accounting(dict(accounting=dict(decimation=8,num_envs=1,physics_substeps={'branch':16})))
    assert result['status']=='LOWER_BOUND_ONLY'
    assert result['individual_control_step_equivalents']['branch']==2
    assert result['pending_physics_substeps'] is None
    assert result['unresolved_unflushed_steps']
