"""Recovery preserves measured evidence and counts, never invents robot outcomes."""
import csv
import json
from pathlib import Path
import runpy
from types import SimpleNamespace
import pytest
from forcecritic.runtime import digest
from forcecritic.training.baseline import reused_validation


def test_validation_resume_rejects_changed_csv(tmp_path):
    checkpoint=tmp_path/'act0.pt';checkpoint.write_bytes(b'local checkpoint fixture')
    csv_path=tmp_path/'act0_validation.csv'
    with csv_path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=['seed','policy_sha','source','stable_success'])
        writer.writeheader()
        writer.writerows(dict(seed=s,policy_sha=digest(checkpoint),source='real_forge',stable_success=0)
                         for s in range(20000,20032))
    receipt=dict(status='ACCEPTED_FOR_RESUME',config_hash='c',checkpoint_sha256=digest(checkpoint),
                 evidence_hashes={str(csv_path):digest(csv_path)})
    (tmp_path/'baseline_validation_resume.json').write_text(json.dumps(receipt))
    run=SimpleNamespace(report=tmp_path,result=tmp_path,config_hash='c')
    assert len(reused_validation(run,checkpoint))==32
    csv_path.write_text(csv_path.read_text().replace(',0\n',',1\n',1))
    with pytest.raises(ValueError,match='evidence changed'):
        reused_validation(run,checkpoint)


def test_isolated_phases_account_once():
    module=runpy.run_path(str(Path(__file__).resolve().parents[1]/'scripts/run_robot_stage.py'))
    result=module['aggregate']([
        {'individual_control_step_equivalents':{'restore':3,'evaluation':8},'instances':[{'phase':'evaluation'}]},
        {'individual_control_step_equivalents':{'restore':7,'interaction':16},'instances':[{'phase':'shared'}]}])
    assert result['individual_control_step_equivalents']==dict(restore=10,evaluation=8,interaction=16,branch=0,prefix=0)
    assert len(result['instances'])==2
