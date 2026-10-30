"""Monitoring regression checks, never robot performance evidence."""
import importlib.util
import json
import os
from pathlib import Path
import time


def monitor(tmp_path, monkeypatch):
    spec=importlib.util.spec_from_file_location('direct_q_status',Path('scripts/training_status.py'))
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for name in ('REPORT','RESULT','CHECKPOINT'):
        path=tmp_path/name
        path.mkdir()
        monkeypatch.setattr(module,name,path)
    monkeypatch.setattr(module,'ROOT',tmp_path)
    (module.REPORT/'background_job.json').write_text(json.dumps({'stage':'diagnostic','status':'STOPPED','pid':None}))
    (module.REPORT/'diagnostic_live_progress.json').write_text(json.dumps({'last_physics_step':time.time()-3600}))
    return module


def test_new_real_evaluation_does_not_inherit_old_diagnostic_stall(tmp_path,monkeypatch):
    module=monitor(tmp_path,monkeypatch)
    (module.REPORT/'direct_q_job.json').write_text(json.dumps(dict(
        status='RUNNING',phase='direct_B0',pid=os.getpid(),started=time.time())))
    data=module.snapshot()
    assert data['supervisor_alive']
    assert data['execution_evidence_age_seconds']<2
    assert data['progress_warning'] is None
    assert data['queue']['diagnostic']=='NOT_RUN'


def test_completed_ordinary_control_does_not_pass_paired_gate(tmp_path,monkeypatch):
    module=monitor(tmp_path,monkeypatch)
    (module.REPORT/'run_manifest.json').write_text(json.dumps({'stages':{'diagnostic':{'status':'NO_GO_REPLAY'}}}))
    (module.REPORT/'direct_q_job.json').write_text(json.dumps(dict(
        status='COMPLETED',phase='real_evaluations_complete',pid=os.getpid())))
    data=module.snapshot()
    assert not data['supervisor_alive']
    assert '普通直接 Q 对照已完成' in data['phase']
    assert data['queue']['diagnostic']=='NO_GO_REPLAY'
    assert data['queue']['short-ablation']=='NOT_RUN'
