import copy
import random
import numpy as np
import torch
from torch import nn
import pytest
from forcecritic.training.checkpoint import save_checkpoint,restore_checkpoint
from forcecritic.training.replay import Transition,Replay,ReturnScale
from forcecritic.evaluation.gates import decide
from forcecritic.training.updates import critic_step,td_target


def test_checkpoint_restores_optimizer_rng_and_progress(tmp_path):
    a=nn.Linear(2,2);q=nn.Linear(2,2)
    ao=torch.optim.Adam(a.parameters());qo=torch.optim.Adam(q.parameters())
    ao.zero_grad();a(torch.ones(1,2)).sum().backward();ao.step()
    path=tmp_path/'resume.pt'
    save_checkpoint(path,a,q,ao,qo,update=7,config_hash='c',policy_sha='p',ledger={'prefix':8})
    expected=(torch.rand(3),np.random.rand(3),random.random())
    with torch.no_grad():a.weight.fill_(999)
    state=restore_checkpoint(path,a,q,ao,qo,'c')
    assert state['update']==7 and state['ledger']['prefix']==8
    assert torch.equal(expected[0],torch.rand(3))
    np.testing.assert_array_equal(expected[1],np.random.rand(3))
    assert expected[2]==random.random()
    assert a.weight.max()<999
    assert len(ao.state)>0
    with pytest.raises(ValueError):restore_checkpoint(path,a,q,ao,qo,'different')


def test_replay_rejects_hf_fabrication_and_nonprefix():
    args=dict(parent_id='p',split='train',behavior_sha='sha',observation={},plan=np.zeros((4,4)),
              valid=np.array([1,1,0,0],bool),returns=np.zeros(2),next_observation={},bootstrap=False,gamma_k=.99**2)
    replay=Replay(10)
    replay.append(Transition(**args))
    assert len(replay.sample(3))==3
    with pytest.raises(ValueError):replay.append(Transition(**args,source='hf'))
    args['valid']=np.array([1,0,1,0],bool)
    with pytest.raises(ValueError):replay.append(Transition(**args))


def test_zero_signal_critic_cannot_enter_calibration():
    with pytest.raises(ValueError,match='NO_REWARD_VARIATION'):ReturnScale().fit([[0,0],[0,1]],['train']*2)
    with pytest.raises(ValueError,match='train only'):ReturnScale().fit([[0,0],[1,1]],['train','test'])


def test_gate_requires_measured_evidence_and_load_safety():
    assert decide({})['status']=='INCONCLUSIVE'
    e=dict(environment_passed=True,baseline_passed=True,replay_passed=False,hard_total=16,hard_identifiable=4,
        critic_judgment_problem=True,critic_is_primary_cause=True,independent_prediction_improved=True,
        actual_actor_improved=True,success_difference_pp=6,success_difference_ci=[1,11],overload_worse=False,
        peak_load_worse=False,baseline_overload_events=12,overload_relative_reduction=.25,intervals_informative=True)
    assert decide(e)['status']=='NO_GO_CALIBRATION'
    e['replay_passed']=True
    assert decide(e)['status']=='GO_PILOT'
    e['overload_worse']=True
    assert decide(e)['status']=='NO_GO_CALIBRATION'
    e['overload_worse']=False;e['success_difference_pp']=-2;e['baseline_overload_events']=1
    assert decide(e)['status']=='INCONCLUSIVE'


def test_td_terminal_and_target_no_grad():
    class Actor(nn.Module):
        def forward(self,x):return x
    class Critic(nn.Module):
        def forward(self,h,a):return (a+h)[:,None,:].expand(-1,2,-1)
    x=torch.ones(2,2,requires_grad=True)
    target=td_target(Actor(),Critic(),x,x,torch.zeros(2,2),torch.tensor([.81,.9]),torch.tensor([0.,1.]))
    assert not target.requires_grad
    torch.testing.assert_close(target,torch.tensor([[0.,0.],[1.8,1.8]]))


def test_not_run_report_does_not_fabricate_zero_success(tmp_path,monkeypatch):
    from forcecritic.runtime import Run
    from forcecritic.evaluation.report import report
    import csv
    monkeypatch.chdir(tmp_path)
    Path=__import__('pathlib').Path
    Path('config.yaml').write_text('protocol: force_critic_act_v2_diagnostic_first\n')
    run=Run('config.yaml','test-run')
    report(run)
    rows=list(csv.DictReader((run.result/'summary.csv').open()))
    assert len(rows)==4
    assert all(r['status']=='NOT_RUN' and r['stable_success_rate']=='NA' for r in rows)


def test_resume_rejects_changed_artifact_and_config(tmp_path,monkeypatch):
    from forcecritic.runtime import Run
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    Path('config.yaml').write_text('protocol: force_critic_act_v2_diagnostic_first\n')
    run=Run('config.yaml','resume')
    evidence=Path('evidence.json');evidence.write_text('{}')
    run.record('doctor','PASSED',artifacts=[evidence])
    assert run.verified('doctor')
    evidence.write_text('{"changed":true}')
    assert not run.verified('doctor')
    Path('scripts').mkdir()
    launcher=Path('scripts/smoke.py');launcher.write_text('version = 1\n')
    run.record('smoke-1','PASSED',artifacts=[evidence])
    assert run.verified('smoke-1')
    launcher.write_text('version = 2\n')
    assert not run.verified('smoke-1')
    Path('config.yaml').write_text('protocol: force_critic_act_v2_diagnostic_first\nseed: 1\n')
    with pytest.raises(ValueError,match='resume config changed'):Run('config.yaml','resume')


def test_live_command_log_preserves_nonzero_exit(tmp_path):
    import sys
    from forcecritic.runtime import command
    log=tmp_path/'launcher.log'
    result=command([sys.executable,'-c','import sys; print("launch failed", flush=True); sys.exit(3)'],log_path=log)
    assert result['returncode']==3
    assert log.read_text()=='launch failed\n'
    assert result['stdout']=='launch failed\n'


def test_smoke_evidence_ignores_unexecuted_training_but_not_launcher(tmp_path,monkeypatch):
    from forcecritic.runtime import Run
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    Path('config.yaml').write_text('protocol: force_critic_act_v2_diagnostic_first\n')
    Path('scripts').mkdir()
    Path('src/forcecritic/training').mkdir(parents=True)
    launcher=Path('scripts/smoke_forge.py');launcher.write_text('version=1\n')
    training=Path('src/forcecritic/training/later.py');training.write_text('version=1\n')
    evidence=Path('measured.json');evidence.write_text('{}')
    run=Run('config.yaml','scope')
    run.record('smoke-1','PASSED',artifacts=[evidence])
    training.write_text('version=2\n')
    assert run.verified('smoke-1')
    launcher.write_text('version=2\n')
    assert not run.verified('smoke-1')


def test_edit_during_launch_cannot_be_recorded_as_current_pass(tmp_path,monkeypatch):
    from forcecritic.runtime import Run,source_hashes
    from pathlib import Path
    monkeypatch.chdir(tmp_path)
    Path('config.yaml').write_text('protocol: force_critic_act_v2_diagnostic_first\n')
    Path('scripts').mkdir()
    launcher=Path('scripts/smoke_forge.py');launcher.write_text('version=1\n')
    evidence=Path('measured.json');evidence.write_text('{}')
    run=Run('config.yaml','during-launch')
    executed=source_hashes()
    launcher.write_text('version=2\n')
    run.record('smoke-1','PASSED',artifacts=[evidence],executed_sources=executed)
    assert run.manifest['stages']['smoke-1']['status']=='STALE_SOURCE'
    assert not run.verified('smoke-1')
