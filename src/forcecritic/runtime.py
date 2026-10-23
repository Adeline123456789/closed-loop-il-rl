"""Atomic evidence records, provenance hashes and resumable stage state."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import yaml

PROTOCOL = 'force_critic_act_v2_diagnostic_first'
STAGES = ('doctor','audit-data','smoke-1','smoke-8','baseline','diagnostic','short-ablation','pilot','confirm','generalization')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp'); temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n'); temp.replace(path)


def command(cmd, timeout=60):
    try:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=timeout,env={**os.environ,'GIT_TERMINAL_PROMPT':'0'})
        return {'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    except (OSError,subprocess.TimeoutExpired) as e:
        return {'command':cmd,'returncode':None,'error':str(e)}


class Run:
    def __init__(self, config, run_id=None):
        self.config_path=Path(config); self.config=yaml.safe_load(self.config_path.read_text())
        self.config_hash=digest(config)
        if self.config['protocol'] != PROTOCOL: raise ValueError('wrong protocol')
        self.run_id=run_id or datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.report=Path('reports')/self.run_id; self.result=Path('results')/self.run_id
        self.report.mkdir(parents=True,exist_ok=True);self.result.mkdir(parents=True,exist_ok=True)
        path=self.report/'run_manifest.json'
        if path.exists():
            self.manifest=json.loads(path.read_text())
            if self.manifest['config_hash'] != self.config_hash:raise ValueError('resume config changed: use a new run ID')
        else:
            self.manifest={'run_id':self.run_id,'protocol':PROTOCOL,'config_hash':self.config_hash,
                'stages':{s:{'status':'NOT_RUN'} for s in STAGES},'events':[],
                'steps':dict(interaction=0,branch=0,prefix=0,restore=0,evaluation=0), 'checkpoints':[]}
        self.save()

    def save(self):
        atomic_json(self.report/'run_manifest.json',self.manifest)
        atomic_json('reports/PROJECT_STATUS.json',{'run_id':self.run_id,'protocol':PROTOCOL,
                    'stages':self.manifest['stages'],'manifest':str(self.report/'run_manifest.json')})

    def verified(self, stage):
        event=self.manifest['stages'].get(stage,{})
        if event.get('status')!='PASSED' or event.get('config_hash')!=self.config_hash:return False
        for p,h in {**event.get('artifacts',{}),**event.get('source_hashes',{})}.items():
            if not Path(p).is_file() or digest(p)!=h:return False
        return bool(event.get('artifacts'))

    def record(self, stage, status, reason='', returncode=0, artifacts=()):
        event={'stage':stage,'status':status,'reason':reason,'returncode':returncode,
               'command':[sys.executable,'-m','forcecritic.cli',*sys.argv[1:]],
               'code_sha':command(['git','rev-parse','HEAD'])['stdout'].strip(),
               'dirty':bool(command(['git','status','--porcelain'])['stdout'].strip()),
               'source_hashes':{str(p):digest(p) for p in sorted(Path('src').rglob('*.py'))},
               'config_hash':self.config_hash,'time':datetime.datetime.now(datetime.timezone.utc).isoformat(),
               'artifacts':{str(p):digest(p) for p in artifacts if Path(p).is_file()}}
        self.manifest['stages'][stage]=event;self.manifest['events'].append(event);self.save()


def lock_protocol(config_path):
    config=yaml.safe_load(Path(config_path).read_text())
    lock={'protocol':PROTOCOL,'config_hash':digest(config_path),'config':config,
          'locked_before_test':True,'physical_thresholds_status':'UNRESOLVED_DO_NOT_RUN',
          'rules':{'replay_failure':'NO_GO_CALIBRATION','hard_coverage':[.25,4],
                   'pilot_success_gain_pp':5,'alternative_noninferiority_pp':-3,'overload_reduction':.2,
                   'minimum_baseline_overload_events':10,'low_event_or_wide_interval':'INCONCLUSIVE','no_test_labels_for_training':True}}
    path=Path('protocol_lock.json')
    if path.exists() and json.loads(path.read_text()) != lock: raise ValueError('protocol lock differs; explicit revision required')
    atomic_json(path,lock)
