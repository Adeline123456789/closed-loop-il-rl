"""Immutable real FORGE batches for the matched stage-v3 comparison."""
import argparse
import json
from pathlib import Path
import time
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm',choices=('A1','A2'),required=True)
parser.add_argument('--batch',choices=('initial','extension'),required=True)
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import torch
from forcecritic.staged.common import locked_plan,CONFIG
from forcecritic.runtime import digest,atomic_json,source_hashes
from forcecritic.envs.forge import ForgeAdapter,configure_camera,TASK
from forcecritic.envs.accounting import PhysicsLedger
from forcecritic.training.baseline import accepted_limits
from forcecritic.training.pretrain import baseline_actor
from forcecritic.training.live import LivePolicy,evaluate
from forcecritic.diagnostics.live import NoiseSchedule
from forcecritic.diagnostics.progress import DiagnosticProgress

p=locked_plan();report=Path(p['report']);old=Path('reports/20261023T104759Z')
name=f'eval_{args.arm}_{args.batch}';folder=report/name
seeds=p['initial_evaluation_seeds'] if args.batch=='initial' else p['extension_seeds']
if args.arm=='A1':
    training=json.loads((report/'A1_training.json').read_text())
    path=Path(training['checkpoint']);expected=training['checkpoint_sha']
    if training['status']!='PASSED' or training['plan_sha']!=digest(CONFIG):raise ValueError('Training not accepted')
else:
    path=Path('checkpoints/20261023T104759Z/direct_Q.pt');expected=p['inputs'][str(path)]
if digest(path)!=expected:raise ValueError('Checkpoint changed')
checkpoint=torch.load(path,map_location='cpu',weights_only=False)
csv_path=Path(p['results'])/(name+'.csv')
if csv_path.exists():raise FileExistsError('Preserve every real evaluation attempt')
cfg=configure_camera(parse_env_cfg(TASK,device=args.device,num_envs=1));cfg.seed=0
ledger=PhysicsLedger(1,cfg.decimation,p['evaluation_control_step_cap_per_batch'])
observer=DiagnosticProgress(ledger,folder,Path(p['checkpoints'])/name)
observer.mark(phase='environment_initialization',arm=args.arm,batch=args.batch,target_episodes=len(seeds))
out=dict(status='RUNNING',arm=args.arm,batch=args.batch,policy_sha=expected,config_sha=digest(CONFIG),
    started=time.time(),seeds=seeds,source_hashes=source_hashes(),input_audit_sha=digest(report/'input_audit.json'))
try:
    env=gym.make(TASK,cfg=cfg)
    acceptance=json.loads((old/'p0_acceptance.json').read_text())
    adapter=ForgeAdapter(env,limits=accepted_limits(old/'p0_acceptance.json'),
        hard_limits=acceptance['physical_protection_limits'],physics_ledger=ledger)
    noise=NoiseSchedule(adapter,seeds[0])
    actor=baseline_actor().to(args.device);actor.load_state_dict(checkpoint['actor'])
    policy=LivePolicy(actor,checkpoint['identity']['normalizer'],args.device)
    observer.mark(phase='real_evaluation')
    rows=evaluate(adapter,policy,seeds,ledger,csv_path,arm=args.arm,policy_sha=expected,
        phase='ordinary_direct_q_validation',noise=noise)
    out.update(status='PASSED',episodes=len(rows),csv=str(csv_path),csv_sha=digest(csv_path))
except BaseException as error:
    out.update(status='FAILED_IMPLEMENTATION',error=repr(error));raise
finally:
    out.update(finished=time.time(),accounting=ledger.record())
    atomic_json(report/(name+'.json'),out)
    observer.mark(status=out['status'],phase='finished');observer.close()
    if 'noise' in globals():noise.close()
    if 'adapter' in globals():adapter.close()
    ledger.close();app.close()
