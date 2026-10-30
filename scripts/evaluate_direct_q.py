"""Independent real FORGE evaluation of the permitted ordinary direct-Q fallback."""
import argparse
import json
from pathlib import Path
import time
from isaaclab.app import AppLauncher

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arm',required=True,choices=('direct_B0','direct_Q'))
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
import torch
from forcecritic.runtime import Run,digest,atomic_json,source_hashes
from forcecritic.envs.forge import ForgeAdapter,configure_camera,TASK
from forcecritic.envs.accounting import PhysicsLedger
from forcecritic.training.baseline import accepted_limits
from forcecritic.training.pretrain import baseline_actor
from forcecritic.training.live import LivePolicy,evaluate
from forcecritic.diagnostics.live import NoiseSchedule
from forcecritic.diagnostics.progress import DiagnosticProgress

run=Run('configs/minimal_validation.yaml','20261023T104759Z')
plan_path=run.report/'direct_q_plan.json'
plan=json.loads(plan_path.read_text())
directory=Path('checkpoints')/run.run_id
path=directory/('act0.pt' if args.arm=='direct_B0' else 'direct_Q.pt')
checkpoint=torch.load(path,map_location='cpu',weights_only=False)
if args.arm=='direct_Q' and (checkpoint['actor_updates']!=plan['actor_updates'] or checkpoint['plan_sha']!=digest(plan_path)):
    raise ValueError('Incomplete or mismatched direct-Q checkpoint')
if args.arm=='direct_B0' and digest(path)!=plan['act_sha']:raise ValueError('ACT0 changed')
csv_path=run.result/f'{args.arm}_validation.csv'
if csv_path.exists():raise FileExistsError('Never overwrite real evaluation attempts')
cfg=configure_camera(parse_env_cfg(TASK,device=args.device,num_envs=1))
cfg.seed=0
ledger=PhysicsLedger(1,cfg.decimation,10000)
observer=DiagnosticProgress(ledger,run.report/args.arm,directory/args.arm)
observer.mark(phase='environment_initialization',arm=args.arm,completed_episodes=0,target_episodes=32)
result=dict(status='RUNNING',arm=args.arm,policy_sha=digest(path),plan_sha=digest(plan_path),
    source_hashes=source_hashes(),started=time.time(),renderer='Original baseline configuration',
    reset_protocol='NoiseSchedule with controller targets and interval timers reset; identical for both arms')
try:
    env=gym.make(TASK,cfg=cfg)
    acceptance=json.loads((run.report/'p0_acceptance.json').read_text())
    adapter=ForgeAdapter(env,limits=accepted_limits(run.report/'p0_acceptance.json'),
        hard_limits=acceptance['physical_protection_limits'],physics_ledger=ledger)
    noise=NoiseSchedule(adapter,plan['evaluation_seeds'][0])
    actor=baseline_actor().to(args.device)
    actor.load_state_dict(checkpoint['actor'])
    policy=LivePolicy(actor,checkpoint['identity']['normalizer'],args.device)
    observer.mark(phase='real_evaluation')
    rows=evaluate(adapter,policy,plan['evaluation_seeds'],ledger,csv_path,
        arm=args.arm,policy_sha=result['policy_sha'],phase='ordinary_direct_q_validation',noise=noise)
    result.update(status='COMPLETED',episodes=len(rows),successes=sum(r['stable_success'] for r in rows),
        csv=str(csv_path),csv_sha=digest(csv_path))
except BaseException as error:
    result.update(status='FAILED_IMPLEMENTATION',error=repr(error))
    raise
finally:
    result.update(finished=time.time(),accounting=ledger.record())
    atomic_json(run.report/f'{args.arm}_evaluation.json',result)
    observer.mark(status=result['status'],phase='finished')
    observer.close()
    if 'noise' in globals():noise.close()
    if 'adapter' in globals():adapter.close()
    ledger.close()
    app.close()
