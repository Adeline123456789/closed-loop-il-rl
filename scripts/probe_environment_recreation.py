"""Bounded real FORGE reproduction of sequential 1-env -> 8-env creation."""
import argparse
import faulthandler
import json
from pathlib import Path
import time

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',required=True)
parser.add_argument('--single-env-count',type=int,choices=[1,8])
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args=parser.parse_args()
app=AppLauncher(args).app
out=Path(args.output);out.parent.mkdir(parents=True,exist_ok=True)
stack=out.with_suffix('.stack.log').open('w')
faulthandler.enable(file=stack)
faulthandler.dump_traceback_later(35,repeat=True,file=stack)
import gymnasium as gym
import isaaclab_tasks
from isaaclab_tasks.utils import parse_env_cfg
from forcecritic.envs.forge import TASK,configure_camera,ForgeAdapter
from forcecritic.envs.accounting import PhysicsLedger
from forcecritic.runtime import atomic_json
import numpy as np
records=[]
def save(phase):
    atomic_json(out,dict(phase=phase,time=time.time(),instances=[x.record() for x in records]))
    print('PROBE',phase,flush=True)
try:
    for n in ((args.single_env_count,) if args.single_env_count else (1,8)):
        cfg=configure_camera(parse_env_cfg(TASK,device=args.device,num_envs=n));cfg.seed=0
        ledger=PhysicsLedger(n,cfg.decimation,10000);records.append(ledger)
        save(f'create_{n}_start')
        env=gym.make(TASK,cfg=cfg)
        save(f'create_{n}_finished')
        adapter=ForgeAdapter(env,physics_ledger=ledger)
        adapter.reset(seed=123)
        with ledger.using('interaction'):
            adapter.step(np.zeros((n,4),dtype=np.float32))
        save(f'step_{n}_finished')
        adapter.close();ledger.close()
        save(f'close_{n}_finished')
    save('PASSED')
finally:
    faulthandler.cancel_dump_traceback_later()
    stack.close()
    app.close()
