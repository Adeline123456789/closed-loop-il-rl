"""Actual robot-stage entry point. Launcher precedes every simulator import."""
import argparse
from pathlib import Path
import json

parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['baseline','diagnostic','short-ablation'], required=True)
parser.add_argument('--config', required=True)
parser.add_argument('--run-id', required=True)
parser.add_argument('--demonstrations', required=True)
parser.add_argument('--outcome', required=True)
parser.add_argument('--paired-bundle')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.headless or not args.enable_cameras:
    parser.error('--headless --enable_cameras required')
app = AppLauncher(args).app
from forcecritic.runtime import Run, atomic_json
run = Run(args.config, args.run_id)
outcome = Path(args.outcome)
ledgers = []
try:
    import gymnasium as gym
    import isaaclab_tasks
    from isaaclab_tasks.utils import parse_env_cfg
    from forcecritic.envs.forge import ForgeAdapter, configure_camera, TASK
    from forcecritic.envs.accounting import PhysicsLedger
    from forcecritic.training.baseline import baseline
    def make_adapter(num_envs, limits):
        cfg = configure_camera(parse_env_cfg(TASK, device=args.device, num_envs=num_envs))
        cfg.seed = run.config['seed']
        cap = (run.config['diagnostic']['all_diagnostic_transition_cap'] if args.stage=='diagnostic'
               else 50000 if num_envs == 8 else 100000)
        if args.stage=='diagnostic':
            previous = 0
            for path in run.report.glob('diagnostic_*_outcome.accounting.json'):
                previous += sum(json.loads(path.read_text())['individual_control_step_equivalents'].values())
            cap -= previous
            if cap<=0:raise RuntimeError('BUDGET_EXHAUSTED by earlier diagnostic attempts')
        ledger = PhysicsLedger(num_envs, cfg.decimation, cap)
        ledgers.append(ledger)
        env = gym.make(TASK, cfg=cfg)
        acceptance = json.loads((run.report/'p0_acceptance.json').read_text())
        return ForgeAdapter(env, limits=limits, hard_limits=acceptance['physical_protection_limits'], physics_ledger=ledger), ledger
    if args.stage=='baseline':
        result = baseline(run, args.demonstrations, make_adapter, device=args.device)
    elif args.stage=='diagnostic':
        from forcecritic.diagnostics.driver import diagnostic
        result = diagnostic(run,args.demonstrations,make_adapter,device=args.device)
    else:
        from forcecritic.training.short_live import short_ablation
        if not args.paired_bundle:
            raise ValueError('--paired-bundle required for measured paired supervision')
        result = short_ablation(run, args.demonstrations, args.paired_bundle, make_adapter, device=args.device)
    atomic_json(outcome, result)
except BaseException as error:
    import traceback
    traceback.print_exc()
    atomic_json(outcome, {'status':'FAILED_IMPLEMENTATION', 'error':repr(error)})
    raise
finally:
    categories = ('restore','interaction','branch','prefix','evaluation')
    records = [ledger.record() for ledger in ledgers]
    atomic_json(outcome.with_suffix('.accounting.json'),{
        'individual_control_step_equivalents':{k:sum(r['individual_control_step_equivalents'][k] for r in records) for k in categories},
        'instances':records})
    app.close()
