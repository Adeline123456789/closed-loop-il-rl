"""Bound and journal engineering probes independently of historical D0 results."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from forcecritic.diagnostics.artifacts import atomic_json
from forcecritic.diagnostics.recovery import recover_accounting
from forcecritic.runtime import digest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('label')
parser.add_argument('--variant', default='legacy')
parser.add_argument('--steps', type=int, default=16)
parser.add_argument('--repeats', type=int, default=3)
parser.add_argument('--render-profile', choices=['stock','rtpt','rtpt_forced'], default='stock')
parser.add_argument('--dataset', action='store_true')
parser.add_argument('--omit-expert-rng', action='store_true')
parser.add_argument('--manifest')
parser.add_argument('--control-step-cap', type=int)
args = parser.parse_args()
plan_path = Path('configs/replay_root_cause_20261027.json')
plan = json.loads(plan_path.read_text())
root = Path('reports/replay_root_cause_20261027')
root.mkdir(exist_ok=True)
out = root / args.label
if out.exists() or out.with_suffix('.command.json').exists():
    raise ValueError('Never overwrite an existing probe')
# Charge completed write-ahead journals exactly; retain full reservations when unknown.
reservations = list(root.glob('*.command.json'))
used = 0.
for path in reservations:
    prior = json.loads(path.read_text())
    account = prior.get('accounting', {})
    if prior.get('status') == 'FINISHED' and account.get('status') == 'CHARGED_STEPS_RECOVERED':
        used += sum(account['physics_substeps'].values())/8
    else:
        used += prior['reserved_control_steps']
process_cap = plan['per_process_control_step_cap']
if args.dataset:
    plan_path = Path('configs/replay_dataset_reconstruction_v1.json')
    process_cap = json.loads(plan_path.read_text())['per_process_control_step_cap']
if args.control_step_cap is not None:
    if not 1 <= args.control_step_cap <= process_cap:
        raise ValueError('Requested cap exceeds registered maximum')
    process_cap = args.control_step_cap
if used+process_cap > plan['total_control_step_cap']:
    raise RuntimeError('Engineering budget exhausted; no automatic extension')
cmd = [sys.executable, '-u', 'scripts/probe_replay_root_cause.py', '--output', str(out),
       '--variant', args.variant, '--steps', str(args.steps), '--repeats', str(args.repeats),
       '--render-profile', args.render_profile,
       '--control-step-cap', str(process_cap),
       '--headless', '--enable_cameras', '--kit_args='+json.loads(Path('configs/isaac_launcher.json').read_text())['kit_args']]
if args.dataset:
    cmd = [sys.executable,'-u','scripts/replay_demonstrations.py','--output',str(out),
           '--steps',str(args.steps),'--headless','--enable_cameras',
           '--control-step-cap',str(process_cap),
           '--kit_args='+json.loads(Path('configs/isaac_launcher.json').read_text())['kit_args']]
    if args.omit_expert_rng: cmd.append('--omit-expert-rng')
    if args.manifest: cmd += ['--manifest',args.manifest]
record = dict(command=cmd, started=time.time(), plan_sha256=digest(plan_path),
              reserved_control_steps=process_cap, charged_or_reserved_before=used, status='RUNNING')
atomic_json(out.with_suffix('.command.json'), record)
with out.with_suffix('.log').open('x') as log:
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        code = proc.wait(timeout=300)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGTERM)
        try: code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            code = proc.wait()
        record['timeout'] = True
measurement_path = out/'result.json'
measurement = json.loads(measurement_path.read_text()) if measurement_path.exists() else {}
record.update(returncode=code, finished=time.time(), status='FINISHED',
              measurement_status=measurement.get('status', 'MISSING'))
progress_path = out/'diagnostic_live_progress.json'
if progress_path.exists():
    record['accounting'] = recover_accounting(json.loads(progress_path.read_text()))
atomic_json(out.with_suffix('.command.json'), record)
print(json.dumps(record), flush=True)
sys.exit(0 if code == 0 and measurement.get('status') == 'MEASURED' else 1)
