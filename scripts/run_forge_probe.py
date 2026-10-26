"""Run a real P0 probe with immutable command/source evidence and step accounting."""
import argparse
import json
import sys
import time
from pathlib import Path
from forcecritic.runtime import Run, command, atomic_json, source_hashes

p = argparse.ArgumentParser()
p.add_argument('probe', choices=['audit', 'collect'])
p.add_argument('--run-id', required=True)
p.add_argument('--output', required=True)
args, extra = p.parse_known_args()
run = Run('configs/minimal_validation.yaml', args.run_id)
name = f'p0_{args.probe}_{time.time_ns()}'
launcher = json.loads(Path('configs/isaac_launcher.json').read_text())
cmd = [sys.executable, '-u', f'scripts/{args.probe}_forge.py', '--output', args.output,
       '--headless', '--enable_cameras', '--kit_args='+launcher['kit_args'], *extra]
sources = source_hashes()
record = command(cmd, timeout=7200, log_path=run.report/f'{name}.log')
record['execution_source_hashes'] = sources
atomic_json(run.report/f'{name}.command.json', record)
measurement_path = Path(args.output)/('contract_audit.json' if args.probe=='audit' else 'manifest.json')
measurement = json.loads(measurement_path.read_text()) if measurement_path.exists() else {}
accounting = measurement.get('accounting', {})
counts = accounting.get('individual_control_step_equivalents', {})
# Reload after a long subprocess so independently finished evidence is retained.
run = Run('configs/minimal_validation.yaml', args.run_id)
for key, value in counts.items():
    run.manifest['steps'][key] += value
run.manifest.setdefault('p0_probes', []).append(dict(name=name, command_record=str(run.report/f'{name}.command.json'),
    measurement=str(measurement_path), returncode=record.get('returncode'), accounting=accounting))
run.save()
print(json.dumps({'name':name,'returncode':record.get('returncode'),'measurement':str(measurement_path)}))
sys.exit(0 if record.get('returncode')==0 and measurement.get('status') in
         ('MEASURED_NOT_YET_ACCEPTED','COLLECTED','BUDGET_EXHAUSTED') else 2)
