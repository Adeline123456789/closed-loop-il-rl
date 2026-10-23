"""Invoke actual subprocess CLIs; exit 2 is an observed prerequisite block, not a robot run."""
import json
from pathlib import Path
import subprocess
import sys
rid=json.loads(Path('reports/PROJECT_STATUS.json').read_text())['run_id']
results=[]
for stage in ['baseline','diagnostic','short-ablation','pilot','confirm']:
    cmd=[sys.executable,'-m','forcecritic.cli','run','--stage',stage,'--config','configs/minimal_validation.yaml']
    if stage in ['pilot','confirm']:cmd+=['--require-go',f'reports/{rid}/decision.json']
    p=subprocess.run(cmd,capture_output=True,text=True)
    results.append({'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
cmd=[sys.executable,'-m','forcecritic.cli','evaluate','--suite','generalization','--no-training']
p=subprocess.run(cmd,capture_output=True,text=True)
results.append({'command':cmd,'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr})
Path(f'reports/{rid}/stage_gate_invocations.json').write_text(json.dumps(results,indent=2)+'\n')
print(json.dumps(results,indent=2))
