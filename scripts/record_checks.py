"""Run required local checks and preserve exact commands/status without synthetic metrics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,'src')
from forcecritic.runtime import Run,atomic_json,command,digest,source_hashes
os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
r=Run('configs/minimal_validation.yaml',json.loads(Path('reports/PROJECT_STATUS.json').read_text())['run_id'])
checks=[]
for name,cmd in [('cpu_tests',[sys.executable,'-m','pytest','-q']),('diff_check',['git','diff','--check'])]:
    start=time.time();result=command(cmd,120);result['elapsed_seconds']=time.time()-start
    checks.append({'name':name,**result})
    (r.report/f'{name}.log').write_text(result.get('stdout','')+result.get('stderr',''))
atomic_json(r.report/'checks.json',{'checks':checks,'source_hashes':source_hashes(),
                                  'test_hashes':{str(p):digest(p) for p in Path('tests').rglob('*.py')}})
print(json.dumps([{k:v for k,v in c.items() if k not in ('stdout','stderr')} for c in checks],indent=2))
raise SystemExit(0 if all(c['returncode']==0 for c in checks) else 1)
