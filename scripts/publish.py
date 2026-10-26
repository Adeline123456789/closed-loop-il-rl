"""Conservative local Git publication. Connector publication is performed by Codex separately."""
import json
import os
from pathlib import Path
import subprocess
import sys
import re
import hashlib

os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['GIT_TERMINAL_PROMPT']='0'
ROOTS=['src','scripts','tests','configs','reports','results','pyproject.toml','.gitignore',
       'README.md','PROTOCOL_STAGE_V3.md','TRAINING_GUIDE.md','DATA_FORMAT.md','EXPERIMENT_PROGRESS.md','CODEX_TASK.md','AGENTS.md','THIRD_PARTY_NOTICES.md','third_party',
       'environment.lock.json','upstream.lock.json','protocol_lock.json','artifacts_manifest.json']
def run(args, timeout=300):
    try:
        return subprocess.run(args,text=True,capture_output=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        return subprocess.CompletedProcess(args,124,'',f'Timed out after {timeout}s: {args[0]}')
if run(['git','branch','--show-current']).stdout.strip()!='codex/force-critic-act-v2':
    raise SystemExit('Refuse publication outside task branch')
tests=run([sys.executable,'-m','pytest','-q'])
print(tests.stdout+tests.stderr)
if tests.returncode:raise SystemExit('CPU tests failed; refuse publication')
check=run(['git','diff','--check'])
if check.returncode:raise SystemExit(check.stdout+check.stderr)
paths=[p for p in ROOTS if Path(p).exists()]
for root in paths:
    p=Path(root)
    files=[p] if p.is_file() else p.rglob('*')
    for f in files:
        if not f.is_file() or '__pycache__' in str(f):continue
        if f.stat().st_size>5_000_000:raise SystemExit(f'Unexpected large artifact {f}')
        text=f.read_text(errors='replace')
        if any(re.search(p,text) for p in [r'gh[pousr]_[A-Za-z0-9]{30,}',r'github_pat_[A-Za-z0-9_]{30,}',r'hf_[A-Za-z0-9]{30,}',r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----']):
            raise SystemExit(f'Secret-like content in {f}; refuse publication')
# Only explicit task paths; raw data/cache/credentials are never staged.
r=run(['git','add','--',*paths])
if r.returncode:raise SystemExit(r.stderr)
# Hash the exact staged bytes, even when another process updates a local log.
artifacts={}
for entry in subprocess.check_output(['git','ls-files','--stage','-z','--','reports','results']).split(b'\0'):
    if not entry:continue
    metadata,path=entry.split(b'\t',1);blob=metadata.split()[1].decode()
    payload=subprocess.check_output(['git','cat-file','blob',blob])
    artifacts[path.decode()]={'sha256':hashlib.sha256(payload).hexdigest(),'bytes':len(payload)}
Path('artifacts_manifest.json').write_text(json.dumps({'run_id':'20261023T104759Z','additional_runs':['stage_v3_20261026'] if Path('reports/stage_v3_20261026').exists() else [],'artifacts':artifacts,'weights':[]},indent=2)+'\n')
r=run(['git','add','--','artifacts_manifest.json'])
if r.returncode:raise SystemExit(r.stderr)
if run(['git','diff','--cached','--quiet']).returncode:
    r=run(['git','commit','-m','Record force critic ACT implementation and measured experiment evidence'])
    if r.returncode:raise SystemExit(r.stderr)
r=run(['git','-c','http.version=HTTP/1.1','-c','http.lowSpeedLimit=1','-c','http.lowSpeedTime=30',
       'push','origin','HEAD:refs/heads/codex/force-critic-act-v2'],timeout=180)
print(r.stdout+r.stderr)
if r.returncode:
    bundle=Path('.cache/force-critic-act-v2.bundle');bundle.parent.mkdir(exist_ok=True)
    run(['git','bundle','create',str(bundle),'codex/force-critic-act-v2'])
    print('PUBLICATION_FAILED; local commit and .cache bundle retained. See actual transport error above.')
raise SystemExit(r.returncode)
