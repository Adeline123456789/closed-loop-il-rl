"""Conservative local Git publication. Connector publication is performed by Codex separately."""
import json
import os
from pathlib import Path
import subprocess
import sys
import re

os.environ['PYTEST_DISABLE_PLUGIN_AUTOLOAD']='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
ROOTS=['src','scripts','tests','configs','reports','results','pyproject.toml','.gitignore',
       'README.md','CODEX_TASK.md','AGENTS.md','THIRD_PARTY_NOTICES.md','third_party',
       'environment.lock.json','upstream.lock.json','protocol_lock.json','artifacts_manifest.json']
def run(args):return subprocess.run(args,text=True,capture_output=True)
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
if run(['git','diff','--cached','--quiet']).returncode:
    r=run(['git','commit','-m','Implement diagnostic-first force critic ACT and record measured P0 stop'])
    if r.returncode:raise SystemExit(r.stderr)
r=run(['git','push','origin','HEAD:refs/heads/codex/force-critic-act-v2'])
print(r.stdout+r.stderr)
if r.returncode:
    bundle=Path('.cache/force-critic-act-v2.bundle');bundle.parent.mkdir(exist_ok=True)
    run(['git','bundle','create',str(bundle),'codex/force-critic-act-v2'])
    print('BLOCKED_AUTH for shell transport; local commit and .cache bundle retained. Connector may still publish.')
raise SystemExit(r.returncode)
