"""Save distributions and provenance from the actual interpreter; never dump environment variables."""
from pathlib import Path
import importlib.metadata as md
import json
import platform
import subprocess
import sys
sys.path.insert(0,'src')
from forcecritic.runtime import atomic_json
run_id=json.loads(Path('reports/PROJECT_STATUS.json').read_text())['run_id']
p=Path('reports')/run_id
lock={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
      'packages':sorted([{'name':d.metadata['Name'],'version':d.version} for d in md.distributions()],key=lambda d:d['name'].lower())}
atomic_json(p/'cpu_environment.lock.json',lock)
