"""Export only verified timestamp/counter journals, never raw sensor datasets."""
import gzip
import json
from pathlib import Path
from forcecritic.runtime import digest,atomic_json
from forcecritic.staged.audit import validate_physics_journal

report=Path('reports/stage_v3_20261026');archives={}
for name in ('eval_A1_initial','eval_A1_extension','eval_A2_extension'):
    path=report/(name+'.supervisor.json')
    if not path.exists():continue
    supervisor=json.loads(path.read_text())
    if supervisor['returncode']!=0:continue
    progress=json.loads((report/name/'diagnostic_live_progress.json').read_text())
    journal=Path(progress['journal_path'])
    counts=supervisor['accounting']['physics_substeps']
    verified=validate_physics_journal(journal,counts)
    destination=report/(name+'.physics.jsonl.gz')
    compressed=gzip.compress(journal.read_bytes(),mtime=0)
    if destination.exists():
        if gzip.decompress(destination.read_bytes())!=journal.read_bytes():raise ValueError('Existing published journal changed')
    else:destination.write_bytes(compressed)
    archives[name]=dict(status='PASSED',**verified,source=str(journal),source_sha=digest(journal),
        archive=str(destination),archive_sha=digest(destination),archive_bytes=destination.stat().st_size,
        schema='Only event, timestamp and five phase counters; no raw sensor data or images')
atomic_json(report/'physics_ledger_archives.json',archives)
print('Verified and exported',len(archives),'completed physics journals')
