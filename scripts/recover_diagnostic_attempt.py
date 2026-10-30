"""Preserve and recheck interrupted D0 evidence before an implementation repair retry."""
import json
from pathlib import Path
import shutil
import torch
from forcecritic.runtime import Run, digest
from forcecritic.diagnostics.artifacts import atomic_json
from forcecritic.diagnostics.live import compare

run=Run('configs/minimal_validation.yaml','20261023T104759Z')
dest=run.report/'diagnostic_repair.json'
if dest.exists():raise SystemExit('Recovery already recorded; do not repeat accounting')
directory=Path('checkpoints')/run.run_id/'fidelity_initial'
thresholds=json.loads((run.report/'protocol_effect_thresholds.json').read_text())['fidelity_tolerances']
scale=json.loads((run.report/'baseline_training.json').read_text())['return_scale']
partial=json.loads((directory/'fidelity.json').read_text())
known={c['parent_id']:c['hard'] for c in partial['checkpoints']}
points=[]
for path in sorted(directory.glob('diagnostic-*.pt')):
    value=torch.load(path,weights_only=False,map_location='cpu')
    for row in value['records']:
        check=compare(row['original'],row['repeated'],thresholds,scale)
        if check['passed']!=row['passed'] or check['errors']!=row['errors']:
            raise ValueError('Saved physical comparisons do not reproduce')
    points.append(dict(parent_id=path.stem,hard=known.get(path.stem),
        accepted_methods=value['accepted_methods'],raw_sha256=digest(path),
        comparisons=[{k:v for k,v in r.items() if k not in ('original','repeated')} for r in value['records']]))
archive=directory.with_name('fidelity_initial_before_repair')
if archive.exists():raise FileExistsError(archive)
directory.rename(archive)
csv_path=run.result/'fidelity.csv'
archived_csv=run.result/'fidelity_before_repair.csv'
shutil.copyfile(csv_path,archived_csv)
csv_path.write_text(csv_path.read_text().splitlines()[0]+'\n')
atomic_json(run.report/'fidelity_before_repair.json',dict(checkpoints=points,
    completed_points=len(points),continuations_used=21*len(points),comparisons=18*len(points),
    note='Recomputed from saved real trajectories. Missing second-point hard classification remains null; no replacement data inferred.'))
probes=[]
steps=0
short_branches=0
for name in ('original','targets_fixed'):
    path=run.report/f'replay_probe_{name}.json'
    value=json.loads(path.read_text())
    if value['status']!='MEASURED':raise ValueError('Engineering probe did not finish')
    counts=value['accounting']['individual_control_step_equivalents']
    steps+=sum(counts.values())
    short_branches+=sum(r['kind']=='restore_one_step' for r in value['comparisons'])*2
    for k,v in counts.items():run.manifest['steps'][k]+=v
    probes.append(dict(path=str(path),sha256=digest(path),accounting=value['accounting']))
atomic_json(dest,dict(status='REPAIR_READY',previous_full_continuations=21*len(points),
    previous_engineering_one_step_branches=short_branches,
    previous_continuations=21*len(points)+short_branches,engineering_control_steps=steps,
    recovered_summary=str(run.report/'fidelity_before_repair.json'),archived_csv=str(archived_csv),
    original_failure='diagnostic_1793001458468229690_outcome.json',
    engineering_probes=probes,baseline_reused=True,thresholds_unchanged=True,
    reason='NumPy JSON serialization, actuator target corruption by reset_to, and interval clock not reset. Original failed evidence retained; retry tests corrected implementation, not relaxed thresholds.'))
run.manifest.setdefault('diagnostic_engineering_repairs',[]).append(str(dest))
run.save()
print(json.dumps(dict(recovered_points=len(points),engineering_steps=steps,baseline_verified=run.verified('baseline'))))
