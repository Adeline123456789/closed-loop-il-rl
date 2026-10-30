"""Recover charged native steps; never silently turn a missing finally into zero."""
import json
import math
from pathlib import Path


def recover_accounting(progress):
    counts = progress.get('accounting',{}).get('physics_substeps',{})
    trajectory = progress.get('last_trajectory')
    journal = (Path(progress['journal_path']) if progress.get('journal_path') else
               Path(trajectory).parent/'physics_ledger.jsonl' if trajectory else None)
    last = None
    if journal and journal.exists():
        with journal.open('rb') as stream:
            stream.seek(max(0,journal.stat().st_size-8192))
            for line in stream.read().splitlines():
                try: value=json.loads(line)
                except ValueError: continue
                if value.get('event') in ('before','after'):last=value
    if last:counts=last['counts']
    decimation=progress.get('accounting',{}).get('decimation',8)
    return dict(status='CHARGED_STEPS_RECOVERED' if last else 'LOWER_BOUND_ONLY',
        physics_substeps=counts,
        individual_control_step_equivalents={k:math.ceil(v/decimation) for k,v in counts.items()},
        pending_physics_substeps=(progress['accounting']['num_envs'] if last['event']=='before' else 0) if last else None,
        unresolved_unflushed_steps=not bool(last),
        note='Before-step journal charges a possibly incomplete final substep. Without that journal, unflushed work remains unknown; zero is not inferred.')
