"""Canonical episode table for one explicitly selected real evaluation cohort."""
import csv
from .report import SCHEMAS
from ..runtime import atomic_json,digest


def export_episodes(run,outcomes,cohort,raw_paths):
    rows=[]
    for arm,episodes in outcomes.items():
        for r in episodes:
            if r['source']!='real_forge':raise ValueError('Robot summary requires actual simulator episodes')
            rows.append(dict(parent_episode=r['parent_id'],split=r['split'],policy_sha=r['policy_sha'],
                arm=arm,seed=run.config['seed'],stable_success=r['stable_success'],ever_success=r['ever_success'],
                final_success=r['final_success'],completion_time=r['steps']*r['control_dt'] if r['stable_success'] else '',
                force_peak=r['peak_force_N'],torque_peak=r['peak_torque_Nm'],overload=int(r['overload_steps']>0),
                termination_reason=r['termination_reason'],env_steps=r['steps'],status='COMPLETED'))
    path=run.result/'episode_metrics.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=SCHEMAS[path.name].split(','));writer.writeheader();writer.writerows(rows)
    atomic_json(run.report/'episode_cohort.json',dict(cohort=cohort,training_seed=run.config['seed'],
        reset_conditions='reset seed encoded in parent_episode; not independent training seeds',
        control_dt=sorted({r['control_dt'] for episodes in outcomes.values() for r in episodes}),
        raw_evidence_hashes={str(p):digest(p) for p in raw_paths},
        note='One cohort only. Earlier raw CSVs remain separate; no pooling baseline and short-update parents.'))
