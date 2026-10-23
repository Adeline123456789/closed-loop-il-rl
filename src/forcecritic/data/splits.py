"""Episode/run grouping and train-only statistics."""
import hashlib
import numpy as np


def group_splits(episodes, seed=0):
    # A collection run can span success/failure repositories: do not include outcome in group key.
    groups = {}
    for row in episodes:
        key = row.get('collection_run_id') or row['parent_id']
        groups.setdefault(key, []).append(row['parent_id'])
    keys = sorted(groups, key=lambda x: hashlib.sha256(f'{seed}:{x}'.encode()).hexdigest())
    n = len(keys)
    out = {}
    for i, key in enumerate(keys):
        split = 'train' if i < int(.7*n) else 'validation' if i < int(.85*n) else 'test'
        out.update({parent: split for parent in groups[key]})
    return out


def inherit_split(parent_id, mapping):
    return mapping[parent_id]


def train_statistics(rows):
    values = [np.asarray(r['values']) for r in rows if r['split'] == 'train']
    if not values:
        raise ValueError('no training observations')
    x = np.concatenate(values, 0)
    return {'mean': x.mean(0), 'std': x.std(0).clip(1e-6)}
