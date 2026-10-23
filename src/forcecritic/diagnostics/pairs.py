"""Finite paired contrasts, not claims of contact derivatives."""
import numpy as np


def candidates(a0, direction, epsilon):
    a0, d = np.asarray(a0), np.asarray(direction)
    norm = np.linalg.norm(d)
    if norm == 0 or epsilon <= 0:
        raise ValueError('nonzero direction and positive amplitude required')
    d = d/norm
    plus, minus = np.clip(a0 + epsilon*d, -1, 1), np.clip(a0 - epsilon*d, -1, 1)
    return plus, minus, bool(np.allclose(plus-a0, a0-minus))


def paired_effect(plus, minus, base, noise_ids, seed=0):
    p, m, b = map(np.asarray, (plus, minus, base))
    if p.shape != m.shape or p.shape != b.shape or len(p) != len(noise_ids):
        raise ValueError('aligned paired full-horizon returns required')
    delta = p-m
    out = {'delta': delta.mean(0).tolist(), 'i_plus': (p-b).mean(0).tolist(),
           'i_minus': (m-b).mean(0).tolist(), 'ci': None, 'independent_noise_blocks': len(set(noise_ids))}
    if len(set(noise_ids)) < 3:
        out['status'] = 'INCONCLUSIVE'
        return out
    # Bootstrap independent noise blocks; deterministic duplicates do not increase n.
    blocks = np.array([delta[np.array(noise_ids)==k].mean(0) for k in sorted(set(noise_ids))])
    rng = np.random.default_rng(seed)
    means = blocks[rng.integers(len(blocks), size=(2000,len(blocks)))].mean(1)
    out['ci'] = np.quantile(means, [.025,.975], axis=0).tolist()
    out['status'] = 'SCREENING_ONLY'
    return out


def fidelity(original, repeated, tolerance):
    """Equivalence to O, not merely R/R repeatability; check all locked dimensions."""
    if set(original) != set(tolerance) or not repeated:
        raise ValueError('all physical and return/cost tolerances must be locked')
    errors = {key: max(float(np.max(np.abs(np.asarray(r[key])-np.asarray(original[key]))))
                       for r in repeated) for key in original}
    return {'passed': all(np.isfinite(errors[k]) and errors[k] <= tolerance[k] for k in errors), 'errors': errors}


def training_whitelist(rows, continuation_sha):
    if any(r['split'] != 'train' or r['continuation_sha'] != continuation_sha for r in rows):
        raise ValueError('non-training or stale continuation labels')
    return sorted({r['checkpoint_id'] for r in rows if r['replay_passed'] and r['independent_confirmation']
                   and r['improvement_lower'] > 0 and not r['cost_worse']})
