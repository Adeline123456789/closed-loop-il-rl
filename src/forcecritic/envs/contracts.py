"""Pure controller contracts; unit tests are not simulator measurements."""
from dataclasses import dataclass, field
import numpy as np


def native_to_stock(action):
    a = np.asarray(action)
    if a.shape[-1] != 4 or not np.isfinite(a).all():
        raise ValueError('expected finite x/y/z/yaw normalized command')
    out = np.zeros((*a.shape[:-1], 7), dtype=a.dtype)
    out[..., [0, 1, 2, 5]] = np.clip(a, -1, 1)
    out[..., 6] = -1  # placeholder, never a gripper or a success criterion
    return out  # EMA belongs solely to stock environment


def stock_to_native(action, *, ema_stage):
    if ema_stage != 'pre':
        raise ValueError('HF action EMA stage unverified or post-EMA; cannot use as command label')
    a = np.asarray(action)
    if a.shape[-1] != 7:
        raise ValueError('expected stock seven-dimensional action')
    return a[..., [0, 1, 2, 5]].copy()


def transform_wrench(wrench, rotation_new_from_old, old_origin_in_new):
    """r is the vector NEW origin -> OLD origin, expressed in NEW coordinates."""
    w = np.asarray(wrench)
    if w.shape[-1] != 6:
        raise ValueError('six-dimensional wrench required; never pad ft_noisy')
    f = np.einsum('ij,...j->...i', rotation_new_from_old, w[..., :3])
    t = np.einsum('ij,...j->...i', rotation_new_from_old, w[..., 3:])
    return np.concatenate((f, t + np.cross(old_origin_in_new, f)), -1)


def contact_cost(wrench_physical, limits, unsafe=False, unsafe_penalty=1.):
    w, lim = np.asarray(wrench_physical), np.asarray(limits)
    if w.shape[-1] != 6 or lim.shape != (6,) or not np.isfinite(lim).all() or (lim <= 0).any():
        raise ValueError('locked six physical N/Nm limits required')
    return (
        np.square(np.maximum(np.abs(w) / lim - 1, 0)).sum(-1) + unsafe_penalty * np.asarray(unsafe))


@dataclass
class StepLedger:
    cap: int
    counts: dict = field(default_factory=lambda: dict(interaction=0, branch=0, prefix=0, restore=0, evaluation=0))

    @property
    def total(self):
        return sum(self.counts.values())

    def charge(self, category, num_envs=1, control_steps=1):
        n = num_envs * control_steps
        if category not in self.counts or num_envs < 1 or control_steps < 0:
            raise ValueError('invalid step accounting')
        if self.total + n > self.cap:
            raise RuntimeError('BUDGET_EXHAUSTED before simulator step')
        self.counts[category] += n


def execute_chunk(plan, step, ledger, category='interaction', gamma=.99):
    """Single-env diagnostic executor. step returns explicit pre-reset final observation.

    timeout is a logging truncation (bootstrap); budget exhaustion is a true finite
    horizon termination. The simulator adapter must distinguish these explicitly.
    """
    plan = np.asarray(plan).copy()
    if plan.shape != (4, 4):
        raise ValueError('K=M=4')
    returns = np.zeros(2)
    valid = np.zeros(4, dtype=bool)
    records = []
    for i, command in enumerate(plan):
        ledger.charge(category)
        row = step(native_to_stock(command))
        records.append(row)
        valid[i] = True
        returns += gamma**i * np.asarray([row['reward'], row['cost']])
        ended = any(row.get(k, False) for k in ('terminated','truncated','success','unsafe','budget_exhausted'))
        if ended:
            if 'final_observation' not in row:
                raise ValueError('terminal pre-reset observation missing')
            break
    terminal = any(row.get(k, False) for k in ('terminated','success','unsafe','budget_exhausted'))
    return dict(plan=plan, valid=valid, k=int(valid.sum()), gamma_k=gamma**valid.sum(),
                returns=returns, bootstrap=not terminal,
                next_observation=row['final_observation'] if ended else row['observation'], records=records)
