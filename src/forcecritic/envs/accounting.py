"""Count every physics advance, including scene initialization and auto resets."""
from contextlib import contextmanager
import math


class PhysicsLedger:
    def __init__(self, num_envs, decimation, control_step_cap):
        from isaaclab.sim import SimulationContext
        self.owner = SimulationContext
        self.original = SimulationContext.step
        self.num_envs = num_envs
        self.decimation = decimation
        self.cap = control_step_cap * decimation
        self.phase = 'restore'
        self.counts = {k: 0 for k in ('restore', 'interaction', 'branch', 'prefix', 'evaluation')}
        def counted(sim, *args, **kwargs):
            if sum(self.counts.values()) + num_envs > self.cap:
                raise RuntimeError('BUDGET_EXHAUSTED before physics step')
            self.counts[self.phase] += num_envs
            return self.original(sim, *args, **kwargs)
        SimulationContext.step = counted

    @contextmanager
    def using(self, phase):
        previous = self.phase
        self.phase = phase
        try:
            yield
        finally:
            self.phase = previous

    def record(self):
        return {'physics_substeps': dict(self.counts), 'decimation': self.decimation,
                'num_envs':self.num_envs,'control_step_cap':self.cap/self.decimation,
                'individual_control_step_equivalents': {k: math.ceil(v/self.decimation) for k,v in self.counts.items()},
                'charged_before_step': True}

    def close(self):
        self.owner.step = self.original
