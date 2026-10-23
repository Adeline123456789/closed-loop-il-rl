"""Block replay with explicit terminal and behavior/continuation version semantics."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Transition:
    parent_id: str
    split: str
    behavior_sha: str
    observation: dict
    plan: np.ndarray
    valid: np.ndarray
    returns: np.ndarray
    next_observation: dict
    bootstrap: bool
    gamma_k: float
    source: str = 'real_forge'

    def validate(self,gamma=.99):
        if self.source!='real_forge':raise ValueError('HF/fixtures cannot masquerade as online TD replay')
        if self.plan.shape!=(4,4) or self.valid.shape!=(4,):raise ValueError('invalid block')
        k=int(self.valid.sum())
        if not 1<=k<=4 or not np.array_equal(self.valid,np.arange(4)<k):raise ValueError('invalid prefix')
        if not np.isclose(self.gamma_k,gamma**k):raise ValueError('invalid discount')
        if not self.parent_id or not self.behavior_sha or self.next_observation is None:raise ValueError('missing provenance/final observation')


class Replay:
    def __init__(self,capacity,seed=0):
        self.capacity=capacity;self.rows=[];self.rng=np.random.default_rng(seed)
    def append(self,transition):
        transition.validate()
        if transition.split!='train':raise ValueError('only training parents enter training replay')
        if len(self.rows)==self.capacity:self.rows.pop(0)
        self.rows.append(transition)
    def sample(self,batch_size):
        if not self.rows:raise ValueError('empty replay')
        return [self.rows[i] for i in self.rng.integers(len(self.rows),size=batch_size)]


class ReturnScale:
    def fit(self,returns,splits):
        if any(s!='train' for s in splits):raise ValueError('normalization from train only')
        values=np.asarray(returns)
        if values.ndim!=2 or values.shape[1]!=2 or not np.isfinite(values).all():raise ValueError('invalid returns')
        self.scale=np.maximum(values.std(0),1e-6)
        if np.ptp(values[:,0])==0:raise ValueError('NO_REWARD_VARIATION: diagnose before calibration')
        return self
    def transform(self,value):return np.asarray(value)/self.scale
