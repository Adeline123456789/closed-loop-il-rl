"""Audited geometric-only Factory reward adapter, without success-prediction/contact penalties.

Caller provides geometry reward terms from pinned Factory `_get_factory_rew_dict`.
This does not claim a measured safety calibration or reward-integrated robot run.
"""
import numpy as np


class GeometricReward:
    TERMS=('kp_baseline','kp_coarse','kp_fine','curr_engaged')
    def __init__(self,num_envs):self.success_paid=np.zeros(num_envs,dtype=bool)
    def reset(self,ids):self.success_paid[ids]=False
    def compute(self,terms,stable_success):
        if any(k not in terms for k in self.TERMS):raise ValueError('missing official geometric terms')
        success=np.asarray(stable_success,dtype=bool)
        bonus=success & ~self.success_paid
        self.success_paid |= success
        # Each geometric term has coefficient 1 in the pinned Factory source.
        return sum(np.asarray(terms[k]) for k in self.TERMS)+bonus.astype(float)
