"""Atomic restart state, including optimizer and RNG; not just model weights."""
import os
from pathlib import Path
import random
import numpy as np
import torch


def save_checkpoint(path, actor, critic, actor_optimizer, critic_optimizer, *, update, config_hash, policy_sha, ledger):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    state={'actor':actor.state_dict(),'critic':critic.state_dict(),
           'actor_optimizer':actor_optimizer.state_dict(),'critic_optimizer':critic_optimizer.state_dict(),
           'update':update,'config_hash':config_hash,'policy_sha':policy_sha,'ledger':ledger,
           'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
           'numpy_rng':np.random.get_state(),'python_rng':random.getstate()}
    temp=path.with_suffix('.tmp');torch.save(state,temp);os.replace(temp,path)


def restore_checkpoint(path, actor, critic, actor_optimizer, critic_optimizer, config_hash):
    # Only locally generated trusted checkpoints; never arbitrary remote pickle.
    state=torch.load(path,map_location='cpu',weights_only=False)
    if state['config_hash']!=config_hash:raise ValueError('checkpoint configuration mismatch')
    actor.load_state_dict(state['actor']);critic.load_state_dict(state['critic'])
    actor_optimizer.load_state_dict(state['actor_optimizer']);critic_optimizer.load_state_dict(state['critic_optimizer'])
    torch.set_rng_state(state['torch_rng'])
    if state['cuda_rng']:torch.cuda.set_rng_state_all(state['cuda_rng'])
    np.random.set_state(state['numpy_rng']);random.setstate(state['python_rng'])
    return state
