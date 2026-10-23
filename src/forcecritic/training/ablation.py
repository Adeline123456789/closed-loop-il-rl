"""Same-data short update engine. Inputs must come from accepted real rollout bundles.

This engine performs optimizer work only; robot outcomes must be measured by the
separate frozen-policy evaluator, never inferred from these losses.
"""
import copy
import random
import numpy as np
import torch
from .updates import actor_step,critic_step,clone_arms,td_target,polyak


def short_updates(actor,critic,actor_optimizer,critic_optimizer,*,td_batches,endpoint_batches,
                  pair_batches,actor_batches,whitelist,data_hash,policy_sha,
                  critic_updates=2000,actor_updates=200,alpha_bc=1.,eta_step=1.,target_tau=.005):
    if not 0<=critic_updates<=2000 or not 0<=actor_updates<=200:
        raise ValueError('short update protocol budget exceeded')
    if not 0<target_tau<=1:raise ValueError('invalid target tau')
    if critic_updates and not all((td_batches,endpoint_batches,pair_batches)):
        raise ValueError('shared TD, endpoint and pair batches required')
    if actor_updates and not actor_batches:raise ValueError('actor batches required')
    states=clone_arms(actor,critic,actor_optimizer,critic_optimizer,data_hash,whitelist)
    results={}
    for arm,start in states.items():
        actor.load_state_dict(start['actor']);critic.load_state_dict(start['critic'])
        actor_optimizer.load_state_dict(start['actor_optimizer']);critic_optimizer.load_state_dict(start['critic_optimizer'])
        torch.set_rng_state(start['rng'])
        if start['cuda_rng']:torch.cuda.set_rng_state_all(start['cuda_rng'])
        np.random.set_state(start['numpy_rng']);random.setstate(start['python_rng'])
        # The continuation actor stays frozen through this critic-fitting phase.
        # Every arm starts with identical target copies and independently updates them.
        target_actor=copy.deepcopy(actor).eval().requires_grad_(False)
        target_critic=copy.deepcopy(critic).eval().requires_grad_(False)
        losses=[]
        for i in range(critic_updates):
            td=td_batches[i%len(td_batches)]
            if any(split!='train' for split in td['splits']):
                raise ValueError('non-training TD sample')
            # Recompute from the current target network; never freeze cached TD labels.
            target=td_target(target_actor,target_critic,td['next_obs'],td['next_condition'],
                             td['returns'],td['gamma_k'],td['bootstrap'])
            losses.append(critic_step(critic,critic_optimizer,td['condition'],td['action'],target,
                endpoints=endpoint_batches[i%len(endpoint_batches)],pairs=pair_batches[i%len(pair_batches)],
                lambda_pair=1. if arm=='B3' else 0.,policy_sha=policy_sha))
            polyak(critic,target_critic,target_tau)
        reference=copy.deepcopy(actor).eval().requires_grad_(False)
        accepted={r['checkpoint_id'] for r in whitelist}
        actor_logs=[]
        for i in range(actor_updates):
            batch=actor_batches[i%len(actor_batches)]
            if any(s!='train' for s in batch['splits']):raise ValueError('actor label leakage')
            mask=torch.tensor([arm=='B1' or k in accepted for k in batch['checkpoint_ids']],
                              device=batch['condition'].device,dtype=torch.bool)
            with torch.no_grad():old=reference(batch['obs'])
            actor_logs.append(actor_step(actor,critic,actor_optimizer,batch['obs'],batch['condition'],mask,old,
                demo=batch['demo'],alpha_bc=alpha_bc,eta_step=eta_step))
        results[arm]={'actor':copy.deepcopy(actor.state_dict()),'critic':copy.deepcopy(critic.state_dict()),
                      'target_critic':copy.deepcopy(target_critic.state_dict()),
                      'critic_losses':losses,'actor_logs':actor_logs,'data_hash':data_hash,
                      'whitelist':copy.deepcopy(whitelist),'policy_sha':policy_sha}
    return results
