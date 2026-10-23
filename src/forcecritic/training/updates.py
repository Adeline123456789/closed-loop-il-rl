"""Direct ACT parameter updates. No simulator derivatives or residual policy."""
from contextlib import contextmanager
import copy
import random
import numpy as np
import torch
from torch.nn import functional as F


@contextmanager
def frozen(module):
    flags = [p.requires_grad for p in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for p, flag in zip(module.parameters(), flags):
            p.requires_grad_(flag)


def masked_mean(values, mask):
    mask = mask.to(values.dtype)
    return (values * mask).sum() / mask.sum().clamp_min(1)


def actor_step(actor, critic, optimizer, obs, condition, mask, old_action,
               demo=None, alpha_bc=1., eta_step=1., lambda_cost=1.):
    optimizer.zero_grad(set_to_none=True)
    with frozen(critic):
        action = actor(obs)
        # Intentionally differentiable wrt action; condition cannot be altered by ACT.
        q = critic(condition.detach(), action)
        q_loss = -masked_mean(q[:, :, 0].min(1).values - lambda_cost * q[:, :, 1].max(1).values, mask)
        params = [p for p in actor.parameters() if p.requires_grad]
        q_grads = torch.autograd.grad(q_loss, params, retain_graph=True, allow_unused=True)
        bc = action.sum() * 0 if demo is None else actor.bc_loss(*demo)
        bc_grads = torch.autograd.grad(bc, params, retain_graph=True, allow_unused=True)
        norm = lambda gs: sum(g.detach().square().sum() for g in gs if g is not None).sqrt().item()
        dot = sum((g.detach()*b.detach()).sum() for g,b in zip(q_grads,bc_grads) if g is not None and b is not None)
        loss = q_loss + alpha_bc * bc + eta_step * F.mse_loss(action, old_action.detach())
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        after = actor(obs)
    return {'loss': float(loss.detach()), 'q_grad_norm': norm(q_grads),
            'bc_grad_norm': norm(bc_grads), 'q_bc_dot': float(dot),
            'before': action.detach(), 'after': after, 'delta_action': after-action.detach(),
            'saturation': float((after.abs() > .99).float().mean())}


def td_target(target_actor, target_critic, next_obs, next_condition, returns, gamma_k, bootstrap):
    with torch.no_grad():
        q = target_critic(next_condition, target_actor(next_obs)).mean(1)
        return returns + gamma_k[:, None] * bootstrap[:, None] * q


def critic_step(critic, optimizer, condition, action, target, endpoints=None, pairs=None,
                lambda_pair=0., policy_sha=None):
    optimizer.zero_grad(set_to_none=True)
    pred = critic(condition.detach(), action.detach())
    loss = F.smooth_l1_loss(pred, target.detach()[:, None].expand_as(pred))
    if endpoints is not None:
        if endpoints['policy_sha'] != policy_sha:
            raise ValueError('stale MC continuation policy')
        point = critic(endpoints['condition'], endpoints['action'].detach())
        loss = loss + F.smooth_l1_loss(point, endpoints['returns'].detach()[:, None].expand_as(point))
    if pairs is not None:
        if pairs['policy_sha'] != policy_sha or any(s != 'train' for s in pairs['splits']):
            raise ValueError('stale or non-training pair labels')
        delta = critic(pairs['condition'], pairs['plus'].detach()) - critic(pairs['condition'], pairs['minus'].detach())
        error = F.smooth_l1_loss(delta, pairs['delta'].detach()[:, None].expand_as(delta), reduction='none').mean((1, 2))
        loss = loss + lambda_pair * masked_mean(error, pairs['accepted'])
    loss.backward()
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def polyak(source, target, tau=.005):
    for a, b in zip(source.parameters(), target.parameters()):
        b.lerp_(a, tau)
    for a, b in zip(source.buffers(), target.buffers()):
        b.copy_(a)


def clone_arms(actor, critic, actor_optimizer, critic_optimizer, data_hash, whitelist):
    """Identical model/optimizer/RNG starts. Caller restores each arm's RNG before training."""
    if any(row['split'] != 'train' for row in whitelist):
        raise ValueError('test/validation label leakage')
    base = {'actor': actor.state_dict(), 'critic': critic.state_dict(),
            'actor_optimizer': actor_optimizer.state_dict(), 'critic_optimizer': critic_optimizer.state_dict(),
            'rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
            'data_hash': data_hash, 'whitelist': whitelist}
    return {arm: copy.deepcopy(base) for arm in ('B1', 'B2', 'B3')}
