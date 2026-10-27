"""Pinned rl-games expert player, including observation normalization and LSTM.

This is used for P0 acceptance, collection, and reconstruction of its RNG calls. It is not
an actor residual, a teacher in the critic loop, or evidence of published scores.
"""
import math
from pathlib import Path
import torch
from ..runtime import digest

EXPERT_SHA256 = '10efdf802e8abfccaa13f6e7a18cc2887ca2f3f96b35526bc9984fde419e59f8'


class Expert:
    def __init__(self, env, checkpoint, seed=0):
        if digest(checkpoint) != EXPERT_SHA256:
            raise ValueError('expert checkpoint does not match the audited publisher artifact')
        from isaaclab_tasks.utils import load_cfg_from_registry
        from isaaclab_rl.rl_games import RlGamesVecEnvWrapper, RlGamesGpuEnv
        from rl_games.common import env_configurations, vecenv
        from rl_games.torch_runner import Runner
        from .forge import TASK
        cfg = load_cfg_from_registry(TASK, 'rl_games_cfg_entry_point')
        cfg['params']['seed'] = seed
        cfg['params']['load_checkpoint'] = True
        cfg['params']['load_path'] = str(Path(checkpoint).resolve())
        cfg['params']['config']['num_actors'] = env.unwrapped.num_envs
        cfg['params']['config']['device'] = str(env.unwrapped.device)
        self.clip_obs = cfg['params']['env'].get('clip_observations', math.inf)
        wrapper = RlGamesVecEnvWrapper(env, str(env.unwrapped.device), self.clip_obs, 1.)
        vecenv.register('IsaacRlgWrapper', lambda config_name, num_actors, **kw:
                        RlGamesGpuEnv(config_name, num_actors, **kw))
        env_configurations.register('rlgpu', {'vecenv_type': 'IsaacRlgWrapper', 'env_creator': lambda **kw: wrapper})
        runner = Runner()
        runner.load(cfg)
        self.player = runner.create_player()
        self.player.restore(str(checkpoint))
        self.player.reset()
        self.started = False

    def action(self, observation):
        obs = observation['policy'].clamp(-self.clip_obs, self.clip_obs)
        if not self.started:
            self.player.get_batch_size(obs, 1)
            if self.player.is_rnn:
                self.player.init_rnn()
            self.started = True
        # rl-games mutates recurrent states when environments finish. Keep
        # ordinary tensors so reset_done can update them outside this call.
        # Pinned rl-games samples internally even when get_action returns the
        # deterministic mean. Historical FORGE collection shares that CUDA RNG:
        # replay must preserve this call before submitting the recorded action.
        with torch.no_grad():
            return self.player.get_action(self.player.obs_to_torch(obs),
                is_deterministic=self.player.is_deterministic).clamp(-1, 1)

    def reset_done(self, done):
        if self.player.is_rnn and self.player.states is not None:
            for state in self.player.states:
                state[:, done, :] = 0.
