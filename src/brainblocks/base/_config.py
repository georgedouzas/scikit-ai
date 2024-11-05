"""Configuration of parameters."""

# Author: Georgios Douzas <gdouzas@icloud.com>

import torch
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.utils.logger.base import LazyLogger

from ._models import DQNCartPoleModel, DQNModel

DEFAULT_PARAMETERS = {
    'DQNAgent': {
        'model': lambda agent: DQNModel(agent.n_observations_, agent.n_actions_),
        'optim': lambda agent: torch.optim.Adam(agent.model_.parameters()),
        'discount_factor': 0.99,
        'estimation_step': 1,
        'target_update_freq': 10,
        'reward_normalization': False,
        'is_double': False,
        'clip_loss_grad': False,
        'lr_scheduler': None,
        'train_collector': lambda agent: Collector(policy=agent.policy_, env=agent.envs_),
        'test_collector': lambda agent: Collector(policy=agent.policy_, env=agent.envs_),
        'max_epoch': 10,
        'batch_size': None,
        'buffer': None,
        'step_per_epoch': 100,
        'repeat_per_collect': None,
        'episode_per_test': 10,
        'update_per_step': 1.0,
        'step_per_collect': 10,
        'episode_per_collect': None,
        'train_fn': None,
        'test_fn': None,
        'stop_fn': None,
        'save_best_fn': None,
        'save_checkpoint_fn': None,
        'resume_from_log': False,
        'reward_metric': None,
        'logger': LazyLogger(),
        'verbose': True,
        'show_progress': True,
        'test_in_train': True,
    },
}
PARAMETERS = {
    'DQNAgent': {
        'CartPole': {
            'model': lambda agent: DQNCartPoleModel(n_observations=agent.n_observations_, n_actions=agent.n_actions_),
            'optim': lambda agent: torch.optim.Adam(params=agent.model_.parameters(), lr=1e-3),
            'discount_factor': 0.9,
            'estimation_step': 3,
            'target_update_freq': 320,
            'train_collector': lambda agent: Collector(
                policy=agent.policy_,
                env=agent.envs_,
                buffer=VectorReplayBuffer(20000, 10),
                exploration_noise=True,
            ),
            'test_collector': lambda agent: Collector(
                policy=agent.policy_,
                env=agent.test_envs_,
                exploration_noise=True,
            ),
            'max_epoch': 10,
            'step_per_epoch': 10000,
            'step_per_collect': 10,
            'update_per_step': 0.1,
            'episode_per_test': 100,
            'batch_size': 64,
            'train_fn': lambda agent: (lambda epoch, env_step: agent.policy_.set_eps(0.1)),
            'test_fn': lambda agent: (lambda epoch, env_step: agent.policy_.set_eps(0.05)),
            'stop_fn': lambda agent: (lambda mean_rewards: mean_rewards >= agent.env_.spec.reward_threshold),
        },
    },
}
