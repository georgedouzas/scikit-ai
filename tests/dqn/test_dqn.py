"""Test the base agent class."""

import gymnasium as gym
import pytest
import tianshou as ts

from skai.dqn import DQNAgent

ENV = gym.wrappers.FlattenObservation(gym.make('Blackjack-v1'))


@pytest.mark.parametrize(
    'env', [ENV, gym.make_vec(ENV.spec, num_envs=2), ts.env.ShmemVectorEnv([lambda: ENV for _ in range(2)])]
)
@pytest.mark.parametrize('logging_terminal', [True, False])
@pytest.mark.parametrize('logging_tensorboard', [True, False])
def test_learn(tmp_path, logging_terminal, logging_tensorboard, env):
    """Test the learn method."""
    if logging_tensorboard:
        logging_tensorboard = {'logging_path': tmp_path}
    agent = DQNAgent()
    agent.learn(env, logging_terminal=logging_terminal, logging_tensorboard=logging_tensorboard)
