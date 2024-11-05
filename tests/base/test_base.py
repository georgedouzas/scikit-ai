"""Test the base agent class."""

from typing import Any, ClassVar, Self

import gymnasium as gym
import numpy as np
import pytest
from tianshou.data.batch import Batch, BatchProtocol
from tianshou.data.types import ActBatchProtocol, ActStateBatchProtocol, ObsBatchProtocol, RolloutBatchProtocol
from tianshou.env import BaseVectorEnv
from tianshou.policy import BasePolicy, TrainingStats

from brainblocks.base import BaseAgent


class Env(gym.Env):
    """Test environment."""

    def __init__(self: Self) -> None:
        """Initialize the test environment."""
        self.observation_space = gym.spaces.Discrete(3)
        self.action_space = gym.spaces.Discrete(2)

    def reset(self: Self, seed: int | None = None, options: dict | None = None):
        """Reset the test environment."""
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self: Self, action: int):
        """Apply an action to the test environment."""
        observation = self.observation_space.sample()
        reward = int(observation == action)
        terminated = observation == action
        return observation, reward, terminated, False, {}

    def render(self):
        """Resets the environment to an initial internal state."""
        observation = self.observation_space.sample()
        return observation, {}


class Policy(BasePolicy):
    """Test policy."""

    def forward(
        self,
        batch: ObsBatchProtocol,
        state: dict | BatchProtocol | np.ndarray | None = None,
        **kwargs: Any,
    ) -> ActBatchProtocol | ActStateBatchProtocol:
        """Compute action over the given batch data."""
        return Batch(act=[0] * len(batch))

    def learn(self, batch: RolloutBatchProtocol, *args: Any, **kwargs: Any) -> TrainingStats:
        """Update policy with a given batch of data."""
        return TrainingStats()


class Agent(BaseAgent):
    """Test agent."""

    _non_numeric_params: ClassVar[dict[str, tuple[Any, ...]]] = {
        'param_str': (str,),
        'param_bool': (bool,),
    }
    _numeric_params: ClassVar[dict[str, dict]] = {
        'param_int': {'target_type': int, 'min_val': 1},
        'param_float': {'target_type': float, 'min_val': 1.0, 'max_val': 6.0},
    }

    def __init__(
        self,
        param_str: str | None = None,
        param_int: int | None = None,
        param_bool: bool = False,
        param_float: float = 1.0,
    ) -> None:
        """Initialize the agent."""
        self.param_str = param_str
        self.param_int = param_int
        self.param_bool = param_bool
        self.param_float = param_float

    def _learn(self: Self, env: Env | BaseVectorEnv, test_env: Env | BaseVectorEnv | None) -> Self:
        super()._learn(env, test_env)
        self.policy_ = Policy(action_space=self.action_space_)
        return self


@pytest.mark.parametrize(
    'params',
    [
        {},
        {'param_int': 9, 'param_float': 5.0},
        {'param_str': 'test', 'param_int': 3, 'param_bool': True, 'param_float': 4.0},
    ],
)
def test_init(params):
    """Test the initialization of agent with parameters."""
    agent = Agent(**params)
    assert agent.param_str is None if params.get('param_str') is None else params['param_str']
    assert agent.param_int is None if params.get('param_int') is None else params['param_int']
    assert agent.param_bool is False if params.get('param_bool') is None else params['param_bool']
    assert agent.param_float == 1.0 if params.get('param_float') is None else params['param_float']


@pytest.mark.parametrize(
    'params',
    [
        {},
        {'param_int': 9, 'param_float': 5.0},
        {'param_str': 'test', 'param_int': 3, 'param_bool': True, 'param_float': 4.0},
    ],
)
def test_learn(params):
    """Test the learn method."""
    agent = Agent(**params)
    env = Env()
    agent.learn(env)
    assert isinstance(env, Env)
    assert agent.param_str is None if params.get('param_str') is None else params['param_str']
    assert agent.param_int is None if params.get('param_int') is None else params['param_int']
    assert agent.param_bool is False if params.get('param_bool') is None else params['param_bool']
    assert agent.param_float == 1.0 if params.get('param_float') is None else params['param_float']
    assert agent.param_str_ is None if params.get('param_str') is None else params['param_str']
    assert agent.param_int_ is None if params.get('param_int') is None else params['param_int']
    assert agent.param_bool_ is False if params.get('param_bool') is None else params['param_bool']
    assert agent.param_float_ == 1.0 if params.get('param_float') is None else params['param_float']
    assert isinstance(agent.observation_space_, gym.spaces.Discrete)
    assert isinstance(agent.action_space_, gym.spaces.Discrete)
    assert agent.n_observations_ == agent.observation_space_.n
    assert agent.n_actions_ == agent.action_space_.n
    assert agent.results_ is None
    assert isinstance(agent.policy_, BasePolicy)
    assert isinstance(agent.env_, Env)
    assert isinstance(agent.eval_env_, Env)
    assert isinstance(agent.envs_, BaseVectorEnv)
    assert isinstance(agent.eval_envs_, BaseVectorEnv)
    assert agent.n_envs_ == 1
    assert agent.n_eval_envs_ == 1


def test_interact():
    """Test the interact method."""
    agent = Agent()
    env = Env()
    agent.learn(env)
    agent.interact(env)
