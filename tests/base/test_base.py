"""Test the base agent class."""

from typing import Any, ClassVar, Self, SupportsFloat

import gymnasium as gym
import numpy as np
import pytest
import tianshou as ts

from skai.base import (
    BaseAgent,
    EnvType,
    check_base_env,
    check_vectorized_envs,
    extract_env_info,
    get_log_dir_path,
)

ENV = gym.wrappers.FlattenObservation(gym.make('Blackjack-v1'))
DEFAULT_PARAMETERS = {
    'backend': 'native',
    'param_str': 'test',
    'param_int': 1,
    'param_bool': False,
    'param_float': 1.0,
}


class Env(gym.Env):
    """Test environment."""

    metadata: ClassVar[dict] = {'render_modes': ['human', 'rgb_array']}
    spec: gym.envs.registration.EnvSpec = gym.envs.registration.EnvSpec(id='Env-v1', entry_point='tests.base.test_base:Env')

    def __init__(self, render_mode: str | None = None):
        self.observation_space = gym.spaces.Discrete(4)
        self.action_space = gym.spaces.Discrete(2)
        self.render_mode = render_mode

    def step(
        self: Self,
        action: gym.core.ActType,
    ) -> tuple[gym.core.ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        return action % 2


class Agent(BaseAgent):
    """Test agent."""

    _non_numeric_params: ClassVar[dict[str, tuple[Any, ...]]] = {
        'param_str': (str,),
        'param_bool': (bool,),
        'backend': (str,),
    }
    _numeric_params: ClassVar[dict[str, dict]] = {
        'param_int': {'target_type': int, 'min_val': 1},
        'param_float': {'target_type': float, 'min_val': 1.0, 'max_val': 6.0},
    }
    _default_parameters: ClassVar[dict[str, dict]] = {'native': DEFAULT_PARAMETERS}

    def __init__(
        self,
        param_str: str | None = None,
        param_int: int | None = None,
        param_bool: bool | None = None,
        param_float: float | None = None,
        backend: str | None = None,
    ) -> None:
        """Initialize the agent."""
        super().__init__(backend=backend)
        self.param_str = param_str
        self.param_int = param_int
        self.param_bool = param_bool
        self.param_float = param_float

    def _learn(
        self: Self,
        env: EnvType,
        eval_env: EnvType | None,
        logging_terminal: bool | dict | None,
        logging_tensorboard: bool | dict | None,
    ) -> Self:
        super()._learn(env, eval_env, logging_terminal, logging_tensorboard)
        self.policy_: dict[str, str] = {}
        self.learn_results_: dict[str, float] = {}
        return self


@pytest.mark.parametrize('param_name', ['env', 'eval_env'])
@pytest.mark.parametrize('env', [None, 5, ['env'], 'env'])
def test_check_base_env_wrong_type_error(env, param_name):
    """Check raising of type error."""
    with pytest.raises(TypeError, match=f'Parameter `{param_name}` should be an object'):
        check_base_env(env, param_name)


@pytest.mark.parametrize('param_name', ['env', 'eval_env'])
def test_check_base_env_no_spec_error(param_name):
    """Check raising of value error."""
    env = gym.Env()
    with pytest.raises(ValueError, match=f'Environment `{param_name}` has a `spec` attribute'):
        check_base_env(env, param_name)


@pytest.mark.parametrize('param_name', ['env', 'eval_env'])
@pytest.mark.parametrize(
    'env', [ENV, gym.make_vec(ENV.spec, num_envs=2), ts.env.ShmemVectorEnv([lambda: ENV for _ in range(2)])]
)
def test_check_base_env(env, param_name):
    """Test check of gymnasium environment."""
    base_env = check_base_env(env, param_name)
    assert isinstance(base_env, gym.Env)
    spec = env.get_env_attr('spec')[0] if isinstance(env, ts.env.ShmemVectorEnv) else env.spec
    assert base_env.spec.id == spec.id
    assert base_env.spec.entry_point == spec.entry_point
    assert base_env.spec.reward_threshold == spec.reward_threshold
    assert base_env.spec.nondeterministic == spec.nondeterministic
    assert base_env.spec.max_episode_steps == spec.max_episode_steps
    assert base_env.spec.order_enforce == spec.order_enforce
    assert base_env.spec.autoreset == spec.autoreset
    assert base_env.spec.disable_env_checker == spec.disable_env_checker
    assert base_env.spec.apply_api_compatibility == spec.apply_api_compatibility
    assert base_env.spec.namespace == spec.namespace
    assert base_env.spec.name == spec.name
    assert base_env.spec.vector_entry_point == spec.vector_entry_point
    assert base_env.spec.kwargs.pop('render_mode') == 'rgb_array'
    assert base_env.spec.kwargs == spec.kwargs


def test_get_log_dir_path(tmp_path):
    """Test the logging directory path."""
    log_dir_path = get_log_dir_path('test_env', 'Testing', 'TestAgent', tmp_path)
    assert log_dir_path.name == 'Experiment 1'
    assert log_dir_path.parent.name == 'TestAgent'
    assert log_dir_path.parent.parent.name == 'Testing'
    assert log_dir_path.parent.parent.parent.name == 'test_env'
    assert log_dir_path.parent.parent.parent.parent.name == 'runs'
    assert log_dir_path.parent.parent.parent.parent.parent.name == tmp_path.name


def test_extract_env_info():
    """Test the extraction of environment information."""
    observation_space, n_observations, action_space, n_actions = extract_env_info(ENV)
    assert observation_space == ENV.observation_space
    assert action_space == ENV.action_space
    assert n_observations == np.prod(ENV.observation_space.shape)
    assert n_actions == np.prod(ENV.action_space.n)


@pytest.mark.parametrize('backend', ['gymnasium', 'tianshou'])
@pytest.mark.parametrize('n_envs', [1, 2])
def test_check_vectorized_envs(n_envs, backend):
    """Test the extraction of vectorized environments."""
    envs = check_vectorized_envs(ENV, n_envs, backend)
    n_single_env, n_multi_envs = 1, 2
    if n_envs == n_single_env and backend == 'gymnasium':
        assert isinstance(envs, gym.experimental.SyncVectorEnv)
    elif n_envs == n_multi_envs and backend == 'gymnasium':
        assert isinstance(envs, gym.experimental.AsyncVectorEnv)
    elif n_envs == n_single_env and backend == 'tianshou':
        assert isinstance(envs, ts.env.DummyVectorEnv)
    elif n_envs == n_multi_envs and backend == 'tianshou':
        assert isinstance(envs, ts.env.ShmemVectorEnv)


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
    assert agent.param_bool is None if params.get('param_bool') is None else params['param_bool']
    assert agent.param_float is None if params.get('param_float') is None else params['param_float']


@pytest.mark.parametrize(
    'env', [ENV, gym.make_vec(ENV.spec, num_envs=2), ts.env.SubprocVectorEnv([lambda: ENV for _ in range(3)])]
)
@pytest.mark.parametrize('logging_terminal', [True, False])
@pytest.mark.parametrize('logging_tensorboard', [True, False])
@pytest.mark.parametrize(
    'params',
    [
        {},
        {'param_int': 9, 'param_float': 5.0},
        {'param_str': 'test', 'param_int': 3, 'param_bool': True, 'param_float': 4.0},
    ],
)
def test_learn(tmp_path, logging_terminal, logging_tensorboard, env, params):
    """Test the learn method."""
    if logging_tensorboard:
        logging_tensorboard = {'logging_path': tmp_path}
    agent = Agent(**params)
    agent.learn(env, logging_terminal=logging_terminal, logging_tensorboard=logging_tensorboard)
    assert agent.param_str is None if params.get('param_str') is None else params['param_str']
    assert agent.param_int is None if params.get('param_int') is None else params['param_int']
    assert agent.param_bool is None if params.get('param_bool') is None else params['param_bool']
    assert agent.param_float is None if params.get('param_float') is None else params['param_float']
    assert agent.param_str_ == DEFAULT_PARAMETERS['param_str'] if agent.param_str is None else agent.param_str
    assert agent.param_int_ == DEFAULT_PARAMETERS['param_int'] if agent.param_int is None else agent.param_int
    assert agent.param_bool_ == DEFAULT_PARAMETERS['param_bool'] if agent.param_bool is None else agent.param_bool
    assert agent.param_float_ == DEFAULT_PARAMETERS['param_float'] if agent.param_float is None else agent.param_float
    assert isinstance(agent.observation_space_, gym.spaces.Box)
    assert isinstance(agent.action_space_, gym.spaces.Discrete)
    assert agent.n_actions_ == agent.action_space_.n
    assert agent.learn_results_ == {}
    assert isinstance(agent.policy_, dict)
    assert isinstance(agent.base_env_, gym.Env)
    assert isinstance(agent.base_eval_env_, gym.Env)
    assert isinstance(agent.envs_, EnvType)
    assert isinstance(agent.eval_envs_, EnvType)


def test_interact():
    """Test the interact method."""
    agent = Agent()
    agent.learn(ENV)
    assert agent.interact(ENV) == {}
