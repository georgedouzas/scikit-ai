"""Implementation of base class of agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>

from inspect import signature
from pathlib import Path
from typing import Any, ClassVar, Self, TypeAlias

import gymnasium as gym
import numpy as np
import tianshou
from rich.panel import Panel
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_scalar
from tensorboard import program

from ..envs import LoggingTensorboard, LoggingTerminal

EnvType: TypeAlias = (
    gym.Env | gym.experimental.AsyncVectorEnv | gym.experimental.SyncVectorEnv | tianshou.env.BaseVectorEnv
)

DEQUE_SIZE = 10000


def check_base_env(env: EnvType, param_name: str) -> gym.Env:
    """Check the base environment.

    Args:
        env:
            A simple or vectorized Gymnasium or Tianshou environment.

        param_name:
            The identifier of the environment.

    Returns:
        base_env:
            The base Gymnasium environment.
    """

    # Check environment type
    if not isinstance(env, EnvType):
        error_msg = (
            f'Parameter `{param_name}` should be an object from either `gym.Env` or `gym.vector.VectorEnv` or '
            f'`tianshou.env.BaseVectorEnv` class. Got {type(env)} instead.'
        )
        raise TypeError(error_msg)

    # Check specification
    if isinstance(env, gym.Env | gym.experimental.AsyncVectorEnv | gym.experimental.SyncVectorEnv):
        spec = env.spec
        render_modes = env.metadata['render_modes']
    else:
        spec = env.spec[0]
        render_modes = env.metadata[0]['render_modes']
    if spec is None:
        error_msg = (
            f'Environment `{param_name}` has a `spec` attribute equal to `None`. Please '
            'use a `gymnasium` environment with a specification.'
        )
        raise ValueError(error_msg)

    # Create base environment
    base_env = gym.wrappers.RecordEpisodeStatistics(
        gym.make(spec, render_mode='rgb_array') if 'rgb_array' in render_modes else gym.make(spec),
        deque_size=DEQUE_SIZE,
    )

    return base_env


def get_log_dir_path(env_name: str, env_type: str, agent_name: str, logging_path: str) -> Path:
    """Get the logging directory.

    Args:
        env_name:
            The environment's name.

        env_type:
            The type of environment.

        agent_name:
            The agent's name.

        logging_path:
            The path of the logging files.

    Returns:
        log_dir_path:
            The logging directory.
    """
    path = Path(logging_path) / 'runs'
    path.mkdir(exist_ok=True)
    env_paths = [env_path for env_path in path.iterdir() if env_path.name == env_name]
    env_path = env_paths[0] if env_paths else path / env_name
    env_train_path = env_path / env_type.title() / agent_name
    n_exps = len(list(env_train_path.iterdir())) if env_train_path.exists() else 0
    log_dir_path = env_train_path / f'Experiment {n_exps + 1}'
    return log_dir_path


def extract_env_info(wrapped_env: gym.Env) -> tuple[gym.spaces.Space, int, gym.spaces.Space, int]:
    """Extract information for the environment.

    Args:
        wrapped_env:
            The wrapped gymnasium environment to extract info.

    Returns:
        info:
            The environment's spaces and their dimensionality.
    """
    observation_space = wrapped_env.observation_space
    action_space = wrapped_env.action_space
    n_observations = np.prod(observation_space.shape).astype(int) if observation_space.shape else observation_space.n
    n_actions = np.prod(action_space.shape).astype(int) if action_space.shape else action_space.n
    info = observation_space, n_observations, action_space, n_actions
    return info


def check_vectorized_envs(
    wrapped_env: gym.Env,
    n_envs: int,
    backend: str,
) -> EnvType:
    """Check the vectorized environments.

    Args:
        wrapped_env:
            The wrapped Gymnasium environment.
        n_envs:
            The number of environments.

        backend:
            The selected backend.

    Returns:
        vectorized_envs:
            The vectorized environments.
    """
    if backend != 'tianshou':
        vectorized_envs = gym.make_vec(
            wrapped_env.spec,
            num_envs=n_envs,
            vectorization_mode='async' if n_envs > 1 else 'sync',
        )
    else:
        vectorized_envs = (
            tianshou.env.ShmemVectorEnv([lambda: wrapped_env for _ in range(n_envs)])
            if n_envs > 1
            else tianshou.env.DummyVectorEnv([lambda: wrapped_env])
        )
    return vectorized_envs


class BaseAgent(BaseEstimator):
    """Base class for agents."""

    _backends: ClassVar[list[str]] = ['native']
    _non_numeric_params: ClassVar[dict[str, tuple[Any, ...]]] = {
        'backend': (str,),
    }
    _numeric_params: ClassVar[dict[str, dict]] = {}
    _default_parameters: ClassVar[dict[str, dict]] = {
        'native': {
            'backend': 'native',
        },
    }
    _optimal_parameters: ClassVar[dict[str, dict]] = {}

    MAX_EPISODES = 100

    def __init__(self: Self, backend: str | None = None) -> None:
        self.backend = backend

    @staticmethod
    def _n_envs(env: EnvType) -> int:
        if isinstance(env, gym.Env):
            n_envs = 1
        elif isinstance(env, gym.experimental.AsyncVectorEnv | gym.experimental.SyncVectorEnv):
            n_envs = env.num_envs
        elif isinstance(env, tianshou.env.BaseVectorEnv):
            n_envs = env.env_num
        return n_envs

    def _check_backend(self: Self) -> Self:
        if not self._backends:
            error_msg = 'No available backends are set for this agent.'
            raise ValueError(error_msg)
        if self.backend is not None and self.backend not in self._backends:
            error_msg = (
                f'Parameter `{self.backend}` should be one of {", ".join(self._backend)} or `None`. '
                f'Got {self.backend} instead.'
            )
            raise ValueError(error_msg)
        return self

    def _check_params_attrs(self: Self) -> Self:

        # Check default values of initialization
        error_msg = 'All parameters default value in agent\'s initialization method should be `None`.'
        assert all(param.default is None for param in signature(self.__init__).parameters.values()), error_msg  # type: ignore[misc]

        # Non numeric
        for param_name, param_types in self._non_numeric_params.items():
            attr_val = getattr(self, param_name)
            if attr_val is not None and not isinstance(attr_val, param_types):
                error_instance_msg = ' or '.join([str(attr) for attr in param_types]) + ' or `None`'
                error_msg = (
                    f'Parameter `{param_name}` should be an instance of '
                    f'{error_instance_msg}. Got {type(attr_val)} instead.'
                )
                raise TypeError(error_msg)
            if hasattr(self, f'_check_{param_name}'):
                getattr(self, f'_check_{param_name}')()

        # Numeric
        for param_name, param_info in self._numeric_params.items():
            attr_val = getattr(self, param_name)
            if attr_val is not None:
                check_scalar(
                    attr_val,
                    name=param_name,
                    target_type=param_info['target_type'],
                    min_val=param_info.get('min_val'),
                    max_val=param_info.get('max_val'),
                )

        # Get backend
        backend = self._backends[0] if self.backend is None else self.backend

        # Get default parameters
        default_params = self._default_parameters.get(backend, {})
        if default_params:
            error_msg = 'Provided default parameters are not complete.'
            assert sorted(default_params) == sorted(signature(self.__init__).parameters), error_msg  # type: ignore[misc]

        # Get environment specific parameters
        params_env = self._optimal_parameters.get(self.base_env_.spec.name, {}).get(backend, {})

        # Create attributes
        for param_name, param_val_default in default_params.items():
            param_val = getattr(self, param_name)
            if param_val is None:
                param_val = params_env.get(param_name, param_val_default)
            setattr(self, f'{param_name}_', param_val)

        return self

    def _launch_tensorboard(self: Self, logging_path: str) -> str:
        tensorboard = program.TensorBoard()
        path = Path(logging_path) / 'runs'
        tensorboard.configure(argv=[None, '--logdir', str(path)])
        tensorboard_url = tensorboard.launch()
        return tensorboard_url

    def _learn(
        self: Self,
        env: EnvType,
        eval_env: EnvType | None,
        logging_terminal: bool | dict | None,
        logging_tensorboard: bool | dict | None,
    ) -> Self:

        # Base environments
        self.base_env_ = check_base_env(env, 'env')
        eval_env = eval_env if eval_env is not None else env
        self.base_eval_env_ = check_base_env(eval_env, 'eval_env')

        # Check attributes
        self._check_params_attrs()

        # Check wrapped environments
        if logging_terminal is None:
            logging_terminal = True
        if logging_tensorboard is None:
            logging_tensorboard = True
        self.wrapped_env_ = self.base_env_
        self.wrapped_eval_env_ = self.base_eval_env_
        if logging_terminal:
            if isinstance(logging_terminal, bool):
                logging_terminal = {}
            self.wrapped_env_ = LoggingTerminal(env=self.wrapped_env_, env_type='Learning', **logging_terminal)
            self.wrapped_eval_env_ = LoggingTerminal(
                env=self.wrapped_eval_env_,
                env_type='Evaluation',
                **logging_terminal,
            )
        title = ''
        if logging_tensorboard:
            if isinstance(logging_tensorboard, bool):
                logging_tensorboard = {}
            logging_path = logging_tensorboard.pop('logging_path', '.')
            tensorboard_url = self._launch_tensorboard(logging_path)
            title += f'\nTensorboard: {tensorboard_url}'
            learning_log_dir_path = get_log_dir_path(
                self.base_env_.spec.name,
                'Learning',
                self.__class__.__name__,
                logging_path,
            )
            evaluation_log_dir_path = get_log_dir_path(
                self.base_env_.spec.name,
                'Evaluation',
                self.__class__.__name__,
                logging_path,
            )
            logging_tensorboard.pop('logging_path', None)
            self.wrapped_env_ = LoggingTensorboard(
                env=self.wrapped_env_,
                log_dir_path=learning_log_dir_path,
                **logging_tensorboard,
            )
            self.wrapped_eval_env_ = LoggingTensorboard(
                env=self.wrapped_eval_env_,
                log_dir_path=evaluation_log_dir_path,
                **logging_tensorboard,
            )

        # Check vectorized environments
        self.envs_ = check_vectorized_envs(self.wrapped_env_, self._n_envs(env), self.backend_)
        self.eval_envs_ = check_vectorized_envs(self.wrapped_eval_env_, self._n_envs(eval_env), self.backend_)

        # Environment features
        self.observation_space_, self.n_observations_, self.action_space_, self.n_actions_ = extract_env_info(
            self.wrapped_env_,
        )

        # Logging
        title = (
            f'[bold]Information[/]\nAgent: {self.__class__.__name__}\nEnvironment: {self.base_env_.spec.name}' + title
        )
        LoggingTerminal.layout['Title'].update(Panel(title))

        # Results
        self.learn_results_ = None

        # Policy
        self.policy_ = None

        return self

    def _interact(
        self: Self,
        env: EnvType,
        logging_terminal: bool | dict | None,
        logging_tensorboard: bool | dict | None,
        n_episodes: int | None,
        n_steps: int | None,
        **kwargs: dict,
    ) -> tuple[EnvType, int | None, int | None]:

        # Check agent is fitted
        if (
            not hasattr(self, 'learn_results_')
            or self.learn_results_ is None
            or not hasattr(self, 'policy_')
            or self.policy_ is None
        ):
            error_msg = (
                f'The `{self.__class__.__name__}` instance has not learned from interacting with the environment. '
                'Call `learn` with appropriate arguments before using this agent.'
            )
            raise NotFittedError(error_msg)

        # Check environment
        base_env = check_base_env(env, 'env')
        wrapped_env = base_env
        if logging_terminal is None:
            logging_terminal = True
        if logging_tensorboard is None:
            logging_tensorboard = True
        if logging_terminal:
            if isinstance(logging_terminal, bool):
                logging_terminal = {}
            wrapped_env = LoggingTerminal(env=wrapped_env, env_type='Interaction', **logging_terminal)
        if logging_tensorboard:
            if isinstance(logging_tensorboard, bool):
                logging_tensorboard = {}
            logging_path = logging_tensorboard.pop('logging_path', '.')
            log_dir_path = get_log_dir_path(base_env.spec.name, 'Interaction', self.__class__.__name__, logging_path)
            logging_tensorboard.pop('logging_path', None)
            wrapped_env = LoggingTensorboard(env=wrapped_env, log_dir_path=log_dir_path, **logging_tensorboard)
        envs = check_vectorized_envs(wrapped_env, self._n_envs(env), self.backend_)

        # Check parameters
        max_episodes = 10
        if n_steps is None and n_episodes is None:
            n_episodes = base_env.spec.max_episode_steps if base_env.spec is not None else max_episodes
            if n_episodes is None:
                n_episodes = max_episodes
        elif n_episodes is None:
            check_scalar(n_steps, 'n_steps', int, min_val=1)
        elif n_steps is None:
            check_scalar(n_episodes, 'n_episodes', int, min_val=1)

        return envs, n_episodes, n_steps

    def learn(
        self: Self,
        env: EnvType,
        eval_env: EnvType | None = None,
        logging_terminal: bool | dict | None = None,
        logging_tensorboard: bool | dict | None = None,
    ) -> Self:
        """Learn from online or offline interaction with the environment."""
        return self._learn(env, eval_env, logging_terminal, logging_tensorboard)

    def interact(
        self: Self,
        env: EnvType,
        logging_terminal: bool | dict | None = None,
        logging_tensorboard: bool | dict | None = None,
        n_episodes: int | None = None,
        n_steps: int | None = None,
        **kwargs: dict,
    ) -> dict[str, Any]:
        """Interact with the environment."""
        return {}
