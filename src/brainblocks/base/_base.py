"""Implementation of base class of agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>

from typing import Any, ClassVar, Self

import numpy as np
from gymnasium import Env
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.utils import check_scalar
from tianshou.data import Collector, CollectStats
from tianshou.env import BaseVectorEnv, DummyVectorEnv, SubprocVectorEnv

from ._config import DEFAULT_PARAMETERS, PARAMETERS


class BaseAgent(BaseEstimator):
    """Base class for agents."""

    _non_numeric_params: ClassVar[dict[str, tuple[Any, ...]]]
    _numeric_params: ClassVar[dict[str, dict]]

    @property
    def _default_params(self: Self) -> dict[str, dict[str, Any]]:
        return DEFAULT_PARAMETERS.get(self.__class__.__name__, {})

    @property
    def _params(self: Self) -> dict[str, object]:
        env_name = self.env_.spec.name if self.env_.spec is not None else None
        if env_name is not None:
            return PARAMETERS.get(self.__class__.__name__, {}).get(env_name, {})
        return {}

    def _check_non_numeric_params(self: Self) -> Self:
        for param_name, param_types in self._non_numeric_params.items():
            attr = getattr(self, param_name)
            if attr is not None and not isinstance(attr, param_types):
                error_instance_msg = ' or '.join([str(attr) for attr in param_types]) + ' or `None`'
                error_msg = (
                    f'Parameter `{param_name}` should be an instance of {error_instance_msg}. Got {type(attr)} instead.'
                )
                raise TypeError(error_msg)
            param_val = getattr(self, param_name)
            setattr(self, f'{param_name}_', self._default_params.get(param_name, param_val))
        return self

    def _check_numeric_params(self: Self) -> Self:
        for param_name, param_info in self._numeric_params.items():
            attr = getattr(self, param_name)
            if attr is not None:
                check_scalar(
                    attr,
                    name=param_name,
                    target_type=param_info['target_type'],
                    min_val=param_info.get('min_val'),
                    max_val=param_info.get('max_val'),
                )
            param_val = getattr(self, param_name)
            setattr(self, f'{param_name}_', self._default_params.get(param_name, param_val))
        return self

    def _check_env(self: Self, env: Env | BaseVectorEnv) -> Self:
        if isinstance(env, Env):
            self.env_ = env
        elif isinstance(env, BaseVectorEnv):
            self.env_ = self.envs_.get_env_attr('env')[0]
        else:
            error_msg = (
                f'Parameter `env` should be an instance of either `gymnasium.Env` or `tianshou.env.BaseVectorEnv`'
                f'. Got {type(env)} instead.',
            )
            raise TypeError(error_msg)
        return self

    def _check_eval_env(self: Self, eval_env: Env | BaseVectorEnv | None) -> Self:
        if isinstance(eval_env, Env):
            self.eval_env_ = eval_env
        elif isinstance(eval_env, BaseVectorEnv):
            self.eval_env_ = self.envs_.get_env_attr('env')[0]
        elif eval_env is None:
            self.eval_env_ = self.env_
        else:
            error_msg = (
                f'Parameter `eval_env` should be an instance of either `gymnasium.Env` or `tianshou.env.BaseVectorEnv`'
                f' or None. Got {type(eval_env)} instead.',
            )
            raise TypeError(error_msg)
        return self

    def _check_env_spaces_info(self: Self) -> Self:
        self.observation_space_ = self.env_.observation_space
        self.action_space_ = self.env_.action_space
        self.n_observations_ = (
            np.prod(self.observation_space_.shape).astype(int)
            if self.observation_space_.shape
            else self.observation_space_.n
        )
        self.n_actions_ = (
            np.prod(self.action_space_.shape).astype(int) if self.action_space_.shape else self.action_space_.n
        )
        return self

    def _check_params(self: Self) -> Self:
        for param_name, param_val in self._params.items():
            setattr(self, f'{param_name}_', param_val)
        return self

    def _check_n_envs(self: Self, env: Env | BaseVectorEnv) -> Self:
        self.n_envs_ = 1 if isinstance(env, Env) else env.env_num
        return self

    def _check_n_eval_envs(self: Self, eval_env: Env | BaseVectorEnv | None) -> Self:
        self.n_eval_envs_ = 1 if (isinstance(eval_env, Env) or eval_env is None) else eval_env.env_num
        return self

    def _check_envs(self: Self, env: Env | BaseVectorEnv) -> Self:
        vector_env = DummyVectorEnv if self.n_envs_ == 1 else SubprocVectorEnv
        self.envs_ = (
            vector_env(env_fns=[lambda: self.env_ for _ in range(self.n_envs_)]) if isinstance(env, Env) else env
        )
        return self

    def _check_eval_envs(
        self: Self,
        eval_env: Env | BaseVectorEnv | None,
    ) -> Self:
        vector_env = DummyVectorEnv if self.n_eval_envs_ == 1 else SubprocVectorEnv
        if eval_env is None or isinstance(eval_env, Env):
            self.eval_envs_ = vector_env(env_fns=[lambda: self.eval_env_ for _ in range(self.n_eval_envs_)])
        else:
            self.eval_envs_ = eval_env
        return self

    def _learn(self: Self, env: Env | BaseVectorEnv, eval_env: Env | BaseVectorEnv | None) -> Self:
        self._check_non_numeric_params()
        self._check_numeric_params()
        self._check_env(env)
        self._check_eval_env(eval_env)
        self._check_env_spaces_info()
        self._check_params()
        self._check_n_envs(env)
        self._check_n_eval_envs(eval_env)
        self._check_envs(env)
        self._check_eval_envs(eval_env)
        self.results_ = None
        self.policy_ = None
        return self

    def learn(self: Self, env: Env | BaseVectorEnv, eval_env: Env | BaseVectorEnv | None = None) -> Self:
        """Learn from online or offline interaction with the environment."""
        return self._learn(env, eval_env)

    def interact(self: Self, env: Env | BaseVectorEnv, **kwargs: dict) -> CollectStats:
        """Interact with the environment."""

        # Check invoke of learn
        if not hasattr(self, 'results_'):
            error_msg = (
                f'The `{self.__class__.__name__}` instance has not learned from interacting with the environment. '
                'Call `learn` with appropriate arguments before using this agent.'
            )
            raise NotFittedError(error_msg)
        elif not hasattr(self, 'policy_') or self.policy_ is None:
            error_msg = f'The `{self.__class__.__name__}` instance has not learned a policy.'
            raise NotFittedError(error_msg)
        if not isinstance(env, Env | BaseVectorEnv):
            error_msg = (
                f'Parameter `env` should be an instance of either `gymnasium.Env` or `tianshou.env.BaseVectorEnv`'
                f'. Got {type(env)} instead.'
            )
            raise TypeError(error_msg)
        elif isinstance(env, Env):
            env = DummyVectorEnv([lambda: env])

        # Check parameters
        n_steps = kwargs.get('n_steps')
        n_episodes = kwargs.get('n_episodes')
        if n_steps is None and n_episodes is None:
            n_episodes = getattr(env.get_env_attr('spec')[0], 'max_episode_steps', 1)
        elif n_episodes is None:
            check_scalar(n_steps, 'n_steps', int, min_val=1)
        elif n_steps is None:
            check_scalar(n_episodes, 'n_episodes', int, min_val=1)
        render_time = kwargs.get('render_time')
        if render_time is None:
            render_time = 1 / env.get_env_attr('metadata')[0].get('render_fps', np.finfo(float).max)
        check_scalar(render_time, 'render_time', float)
        reset_kwargs = {k: v for k, v in kwargs.items() if k not in ('n_steps', 'n_episodes', 'render_time')}

        # Interact
        self.policy_.eval()
        collector = Collector(self.policy_, env)
        return collector.collect(
            n_step=n_steps,
            n_episode=n_episodes,
            random=False,
            render=render_time,
            reset_before_collect=True,
            gym_reset_kwargs=reset_kwargs,
        )
