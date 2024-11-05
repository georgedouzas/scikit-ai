"""Implementation of Q-learning algorithm and its variants."""

# Author: Georgios Douzas <gdouzas@icloud.com>

from collections.abc import Callable
from typing import Any, ClassVar, Self

import torch
from gymnasium import Env
from tianshou.data import Collector, ReplayBuffer
from tianshou.env import BaseVectorEnv
from tianshou.policy import DQNPolicy
from tianshou.trainer import OffpolicyTrainer
from tianshou.utils.logger.base import BaseLogger
from tianshou.utils.lr_scheduler import MultipleLRSchedulers
from tianshou.utils.net.common import Net
from torch.optim.lr_scheduler import LRScheduler

from ..base import BaseAgent


class DQNAgent(BaseAgent):
    """Implementation of Deep Q Network and its variants."""

    _non_numeric_params: ClassVar[dict[str, tuple[Any, ...]]] = {
        'model': (torch.nn.Module, Net),
        'optim': (torch.optim.Optimizer,),
        'lr_scheduler': (LRScheduler, MultipleLRSchedulers),
        'reward_normalization': (bool,),
        'is_double': (bool,),
        'clip_loss_grad': (bool,),
        'train_collector': (Collector,),
        'test_collector': (Collector,),
        'buffer': (ReplayBuffer,),
        'train_fn': (Callable,),
        'test_fn': (Callable,),
        'stop_fn': (Callable,),
        'save_best_fn': (Callable,),
        'save_checkpoint_fn': (Callable,),
        'resume_from_log': (bool,),
        'reward_metric': (Callable,),
        'logger': (BaseLogger,),
        'verbose': (bool,),
        'show_progress': (bool,),
        'test_in_train': (bool,),
    }
    _numeric_params: ClassVar[dict[str, dict]] = {
        'discount_factor': {'target_type': float, 'min_val': 0.0, 'max_val': 1.0},
        'estimation_step': {'target_type': int, 'min_val': 0},
        'target_update_freq': {'target_type': int, 'min_val': 0},
        'max_epoch': {'target_type': int, 'min_val': 0},
        'batch_size': {'target_type': int, 'min_val': 0},
        'step_per_epoch': {'target_type': int, 'min_val': 0},
        'repeat_per_collect': {'target_type': int, 'min_val': 0},
        'episode_per_test': {'target_type': int, 'min_val': 0},
        'update_per_step': {'target_type': float, 'min_val': 0},
        'step_per_collect': {'target_type': int, 'min_val': 0},
        'episode_per_collect': {'target_type': int, 'min_val': 0},
    }

    def __init__(
        self: Self,
        model: torch.nn.Module | Net | None = None,
        optim: torch.optim.Optimizer | None = None,
        discount_factor: float | None = None,
        estimation_step: int | None = None,
        target_update_freq: int | None = None,
        reward_normalization: bool | None = None,
        is_double: bool | None = None,
        clip_loss_grad: bool | None = None,
        lr_scheduler: LRScheduler | MultipleLRSchedulers | None = None,
        train_collector: Collector | None = None,
        test_collector: Collector | None = None,
        max_epoch: int | None = None,
        batch_size: int | None = None,
        buffer: ReplayBuffer | None = None,
        step_per_epoch: int | None = None,
        repeat_per_collect: int | None = None,
        episode_per_test: int | None = None,
        update_per_step: float | None = None,
        step_per_collect: int | None = None,
        episode_per_collect: int | None = None,
        train_fn: Callable | None = None,
        test_fn: Callable | None = None,
        stop_fn: Callable | None = None,
        save_best_fn: Callable | None = None,
        save_checkpoint_fn: Callable | None = None,
        resume_from_log: bool | None = None,
        reward_metric: Callable | None = None,
        logger: BaseLogger | None = None,
        verbose: bool | None = None,
        show_progress: bool | None = None,
        test_in_train: bool | None = None,
    ) -> None:
        self.model = model
        self.optim = optim
        self.discount_factor = discount_factor
        self.estimation_step = estimation_step
        self.target_update_freq = target_update_freq
        self.reward_normalization = reward_normalization
        self.is_double = is_double
        self.clip_loss_grad = clip_loss_grad
        self.lr_scheduler = lr_scheduler
        self.train_collector = train_collector
        self.test_collector = test_collector
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.buffer = buffer
        self.step_per_epoch = step_per_epoch
        self.repeat_per_collect = repeat_per_collect
        self.episode_per_test = episode_per_test
        self.update_per_step = update_per_step
        self.step_per_collect = step_per_collect
        self.episode_per_collect = episode_per_collect
        self.train_fn = train_fn
        self.test_fn = test_fn
        self.stop_fn = stop_fn
        self.save_best_fn = save_best_fn
        self.save_checkpoint_fn = save_checkpoint_fn
        self.resume_from_log = resume_from_log
        self.reward_metric = reward_metric
        self.logger = logger
        self.verbose = verbose
        self.show_progress = show_progress
        self.test_in_train = test_in_train

    def _learn(self: Self, env: Env | BaseVectorEnv, test_env: Env | BaseVectorEnv | None) -> Self:

        super()._learn(env, test_env)

        # Check policy
        self.model_: torch.nn.Module | Net = self.model_(self)
        self.optim_: torch.optim.Optimizer = self.optim_(self)
        self.policy_ = DQNPolicy(
            model=self.model_,
            optim=self.optim_,
            discount_factor=self.discount_factor_,
            estimation_step=self.estimation_step_,
            target_update_freq=self.target_update_freq_,
            reward_normalization=self.reward_normalization_,
            is_double=self.is_double_,
            clip_loss_grad=self.clip_loss_grad_,
            lr_scheduler=self.lr_scheduler_,
            observation_space=self.observation_space_,
            action_space=self.action_space_,
        )

        # Check collectors
        self.train_collector_: Collector = self.train_collector_(self)
        self.test_collector_: Collector = self.test_collector_(self)

        # Check trainer
        if self.train_fn_ is not None:
            self.train_fn_: Callable = self.train_fn_(self)
        if self.test_fn_ is not None:
            self.test_fn_: Callable = self.test_fn_(self)
        if self.stop_fn_ is not None:
            self.stop_fn_: Callable = self.stop_fn_(self)
        if self.save_best_fn_ is not None:
            self.save_best_fn_: Callable = self.save_best_fn_(self)
        if self.save_checkpoint_fn_ is not None:
            self.save_checkpoint_fn_: Callable = self.save_checkpoint_fn_(self)
        if self.reward_metric_ is not None:
            self.reward_metric_: Callable = self.reward_metric_(self)
        self.trainer_ = OffpolicyTrainer(
            policy=self.policy_,
            train_collector=self.train_collector_,
            test_collector=self.test_collector_,
            max_epoch=self.max_epoch_,
            batch_size=self.batch_size_,
            buffer=self.buffer_,
            step_per_epoch=self.step_per_epoch_,
            repeat_per_collect=self.repeat_per_collect_,
            episode_per_test=self.episode_per_test_,
            update_per_step=self.update_per_step_,
            step_per_collect=self.step_per_collect_,
            episode_per_collect=self.episode_per_collect_,
            train_fn=self.train_fn_,
            test_fn=self.test_fn_,
            stop_fn=self.stop_fn_,
            save_best_fn=self.save_best_fn_,
            save_checkpoint_fn=self.save_checkpoint_fn_,
            resume_from_log=self.resume_from_log_,
            reward_metric=self.reward_metric_,
            logger=self.logger_,
            verbose=self.verbose_,
            show_progress=self.show_progress_,
            test_in_train=self.test_in_train_,
        )

        # Run trainer
        self.results_ = self.trainer_.run()

        return self
