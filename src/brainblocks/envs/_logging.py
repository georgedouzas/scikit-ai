"""Implementation of logging wrapper classes."""

# Author: Georgios Douzas <gdouzas@icloud.com>

import warnings
from collections.abc import Callable
from typing import Any, Self, SupportsFloat

import gymnasium as gym
import numpy as np
from rich import print
from rich.layout import Layout
from rich.panel import Panel

with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    import pygame
from torch.utils.tensorboard import SummaryWriter
from torchvision.io import read_video

TRIGGER_EPISODE_NUM = 50


class LoggingTerminal(gym.Wrapper, gym.utils.RecordConstructorArgs):

    layout = Layout()
    layout.split_column(
        Layout(name='Title'),
        Layout(name='Learning'),
        Layout(name='Evaluation'),
        Layout(name='Interaction'),
    )
    layout['Title'].visible = False
    layout['Learning'].visible = False
    layout['Evaluation'].visible = False
    layout['Interaction'].visible = False

    def __init__(
        self: Self,
        env: gym.Env,
        env_type: str,
        episode_trigger: Callable[[int], bool] | None = None,
        window_size: int | None = None,
    ) -> None:
        super().__init__(env)
        self.env_type = env_type
        self.episode_trigger = (
            (lambda num: num % TRIGGER_EPISODE_NUM == 0) if episode_trigger is None else episode_trigger
        )
        self.window_size = TRIGGER_EPISODE_NUM // 5 if window_size is None else window_size
        self.last_episode_num = None
        self.layout['Title'].visible = True

    def reset(
        self: Self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[gym.core.ObsType, dict[str, Any]]:
        return super().reset(seed=seed, options=options)

    def step(
        self: Self,
        action: gym.core.ActType,
    ) -> tuple[gym.core.ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        episode_num = self.episode_count
        if (self.episode_trigger(episode_num) and self.last_episode_num != episode_num and episode_num > 0) or (
            self.last_episode_num != episode_num and episode_num == 1
        ):
            self.last_episode_num = episode_num
            average_reward = np.mean(list(self.return_queue)[-self.window_size :])
            average_length = np.mean(list(self.length_queue)[-self.window_size :])
            self.layout[self.env_type].update(
                Panel(
                    f'[bold]{self.env_type}[/]\nEpisode: {episode_num}\nReward: {average_reward:.2f}\nLength: {average_length:.2f}'
                ),
            )
            self.layout[self.env_type].visible = True
            print(self.layout)

        return super().step(action)


class LoggingTensorboard(gym.wrappers.RecordVideo):

    def __init__(
        self: Self,
        env: gym.Env,
        log_dir_path: str | None = None,
        video_folder: str | None = None,
        video_episode_trigger: Callable[[int], bool] | None = None,
        video_step_trigger: Callable[[int], bool] | None = None,
        video_length: int | None = None,
        video_name_prefix: str | None = None,
    ) -> None:
        self.log_dir_path = 'runs' if log_dir_path is None else log_dir_path
        self.video_folder = self.log_dir_path if video_folder is None else video_folder
        self.video_episode_trigger = (
            (lambda num: num % TRIGGER_EPISODE_NUM == 0)
            if (video_episode_trigger is None and video_step_trigger is None)
            else video_episode_trigger
        )
        self.video_step_trigger = video_step_trigger
        self.video_length = 0 if video_length is None else video_length
        self.video_name_prefix = 'video' if video_name_prefix is None else video_name_prefix
        self.writer = SummaryWriter(self.log_dir_path)
        super().__init__(
            env=env,
            video_folder=self.video_folder,
            episode_trigger=self.video_episode_trigger,
            step_trigger=self.video_step_trigger,
            video_length=self.video_length,
            name_prefix=self.video_name_prefix,
            disable_logger=True,
        )

    def close_video_recorder(self: Self) -> None:
        if self.video_recorder is not None:
            video_tensor = read_video(self.video_recorder.path, output_format='tchw', pts_unit='sec')[0]
            if video_tensor.nelement() > 0:
                video_tensor = video_tensor.reshape(-1, *video_tensor.shape)
                self.writer.add_video(f'Video of episode {self.episode_id}', video_tensor)
        super().close_video_recorder()
        pygame.init()

    def step(
        self: Self,
        action: gym.core.ActType,
    ) -> tuple[gym.core.ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        if self.return_queue:
            self.writer.add_scalar('Reward', self.return_queue[-1][0], self.step_id)
            self.writer.add_scalar('Length', self.length_queue[-1][0], self.step_id)
        return super().step(action)

    def close(self: Self) -> None:
        self.writer.flush()
        self.writer.close()
        return super().close()
