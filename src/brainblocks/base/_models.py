"""Implementation of models."""

# Author: Georgios Douzas <gdouzas@icloud.com>

from typing import Any, Self

import torch

OBS = Any
STATE = Any


class DQNModel(torch.nn.Module):
    """Default model for the DQN agent."""

    def __init__(self: Self, n_observations: int, n_actions: int) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_observations, n_actions),
        )

    def forward(
        self: Self,
        obs: OBS,
        state: STATE | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[Any, STATE]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state


class DQNCartPoleModel(torch.nn.Module):
    """Model for the DQN agent and CartPole environment."""

    def __init__(self: Self, n_observations: int, n_actions: int) -> None:
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(n_observations, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(128, n_actions),
        )

    def forward(
        self: Self,
        obs: OBS,
        state: STATE | None = None,
        info: dict[str, Any] | None = None,
    ) -> tuple[Any, STATE]:
        if not isinstance(obs, torch.Tensor):
            obs = torch.tensor(obs, dtype=torch.float)
        batch = obs.shape[0]
        logits = self.model(obs.view(batch, -1))
        return logits, state
