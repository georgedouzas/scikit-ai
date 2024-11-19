"""Implementation of base class of agents."""

# Author: Georgios Douzas <gdouzas@icloud.com>

from typing import Self

from gymnasium import Env
from sklearn.base import BaseEstimator


class BaseAgent(BaseEstimator):
    """Base class for agents."""

    def learn(self: Self, env: Env) -> Self:
        """Learn from online or offline interaction with the environment."""
        return self

    def interact(self: Self, env: Env) -> dict:
        """Interact with the environment."""
        return {}
