"""Import optional dependencies and read credentials."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import importlib
import os
from numbers import Integral
from types import ModuleType

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils._param_validation import InvalidParameterError

from ._errors import MissingCredentialError, MissingExtraError

MAX_SEED = 2**31 - 1


def draw_seeds(random_state: int | np.random.RandomState | None, count: int) -> list[int]:
    """Draw the seeds that components send to backends.

    Args:
        random_state:
            The `random_state` parameter of the component.

        count:
            The number of seeds.

    Returns:
        The seeds, reproducible for an integer `random_state`.
    """
    return [int(seed) for seed in check_random_state(random_state).randint(0, MAX_SEED, size=count)]


def import_optional(module: str, extra: str) -> ModuleType:
    """Import a module that belongs to an optional dependency group.

    Args:
        module:
            The name of the module to import.

        extra:
            The name of the optional dependency group that provides the module.

    Returns:
        The imported module.

    Raises:
        MissingExtraError:
            If the module is not installed.
    """
    try:
        return importlib.import_module(module)
    except ModuleNotFoundError as error:
        msg = f'Module `{module}` is required. Install it with `pip install scikit-ai[{extra}]`.'
        raise MissingExtraError(msg) from error


SIZE_LENGTH = 2


def check_size(size: object, owner: str) -> tuple[int, int]:
    """Check that a size parameter is a pair of positive integers.

    Args:
        size:
            The value of the `size` parameter.

        owner:
            The name of the component, used in the error message.

    Returns:
        The width and height.

    Raises:
        InvalidParameterError:
            If the size is not a `(width, height)` pair of positive integers.
    """
    if not (
        isinstance(size, tuple)
        and len(size) == SIZE_LENGTH
        and all(isinstance(side, Integral) and int(side) > 0 for side in size)
    ):
        msg = f'The `size` parameter of {owner} must be a (width, height) pair of positive integers. Got {size}.'
        raise InvalidParameterError(msg)
    return int(size[0]), int(size[1])


def read_env(name: str) -> str:
    """Read the value of the environment variable that holds a credential.

    Args:
        name:
            The name of the environment variable.

    Returns:
        The value of the environment variable.

    Raises:
        MissingCredentialError:
            If the environment variable is not set or is empty.
    """
    value = os.environ.get(name)
    if not value:
        msg = f'Set the environment variable `{name}` to the credential of the backend.'
        raise MissingCredentialError(msg)
    return value
