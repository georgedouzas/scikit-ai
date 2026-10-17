"""Import optional dependencies and read credentials."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import importlib
import os
from types import ModuleType

from ._errors import MissingCredentialError, MissingExtraError


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
