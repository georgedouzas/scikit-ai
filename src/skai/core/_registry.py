"""Register backends by name and resolve the `backend` parameter of components."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from importlib.metadata import entry_points

from sklearn.base import clone

from ._backend import Backend
from ._errors import UnsupportedBackendError

ENTRY_POINT_GROUP = 'skai.backends'
BACKENDS: dict[str, type[Backend]] = {}


def _load_entry_points() -> None:
    """Add the backends that installed packages declare as entry points, without replacing registered ones."""
    for entry_point in entry_points(group=ENTRY_POINT_GROUP):
        if entry_point.name not in BACKENDS:
            BACKENDS[entry_point.name] = entry_point.load()


def register(name: str, backend_class: type[Backend]) -> None:
    """Register a backend class under a name, so the name can be used in `backend` strings.

    Args:
        name:
            The provider name, used as `'<name>'` or `'<name>:<model>'`.

        backend_class:
            The backend class.

    Raises:
        TypeError:
            If `backend_class` is not a subclass of `Backend`.

    Examples:
        >>> from skai.core import Backend, register
        >>> class EchoBackend(Backend):
        ...     def summarize(self, request):
        ...         return request.text
        >>> register('echo', EchoBackend)
    """
    if not (isinstance(backend_class, type) and issubclass(backend_class, Backend)):
        msg = f'The backend class of `{name}` must be a subclass of `Backend`.'
        raise TypeError(msg)
    BACKENDS[name] = backend_class


def list_backends(task: str | None = None, predicate: Callable[[type[Backend]], bool] | None = None) -> list[str]:
    """List the names of the registered backends.

    Args:
        task:
            Keep only the backends that support this task.

        predicate:
            Keep only the backend classes for which this function returns `True`.

    Returns:
        The sorted backend names.

    Examples:
        >>> from skai.core import list_backends
        >>> 'dummy' in list_backends('summarize')
        True
    """
    _load_entry_points()
    return sorted(
        name
        for name, backend_class in BACKENDS.items()
        if (task is None or task in backend_class.supported_tasks()) and (predicate is None or predicate(backend_class))
    )


def resolve_backend(backend: str | Backend, task: str) -> Backend:
    """Turn the `backend` parameter of a component into a connected backend that supports a task.

    Args:
        backend:
            A string `'<provider>'` or `'<provider>:<model>'`, or a backend object, which is cloned.

        task:
            The task the backend must support.

    Returns:
        The connected backend.

    Raises:
        UnsupportedBackendError:
            If the provider is unknown or does not support the task.
    """
    if isinstance(backend, Backend):
        instance = clone(backend)
    else:
        _load_entry_points()
        name, _, model = backend.partition(':')
        if name not in BACKENDS:
            msg = f'Unknown backend `{name}`. Backends that support `{task}`: {list_backends(task)}.'
            raise UnsupportedBackendError(msg)
        instance = BACKENDS[name](model=model or None)
    if task not in type(instance).supported_tasks():
        msg = f'{type(instance).__name__} does not support `{task}`. Backends that do: {list_backends(task)}.'
        raise UnsupportedBackendError(msg)
    return instance.connect()
