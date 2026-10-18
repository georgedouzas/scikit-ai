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
        ...     def generate_text(self, request):
        ...         return request.sample
        >>> register('echo', EchoBackend)
    """
    if not (isinstance(backend_class, type) and issubclass(backend_class, Backend)):
        msg = f'The backend class of `{name}` must be a subclass of `Backend`.'
        raise TypeError(msg)
    BACKENDS[name] = backend_class


def list_backends(
    task: str | None = None,
    predicate: Callable[[type[Backend]], bool] | None = None,
    input_kind: str | None = None,
) -> list[str]:
    """List the names of the registered backends.

    Args:
        task:
            Keep only the backends that support this task.

        predicate:
            Keep only the backend classes for which this function returns `True`.

        input_kind:
            Keep only the backends that run the task on samples of this input kind.

    Returns:
        The sorted backend names.

    Examples:
        >>> from skai.core import list_backends
        >>> 'dummy' in list_backends('generate_text')
        True
    """
    _load_entry_points()
    return sorted(
        name
        for name, backend_class in BACKENDS.items()
        if (task is None or backend_class.supports(task, input_kind))
        and (predicate is None or predicate(backend_class))
    )


def resolve_backend(backend: str | Backend, task: str, input_kind: str | None = None) -> Backend:
    """Turn the `backend` parameter of a component into a connected backend that supports a task.

    Args:
        backend:
            A string `'<provider>'` or `'<provider>:<model>'`, or a backend object, which is cloned.

        task:
            The task the backend must support.

        input_kind:
            The input kind of the samples the backend must accept for the task.

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
            backends = list_backends(task, input_kind=input_kind)
            msg = f'Unknown backend `{name}`. Backends that support `{task}`: {backends}.'
            raise UnsupportedBackendError(msg)
        instance = BACKENDS[name](model=model or None)
    if not type(instance).supports(task, input_kind):
        on = '' if input_kind is None else f' on {input_kind} samples'
        msg = (
            f'{type(instance).__name__} does not support `{task}`{on}. '
            f'Backends that do: {list_backends(task, input_kind=input_kind)}.'
        )
        raise UnsupportedBackendError(msg)
    return instance.connect()
