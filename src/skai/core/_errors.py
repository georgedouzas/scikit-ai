"""Define the exceptions and warnings raised by scikit-ai."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT


class UnsupportedBackendError(ValueError):
    """Signal that a backend is unknown or cannot run a task."""


class MissingExtraError(ImportError):
    """Signal that an optional dependency group is not installed."""


class MissingCredentialError(RuntimeError):
    """Signal that the environment variable holding a credential is not set."""


class InvalidSampleError(TypeError):
    """Signal that a sample has the wrong type for a component."""


class MediaReadError(ValueError):
    """Signal that a media sample cannot be read or decoded."""


class InvalidResponseError(ValueError):
    """Signal that a backend returned a response that does not fit the task."""


class FineTuneError(RuntimeError):
    """Signal that a fine-tuning job failed or was cancelled."""


class SampleError(RuntimeError):
    """Signal that processing one sample failed."""


class SampleFailureWarning(UserWarning):
    """Warn that some samples failed and were replaced by missing values."""


class UnusedFitDataWarning(UserWarning):
    """Warn that the backend cannot use something the component learned in `fit`."""
