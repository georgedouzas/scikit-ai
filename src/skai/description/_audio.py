"""Describe audio in natural language."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from numpy.typing import ArrayLike
from sklearn.utils._param_validation import StrOptions

from skai.core import Audio, Backend, BaseSupervisedGenerator, DescriptionRequest


class AudioDescriptor(BaseSupervisedGenerator):
    """Return one natural-language description of the sounds, music and speech of every audio sample.

    `fit(X, y)` with reference descriptions as `y` passes `k_shot` audio and description pairs to the backend as
    examples, on backends that use them.

    Args:
        detail:
            `'brief'` for one sentence or `'detailed'` for a paragraph.

        instructions:
            Extra guidance on what to describe, such as `'Name the instruments.'`.

        k_shot:
            How many reference pairs are passed to the backend as examples. `None` passes all of them up to
            `max_examples`, `0` passes none, an integer selects that many at random, and an array of indices selects
            those pairs.

        max_examples:
            The maximum number of examples when `k_shot` is `None`.

        use_y:
            `'auto'` learns from `y` in `fit` but not in `fit_transform`, which pipelines use for intermediate steps.
            `True` always learns from `y`, and `False` never does.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of samples submitted to the backend at once.

        on_error:
            `'raise'` raises when a sample fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the selection of examples and the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        examples_ (tuple):
            The audio and reference description pairs passed to the backend.

    Examples:
        >>> import numpy as np
        >>> from skai.description import AudioDescriptor
        >>> AudioDescriptor(backend='dummy').fit().predict([(np.zeros(16000), 16000)])[0]
        'A brief description of 1.00 seconds of audio.'
    """

    task: ClassVar[str] = 'describe_audio'
    input_kind: ClassVar[Literal['audio']] = 'audio'
    _parameter_constraints: ClassVar[dict] = {
        **BaseSupervisedGenerator._parameter_constraints,
        'detail': [StrOptions({'brief', 'detailed'})],
        'instructions': [str, None],
    }

    def __init__(
        self,
        *,
        detail: Literal['brief', 'detailed'] = 'brief',
        instructions: str | None = None,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 4,
        use_y: Literal['auto'] | bool = 'auto',
        backend: str | Backend | None = None,
        batch_size: int = 8,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.detail = detail
        self.instructions = instructions
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _build_request(self, sample: Audio, seed: int) -> DescriptionRequest:
        """Build the description request of one audio sample."""
        return DescriptionRequest(sample, self.detail, self.instructions, self.examples_)
