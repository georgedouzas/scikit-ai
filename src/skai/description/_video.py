"""Describe videos in natural language."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral
from typing import ClassVar, Literal

from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval, StrOptions

from skai.core import Backend, BaseSupervisedGenerator, DescriptionRequest, VideoFrames, read_video, sample_frames


class VideoDescriptor(BaseSupervisedGenerator):
    """Return one natural-language description per video, from frames sampled evenly over it.

    Backends that accept audio also receive the audio track. `fit(X, y)` with reference descriptions as `y` passes
    `k_shot` video and description pairs to the backend as examples.

    Args:
        detail:
            `'brief'` for one sentence or `'detailed'` for a paragraph.

        instructions:
            Extra guidance on what to describe, such as `'Focus on what the people do.'`.

        n_frames:
            The number of frames sampled from every video.

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
            The number of videos submitted to the backend at once.

        on_error:
            `'raise'` raises when a video fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the selection of examples and the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        examples_ (tuple):
            The video and reference description pairs passed to the backend.

    Examples:
        >>> import numpy as np
        >>> from skai.core import Video
        >>> from skai.description import VideoDescriptor
        >>> video = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10)
        >>> VideoDescriptor(n_frames=4, backend='dummy').fit().predict([video])[0]
        'A brief description of 4 frames of a 2.00 second video.'
    """

    task: ClassVar[str] = 'describe_video'
    input_kind: ClassVar[Literal['video']] = 'video'
    _parameter_constraints: ClassVar[dict] = {
        **BaseSupervisedGenerator._parameter_constraints,
        'detail': [StrOptions({'brief', 'detailed'})],
        'instructions': [str, None],
        'n_frames': [Interval(Integral, 1, None, closed='left')],
    }

    def __init__(
        self,
        *,
        detail: Literal['brief', 'detailed'] = 'brief',
        instructions: str | None = None,
        n_frames: int = 8,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 2,
        use_y: Literal['auto'] | bool = 'auto',
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.detail = detail
        self.instructions = instructions
        self.n_frames = n_frames
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _read(self, sample: object, index: int) -> VideoFrames:
        """Read a video and sample its frames."""
        return sample_frames(read_video(sample, index), self.n_frames)

    def _build_request(self, sample: VideoFrames, seed: int) -> DescriptionRequest:
        """Build the description request of one video."""
        return DescriptionRequest(sample, self.detail, self.instructions, self.examples_)
