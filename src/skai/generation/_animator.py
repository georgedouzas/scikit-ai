"""Animate images into short videos."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral, Real
from os import PathLike
from typing import ClassVar, Literal

from PIL import Image
from sklearn.utils._param_validation import Interval

from skai.core import AnimationRequest, Backend, BaseMediaGenerator


class ImageAnimator(BaseMediaGenerator):
    """Turn every image into a short video that starts from it.

    Args:
        motion_prompt:
            A description of the motion, such as `'slow zoom in'`, for backends that take one.

        duration:
            The length of the videos in seconds.

        fps:
            The number of frames per second, or `None` for the model's default.

        output_format:
            `'object'` returns `Video` objects, `'bytes'` returns MP4 bytes, and `'path'` writes MP4 files and returns
            their paths. `'bytes'` and `'path'` need the `video` extra for videos that are not already encoded.

        output_dir:
            The directory for `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of images submitted to the backend at once.

        on_error:
            `'raise'` raises when an image fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

    Examples:
        >>> from PIL import Image
        >>> from skai.generation import ImageAnimator
        >>> animator = ImageAnimator(motion_prompt='slow zoom', duration=2, fps=4, backend='dummy').fit()
        >>> video = animator.predict([Image.new('RGB', (32, 24))])[0]
        >>> video.duration, video.frames.shape[1:3]
        (2.0, (24, 32))
    """

    task: ClassVar[str] = 'animate_image'
    input_kind: ClassVar[Literal['image']] = 'image'
    output_kind: ClassVar[Literal['video']] = 'video'
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'motion_prompt': [str, None],
        'duration': [Interval(Real, 0, None, closed='neither')],
        'fps': [Interval(Integral, 1, None, closed='left'), None],
    }

    def __init__(
        self,
        *,
        motion_prompt: str | None = None,
        duration: float = 2.0,
        fps: int | None = None,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 1,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.motion_prompt = motion_prompt
        self.duration = duration
        self.fps = fps
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _build_request(self, sample: Image.Image, seed: int) -> AnimationRequest:
        """Build the animation request of one image."""
        return AnimationRequest(sample, self.motion_prompt, float(self.duration), self.fps, seed)
