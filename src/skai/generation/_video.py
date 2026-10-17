"""Generate videos from text prompts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral, Real
from os import PathLike
from typing import ClassVar, Literal

from sklearn.utils._param_validation import Interval

from skai.core import Backend, BaseMediaGenerator, VideoGenerationRequest, check_size


class VideoGenerator(BaseMediaGenerator):
    """Generate one video per text prompt.

    Args:
        duration:
            The length of the videos in seconds. Backends with fixed lengths generate the closest longer one and cut it.

        size:
            The width and height of the videos in pixels.

        fps:
            The number of frames per second, or `None` for the model's default.

        style:
            A description of the visual style, such as `'claymation'`.

        negative_prompt:
            A description of what the videos must not show.

        output_format:
            `'object'` returns `Video` objects, `'bytes'` returns MP4 bytes, and `'path'` writes MP4 files and returns
            their paths. `'bytes'` and `'path'` need the `video` extra for videos that are not already encoded.

        output_dir:
            The directory for `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of prompts submitted to the backend at once.

        on_error:
            `'raise'` raises when a prompt fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

    Examples:
        >>> from skai.generation import VideoGenerator
        >>> generator = VideoGenerator(duration=2, size=(320, 240), fps=8, backend='dummy').fit()
        >>> video = generator.predict(['a boat at sea'])[0]
        >>> video.duration, video.frames.shape[1:3]
        (2.0, (240, 320))
    """

    task: ClassVar[str] = 'generate_video'
    input_kind: ClassVar[Literal['text']] = 'text'
    output_kind: ClassVar[Literal['video']] = 'video'
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'duration': [Interval(Real, 0, None, closed='neither')],
        'size': [tuple],
        'fps': [Interval(Integral, 1, None, closed='left'), None],
        'style': [str, None],
        'negative_prompt': [str, None],
    }

    def __init__(
        self,
        *,
        duration: float = 4.0,
        size: tuple[int, int] = (1280, 720),
        fps: int | None = None,
        style: str | None = None,
        negative_prompt: str | None = None,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 1,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.duration = duration
        self.size = size
        self.fps = fps
        self.style = style
        self.negative_prompt = negative_prompt
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Check the size."""
        check_size(self.size, type(self).__name__)

    def _build_request(self, sample: str, seed: int) -> VideoGenerationRequest:
        """Build the video generation request of one prompt."""
        return VideoGenerationRequest(
            sample,
            float(self.duration),
            check_size(self.size, type(self).__name__),
            self.fps,
            self.style,
            self.negative_prompt,
            seed,
        )
