"""Generate images from text prompts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral
from os import PathLike
from typing import ClassVar, Literal

from sklearn.utils._param_validation import InvalidParameterError

from skai.core import Backend, BaseMediaGenerator, ImageGenerationRequest

SIZE_LENGTH = 2


class ImageGenerator(BaseMediaGenerator):
    """Generate one image per text prompt.

    Args:
        size:
            The width and height of the images in pixels.

        style:
            A description of the visual style, such as `'watercolor'`.

        negative_prompt:
            A description of what the images must not show, for backends that support it.

        output_format:
            `'object'` returns PIL images, `'bytes'` returns PNG bytes, and `'path'` writes PNG files and returns
            their paths.

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
        >>> from skai.generation import ImageGenerator
        >>> generator = ImageGenerator(size=(512, 512), backend='dummy').fit()
        >>> generator.predict(['a cat on a sofa'])[0].size
        (512, 512)
    """

    task: ClassVar[str] = 'generate_image'
    input_kind: ClassVar[Literal['text']] = 'text'
    output_kind: ClassVar[Literal['image']] = 'image'
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'size': [tuple],
        'style': [str, None],
        'negative_prompt': [str, None],
    }

    def __init__(
        self,
        *,
        size: tuple[int, int] = (1024, 1024),
        style: str | None = None,
        negative_prompt: str | None = None,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.size = size
        self.style = style
        self.negative_prompt = negative_prompt
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Check that `size` holds a positive width and height."""
        if len(self.size) != SIZE_LENGTH or not all(isinstance(side, Integral) and side > 0 for side in self.size):
            msg = (
                'The `size` parameter of ImageGenerator must be a (width, height) pair of positive integers. '
                f'Got {self.size}.'
            )
            raise InvalidParameterError(msg)

    def _build_request(self, sample: str, seed: int) -> ImageGenerationRequest:
        """Build the image generation request of one prompt."""
        width, height = self.size
        return ImageGenerationRequest(sample, (int(width), int(height)), self.style, self.negative_prompt, seed)
