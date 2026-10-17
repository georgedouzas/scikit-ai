"""Generate images from text prompts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os import PathLike
from typing import ClassVar, Literal

from skai.core import (
    Backend,
    BaseMediaGenerator,
    DescriptionRequest,
    ImageGenerationRequest,
    UnsupportedBackendError,
    check_size,
    list_backends,
    read_image,
)

STYLE_INSTRUCTIONS = (
    'Describe only the visual style in one sentence, such as medium, palette and lighting, not the content.'
)


class ImageGenerator(BaseMediaGenerator):
    """Generate one image per text prompt.

    Args:
        size:
            The width and height of the images in pixels.

        style:
            A description of the visual style, such as `'watercolor'`. It takes precedence over a learned style.

        style_images:
            Reference images whose shared visual style `fit` learns, using the image description task of the backend.

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

        style_ (str | None):
            The style learned from `style_images`, or `None` without them.

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
        'style_images': [list, None],
        'negative_prompt': [str, None],
    }

    def __init__(
        self,
        *,
        size: tuple[int, int] = (1024, 1024),
        style: str | None = None,
        style_images: list | None = None,
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
        self.style_images = style_images
        self.negative_prompt = negative_prompt
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Check the size and learn the style of the reference images."""
        check_size(self.size, type(self).__name__)
        self.style_ = self._learn_style()

    def _learn_style(self) -> str | None:
        """Describe the shared visual style of the reference images with the backend."""
        if not self.style_images:
            return None
        if 'describe_image' not in type(self.backend_).supported_tasks():
            backends = list_backends('describe_image')
            msg = f'{type(self.backend_).__name__} cannot describe images. Backends that can: {backends}.'
            raise UnsupportedBackendError(msg)
        requests = [
            DescriptionRequest(read_image(image, index), instructions=STYLE_INSTRUCTIONS)
            for index, image in enumerate(self.style_images)
        ]
        descriptions = self.backend_.run('describe_image', requests)
        for description in descriptions:
            if isinstance(description, Exception):
                raise description
        return ' '.join(str(description).strip() for description in descriptions)

    def _build_request(self, sample: str, seed: int) -> ImageGenerationRequest:
        """Build the image generation request of one prompt."""
        width, height = self.size
        style = self.style or self.style_
        return ImageGenerationRequest(sample, (int(width), int(height)), style, self.negative_prompt, seed)
