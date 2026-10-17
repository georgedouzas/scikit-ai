"""Describe images in natural language."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from PIL import Image
from sklearn.utils._param_validation import StrOptions

from skai.core import Backend, BaseGenerator, DescriptionRequest


class ImageDescriptor(BaseGenerator):
    """Return one natural-language description per image.

    Args:
        detail:
            `'brief'` for one sentence or `'detailed'` for a paragraph.

        instructions:
            Extra guidance on what to describe, such as `'Focus on the clothes.'`.

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
        >>> from skai.description import ImageDescriptor
        >>> descriptor = ImageDescriptor(backend='dummy').fit()
        >>> descriptor.predict([Image.new('RGB', (64, 32))])[0]
        'A brief description of a 64x32 image.'
    """

    task: ClassVar[str] = 'describe_image'
    input_kind: ClassVar[Literal['image']] = 'image'
    _parameter_constraints: ClassVar[dict] = {
        **BaseGenerator._parameter_constraints,
        'detail': [StrOptions({'brief', 'detailed'})],
        'instructions': [str, None],
    }

    def __init__(
        self,
        *,
        detail: Literal['brief', 'detailed'] = 'brief',
        instructions: str | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 8,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.detail = detail
        self.instructions = instructions
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _build_request(self, sample: Image.Image, seed: int) -> DescriptionRequest:
        """Build the description request of one image."""
        return DescriptionRequest(sample, self.detail, self.instructions)
