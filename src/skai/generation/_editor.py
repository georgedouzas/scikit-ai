"""Edit images with an instruction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os import PathLike
from pathlib import PurePath
from typing import ClassVar, Literal

import numpy as np
from PIL import Image

from skai.core import (
    Backend,
    BaseMediaGenerator,
    ImageEditRequest,
    UnsupportedBackendError,
    list_backends,
    read_image,
)


class ImageEditor(BaseMediaGenerator):
    """Apply one fixed instruction, such as `'make it night'`, to every image.

    Args:
        instruction:
            The edit to apply. It is required.

        mask:
            A grayscale image whose white region limits the edit, on backends that support masks. It is resized to
            every input image.

        output_format:
            `'object'` returns PIL images, `'bytes'` returns PNG bytes, and `'path'` writes PNG files and returns their
            paths.

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

        mask_ (PIL.Image.Image | None):
            The mask read from `mask`.

    Examples:
        >>> from PIL import Image
        >>> from skai.generation import ImageEditor
        >>> editor = ImageEditor(instruction='add snow', backend='dummy').fit()
        >>> [image.size for image in editor.predict([Image.new('RGB', (32, 24)), Image.new('RGB', (8, 8))])]
        [(32, 24), (8, 8)]
    """

    task: ClassVar[str] = 'edit_image'
    input_kind: ClassVar[Literal['image']] = 'image'
    output_kind: ClassVar[Literal['image']] = 'image'
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'instruction': [str],
        'mask': [str, bytes, Image.Image, np.ndarray, PurePath, None],
    }

    def __init__(
        self,
        *,
        instruction: str | None = None,
        mask: str | bytes | Image.Image | np.ndarray | PurePath | None = None,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.instruction = instruction
        self.mask = mask
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Read the mask and check that the backend supports masks."""
        if self.mask is None:
            self.mask_: Image.Image | None = None
            return
        if not self.backend_.supports_masks:
            backends = list_backends(self.task, lambda backend_class: backend_class.supports_masks)
            msg = f'{type(self.backend_).__name__} cannot edit inside a mask. Backends that can: {backends}.'
            raise UnsupportedBackendError(msg)
        self.mask_ = read_image(self.mask).convert('L')

    def _build_request(self, sample: Image.Image, seed: int) -> ImageEditRequest:
        """Build the edit request of one image."""
        assert self.instruction is not None
        mask = None if self.mask_ is None else self.mask_.resize(sample.size)
        return ImageEditRequest(sample, self.instruction, mask, seed)
