"""Generate images from texts, images and videos."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral, Real
from os import PathLike
from pathlib import PurePath
from typing import Any, ClassVar, Literal, TypeAlias

import numpy as np
from PIL import Image
from sklearn.utils._param_validation import Interval, InvalidParameterError

from skai.core import (
    Backend,
    BaseMediaGenerator,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageOperationRequest,
    TextGenerationRequest,
    ThumbnailRequest,
    UnsupportedBackendError,
    Video,
    check_size,
    collage,
    draw_seeds,
    frame_at,
    list_backends,
    read_image,
    read_video,
    sample_frames,
)

STYLE_INSTRUCTIONS = (
    'Describe only the visual style in one sentence, such as medium, palette and lighting, not the content.'
)
ImageSource: TypeAlias = str | bytes | Image.Image | np.ndarray | PurePath


class TextToImageGenerator(BaseMediaGenerator):
    """Create one image per prompt, in a style given or learned from reference images.

    `fit` learns `style_`, a description of the shared style of `style_images` written by the backend's image
    description, and applies it to every prompt unless `style` is given. `reference_images` show a subject that every
    image keeps consistent, on backends that take reference images, and with `reference_prompt` instead, `fit`
    generates one reference image of the subject and sends it with every request.

    Args:
        operation:
            `'generate'`, the only operation.

        size:
            The width and height of every image in pixels.

        style:
            A description of the visual style. It takes precedence over the learned style.

        style_images:
            Images whose shared style is learned in `fit`.

        negative_prompt:
            A description of what the images must not show.

        reference_images:
            Images of a subject that every image must show consistently.

        reference_prompt:
            A description of the subject, from which `fit` generates the reference image when `reference_images` is
            not given.

        output_format:
            `'object'` returns PIL images, `'bytes'` PNG bytes, and `'path'` the paths of new PNG files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

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
            The style learned from `style_images`.

        reference_images_ (tuple):
            The read or generated reference images.

    Examples:
        >>> from skai.generation import TextToImageGenerator
        >>> generator = TextToImageGenerator(size=(64, 48), backend='dummy').fit()
        >>> generator.predict(['a cat'])[0].size
        (64, 48)
    """

    input_kind: ClassVar[Literal['text']] = 'text'
    output_kind: ClassVar[Literal['image']] = 'image'
    operations: ClassVar[dict[str, str | None]] = {'generate': 'generate_image'}
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'size': [tuple],
        'style': [str, None],
        'style_images': [list, None],
        'negative_prompt': [str, None],
        'reference_images': [list, None],
        'reference_prompt': [str, None],
    }

    def __init__(
        self,
        *,
        operation: Literal['generate'] = 'generate',
        size: tuple[int, int] = (1024, 1024),
        style: str | None = None,
        style_images: list | None = None,
        negative_prompt: str | None = None,
        reference_images: list | None = None,
        reference_prompt: str | None = None,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.size = size
        self.style = style
        self.style_images = style_images
        self.negative_prompt = negative_prompt
        self.reference_images = reference_images
        self.reference_prompt = reference_prompt
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Check the size, read the reference images and learn the style of the style images."""
        assert self.backend_ is not None
        check_size(self.size, type(self).__name__)
        references = self.reference_images or []
        self.reference_images_ = tuple(read_image(image, index) for index, image in enumerate(references))
        if (self.reference_images_ or self.reference_prompt) and not self.backend_.supports_reference_images:
            backends = list_backends(self.task_, lambda backend_class: backend_class.supports_reference_images)
            msg = f'{type(self.backend_).__name__} cannot take reference images. Backends that can: {backends}.'
            raise UnsupportedBackendError(msg)
        self.style_ = self._learn_style()
        if not self.reference_images_ and self.reference_prompt:
            self.reference_images_ = (self._generate_reference(),)

    def _generate_reference(self) -> Image.Image:
        """Generate the reference image of the subject described by `reference_prompt`."""
        assert self.backend_ is not None
        assert self.reference_prompt is not None
        size = check_size(self.size, type(self).__name__)
        (seed,) = draw_seeds(self.random_state, 1)
        request = ImageGenerationRequest(self.reference_prompt, size, self.style or self.style_, None, seed)
        (reference,) = self.backend_.run('generate_image', [request])
        if isinstance(reference, Exception):
            raise reference
        return reference

    def _learn_style(self) -> str | None:
        """Describe the shared visual style of the style images with the backend."""
        assert self.backend_ is not None
        if not self.style_images:
            return None
        if not type(self.backend_).supports('generate_text', 'image'):
            backends = list_backends('generate_text', input_kind='image')
            msg = f'{type(self.backend_).__name__} cannot describe images. Backends that can: {backends}.'
            raise UnsupportedBackendError(msg)
        requests = [
            TextGenerationRequest(read_image(image, index), STYLE_INSTRUCTIONS)
            for index, image in enumerate(self.style_images)
        ]
        descriptions = self.backend_.run('generate_text', requests)
        for description in descriptions:
            if isinstance(description, Exception):
                raise description
        return ' '.join(str(description).strip() for description in descriptions)

    def _build_request(self, sample: str, seed: int) -> ImageGenerationRequest:
        """Build the image generation request of one prompt."""
        style = self.style or self.style_
        size = check_size(self.size, type(self).__name__)
        return ImageGenerationRequest(sample, size, style, self.negative_prompt, seed, self.reference_images_)


class ImageToImageGenerator(BaseMediaGenerator):
    """Return one image per image: edited, upscaled, without its background, or a collage of a list of images.

    Args:
        operation:
            `'edit'` follows `instructions` inside an optional `mask`, `'upscale'` enlarges by `scale`,
            `'remove_background'` returns an RGBA image with a transparent background, and `'collage'` lays a list of
            images out in a grid, locally.

        instructions:
            The edit to apply. Required with `operation='edit'`.

        mask:
            The region to edit, as a grayscale image where white marks the region. It is resized to every image.

        scale:
            The factor of upscaling.

        columns:
            The number of columns of a collage, chosen from the number of images when it is `None`.

        tile:
            The side of every tile of a collage in pixels.

        output_format:
            `'object'` returns PIL images, `'bytes'` PNG bytes, and `'path'` the paths of new PNG files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`. Collages need no backend.

        batch_size:
            The number of images submitted to the backend at once.

        on_error:
            `'raise'` raises when an image fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend | None):
            The resolved backend, or `None` for collages.

        mask_ (PIL.Image.Image | None):
            The read mask.

    Examples:
        >>> from PIL import Image
        >>> from skai.generation import ImageToImageGenerator
        >>> generator = ImageToImageGenerator(operation='upscale', scale=2, backend='dummy').fit()
        >>> generator.predict([Image.new('RGB', (64, 48))])[0].size
        (128, 96)
    """

    input_kind: ClassVar[Literal['image']] = 'image'
    output_kind: ClassVar[Literal['image']] = 'image'
    operations: ClassVar[dict[str, str | None]] = {
        'edit': 'edit_image',
        'upscale': 'upscale_image',
        'remove_background': 'remove_background',
        'collage': None,
    }
    list_operations: ClassVar[frozenset[str]] = frozenset({'collage'})
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'instructions': [str, None],
        'mask': [str, bytes, Image.Image, np.ndarray, PurePath, None],
        'scale': [Interval(Integral, 1, None, closed='left')],
        'columns': [Interval(Integral, 1, None, closed='left'), None],
        'tile': [Interval(Integral, 1, None, closed='left')],
    }

    def __init__(
        self,
        *,
        operation: Literal['edit', 'upscale', 'remove_background', 'collage'] = 'edit',
        instructions: str | None = None,
        mask: ImageSource | None = None,
        scale: int = 2,
        columns: int | None = None,
        tile: int = 256,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.instructions = instructions
        self.mask = mask
        self.scale = scale
        self.columns = columns
        self.tile = tile
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Check the instructions of edits, and read the mask and check that the backend supports masks."""
        self.mask_: Image.Image | None = None
        if self.operation != 'edit':
            return
        assert self.backend_ is not None
        if not self.instructions:
            msg = "The 'instructions' parameter of ImageToImageGenerator is required with `operation='edit'`."
            raise InvalidParameterError(msg)
        if self.mask is None:
            return
        if not self.backend_.supports_masks:
            backends = list_backends(self.task_, lambda backend_class: backend_class.supports_masks)
            msg = f'{type(self.backend_).__name__} cannot edit inside a mask. Backends that can: {backends}.'
            raise UnsupportedBackendError(msg)
        self.mask_ = read_image(self.mask).convert('L')

    def _build_request(self, sample: Image.Image, seed: int) -> Any:
        """Build the request of one image for the operation."""
        if self.operation == 'edit':
            assert self.instructions is not None
            mask = None if self.mask_ is None else self.mask_.resize(sample.size)
            return ImageEditRequest(sample, self.instructions, mask, seed)
        return ImageOperationRequest(sample, self.scale, seed)

    def _run_local(self, sample: list[Image.Image] | Image.Image, seed: int) -> Image.Image:
        """Lay a list of images out as a collage."""
        images = sample if isinstance(sample, list) else [sample]
        return collage(images, self.columns, self.tile)


class MultimodalToImageGenerator(BaseMediaGenerator):
    """Edit the `image` of every multimodal sample with the `instructions` of the same sample, inside its `mask`.

    A sample is a dict, or a DataFrame row, with `image` and `instructions`, and an optional `mask`, a grayscale image
    where white marks the region to edit. It lets every image get its own edit, such as the translated text of a
    comic panel.

    Args:
        operation:
            `'edit'`, the only operation.

        output_format:
            `'object'` returns PIL images, `'bytes'` PNG bytes, and `'path'` the paths of new PNG files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of samples submitted to the backend at once.

        on_error:
            `'raise'` raises when a sample fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

    Examples:
        >>> from PIL import Image
        >>> from skai.generation import MultimodalToImageGenerator
        >>> sample = {'image': Image.new('RGB', (64, 48)), 'instructions': 'Add snow.'}
        >>> MultimodalToImageGenerator(backend='dummy').fit().predict([sample])[0].size
        (64, 48)
    """

    input_kind: ClassVar[Literal['multimodal']] = 'multimodal'
    output_kind: ClassVar[Literal['image']] = 'image'
    operations: ClassVar[dict[str, str | None]] = {'edit': 'edit_image'}
    required_keys: ClassVar[dict[str, tuple[str, ...]]] = {'edit': ('image', 'instructions')}

    def __init__(
        self,
        *,
        operation: Literal['edit'] = 'edit',
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _read_one(self, sample: Any, index: int) -> dict[str, Any]:
        """Read the image, the instructions and the mask of one sample."""
        parts = super()._read_one(sample, index)
        parts['image'] = read_image(parts['image'], index)
        if parts.get('mask') is not None:
            parts['mask'] = read_image(parts['mask'], index).convert('L').resize(parts['image'].size)
        return parts

    def _build_request(self, sample: dict[str, Any], seed: int) -> ImageEditRequest:
        """Build the edit request of one sample, checking that the backend supports masks."""
        assert self.backend_ is not None
        mask = sample.get('mask')
        if mask is not None and not self.backend_.supports_masks:
            backends = list_backends(self.task_, lambda backend_class: backend_class.supports_masks)
            msg = f'{type(self.backend_).__name__} cannot edit inside a mask. Backends that can: {backends}.'
            raise UnsupportedBackendError(msg)
        return ImageEditRequest(sample['image'], str(sample['instructions']), mask, seed)


class VideoToImageGenerator(BaseMediaGenerator):
    """Return one image per video: the thumbnail chosen by the backend, or the frame at a time.

    Args:
        operation:
            `'thumbnail'` asks the backend for the frame that best represents the video among `n_frames` sampled
            frames, and `'frame'` returns the frame at `at` seconds, locally.

        n_frames:
            The number of candidate frames of a thumbnail.

        at:
            The time of the frame in seconds, clamped to the video.

        output_format:
            `'object'` returns PIL images, `'bytes'` PNG bytes, and `'path'` the paths of new PNG files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`. Frames need no backend.

        batch_size:
            The number of videos submitted to the backend at once.

        on_error:
            `'raise'` raises when a video fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend | None):
            The resolved backend, or `None` for frames.

    Examples:
        >>> import numpy as np
        >>> from skai.core import Video
        >>> from skai.generation import VideoToImageGenerator
        >>> video = Video(np.zeros((8, 48, 64, 3), dtype=np.uint8), fps=4)
        >>> VideoToImageGenerator(operation='frame', at=1.0).fit().predict([video])[0].size
        (64, 48)
    """

    input_kind: ClassVar[Literal['video']] = 'video'
    output_kind: ClassVar[Literal['image']] = 'image'
    operations: ClassVar[dict[str, str | None]] = {'thumbnail': 'pick_thumbnail', 'frame': None}
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'n_frames': [Interval(Integral, 1, None, closed='left')],
        'at': [Interval(Real, 0, None, closed='left')],
    }

    def __init__(
        self,
        *,
        operation: Literal['thumbnail', 'frame'] = 'thumbnail',
        n_frames: int = 8,
        at: float = 0.0,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.n_frames = n_frames
        self.at = at
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _read_one(self, sample: Any, index: int) -> Video:
        """Read the whole video."""
        return read_video(sample, index)

    def _build_request(self, sample: Video, seed: int) -> ThumbnailRequest:
        """Build the thumbnail request of one video from its sampled frames."""
        return ThumbnailRequest(sample_frames(sample, self.n_frames))

    def _complete(self, result: int, sample: Video) -> Image.Image:
        """Return the chosen frame."""
        return sample_frames(sample, self.n_frames).frames[result]

    def _run_local(self, sample: Video, seed: int) -> Image.Image:
        """Return the frame at the requested time."""
        return frame_at(sample, self.at)
