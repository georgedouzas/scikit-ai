"""Find objects and masks in images."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar, Literal

from PIL import Image

from skai.core import Backend, BaseSegmenter, Region, RegionRequest


class ImageSegmenter(BaseSegmenter):
    """Find the regions of every image that show `labels` or match `prompt`, as boxes or masks.

    Every region is a dict with `label`, `score`, `box` as `(left, top, right, bottom)` pixels, and `media`, the image
    cropped to the box. With `by='mask'` it also has `mask`, a grayscale image of the input size whose white pixels
    belong to the region, and `media` is transparent outside the mask.

    Args:
        by:
            `'object'` for boxes, or `'mask'` for masks.

        labels:
            The kinds of regions to find.

        prompt:
            A description of the regions to find, used when `labels` is not given.

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
        >>> from skai.segmentation import ImageSegmenter
        >>> regions = ImageSegmenter(labels=['person'], backend='dummy').fit().predict([Image.new('RGB', (64, 48))])[0]
        >>> regions[0]['label'], regions[0]['media'].size
        ('person', (32, 24))
    """

    input_kind: ClassVar[Literal['image']] = 'image'
    operations: ClassVar[dict[str, str | None]] = {'object': 'detect_objects', 'mask': 'segment_image'}
    _parameter_constraints: ClassVar[dict] = {
        **BaseSegmenter._parameter_constraints,
        'labels': [list, None],
        'prompt': [str, None],
    }

    def __init__(
        self,
        *,
        by: Literal['object', 'mask'] = 'object',
        labels: list[str] | None = None,
        prompt: str | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.by = by
        self.labels = labels
        self.prompt = prompt
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _build_request(self, sample: Image.Image, seed: int) -> RegionRequest:
        """Build the region request of one image."""
        return RegionRequest(sample, tuple(self.labels or ()), self.prompt)

    def _complete(self, result: list[Region], sample: Image.Image) -> list[dict[str, Any]]:
        """Turn the regions into records with the cropped image."""
        regions = []
        for region in result:
            record: dict[str, Any] = {'label': region.label, 'score': region.score, 'box': region.box}
            if self.by == 'mask' and region.mask is not None:
                cutout = sample.convert('RGBA')
                cutout.putalpha(region.mask.convert('L').resize(sample.size))
                record['mask'] = region.mask
                record['media'] = cutout.crop(region.box)
            else:
                record['media'] = sample.crop(region.box)
            regions.append(record)
        return regions
