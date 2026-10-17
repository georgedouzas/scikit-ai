"""Embed images as vectors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from skai.core import Backend, BaseEmbedder


class ImageEmbedder(BaseEmbedder):
    """Map every image to a fixed-size vector, learning centering and a projection from the images in `fit`.

    With the same joint text and image model as a `TextEmbedder`, for example `'hf:clip-ViT-B-32'`, the vectors of
    both embedders are comparable, so texts can retrieve images.

    Args:
        dimensions:
            The number of dimensions of the vectors. Backends that can shorten embeddings are asked to, and other
            embeddings are projected with PCA learned in `fit`, or cut when `fit` gets no images. The full embedding is
            kept when it is `None`.

        center:
            Whether to subtract the mean embedding of the training images.

        normalize:
            Whether to scale every vector to unit length.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of images submitted to the backend at once.

        on_error:
            `'raise'` raises when an image fails, and `'ignore'` returns a row of NaN for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        n_features_out_ (int):
            The number of dimensions of the vectors.

        mean_ (np.ndarray):
            The mean embedding of the training images, before any projection.

        components_ (np.ndarray | None):
            The PCA projection to `dimensions`, or `None` when no projection is learned.

    Examples:
        >>> from PIL import Image
        >>> from skai.embedding import ImageEmbedder
        >>> images = [Image.new('RGB', (8, 8), color) for color in ('red', 'green', 'blue')]
        >>> ImageEmbedder(dimensions=4, backend='dummy').fit_transform(images).shape
        (3, 4)
    """

    task: ClassVar[str] = 'embed_image'
    input_kind: ClassVar[Literal['image']] = 'image'

    def __init__(
        self,
        *,
        dimensions: int | None = None,
        center: bool = True,
        normalize: bool = False,
        backend: str | Backend | None = None,
        batch_size: int = 16,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.center = center
        self.normalize = normalize
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state
