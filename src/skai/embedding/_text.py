"""Embed texts as vectors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from skai.core import Backend, BaseEmbedder


class TextEmbedder(BaseEmbedder):
    """Map every text to a fixed-size vector, learning centering and a projection from the texts in `fit`.

    `fit(X)` embeds the texts and learns their mean, which is subtracted from every embedding when `center=True`.
    When `dimensions` is set and the backend cannot shorten embeddings, it also learns a PCA projection to
    `dimensions`, which keeps the directions of largest variance instead of the first entries.

    Args:
        dimensions:
            The number of dimensions of the vectors. Backends that can shorten embeddings are asked to, and other
            embeddings are projected with PCA learned in `fit`, or cut when `fit` gets no texts. The full embedding is
            kept when it is `None`.

        center:
            Whether to subtract the mean embedding of the training texts.

        normalize:
            Whether to scale every vector to unit length.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of texts submitted to the backend at once.

        on_error:
            `'raise'` raises when a text fails, and `'ignore'` returns a row of NaN for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        n_features_out_ (int):
            The number of dimensions of the vectors.

        mean_ (np.ndarray):
            The mean embedding of the training texts, before any projection.

        components_ (np.ndarray | None):
            The PCA projection to `dimensions`, or `None` when no projection is learned.

    Examples:
        >>> from skai.embedding import TextEmbedder
        >>> embedder = TextEmbedder(dimensions=4, backend='dummy')
        >>> embedder.fit_transform(['a text', 'another text']).shape
        (2, 4)
    """

    operations: ClassVar[dict[str, str | None]] = {'embed': 'embed_text'}
    input_kind: ClassVar[Literal['text']] = 'text'

    def __init__(
        self,
        *,
        dimensions: int | None = None,
        center: bool = True,
        normalize: bool = False,
        backend: str | Backend | None = None,
        batch_size: int = 64,
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
