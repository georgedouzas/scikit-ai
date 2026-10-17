"""Embed texts as vectors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from skai.core import Backend, BaseEmbedder


class TextEmbedder(BaseEmbedder):
    """Map every text to a fixed-size vector.

    Args:
        dimensions:
            The number of dimensions of the vectors. Longer embeddings are shortened, and backends that can shorten
            embeddings are asked to. The full embedding is kept when it is `None`.

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

    Examples:
        >>> from skai.text import TextEmbedder
        >>> embedder = TextEmbedder(dimensions=4, backend='dummy')
        >>> embedder.fit_transform(['a text', 'another text']).shape
        (2, 4)
    """

    task: ClassVar[str] = 'embed_text'
    input_kind: ClassVar[Literal['text']] = 'text'

    def __init__(
        self,
        *,
        dimensions: int | None = None,
        normalize: bool = False,
        backend: str | Backend | None = None,
        batch_size: int = 64,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.dimensions = dimensions
        self.normalize = normalize
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state
