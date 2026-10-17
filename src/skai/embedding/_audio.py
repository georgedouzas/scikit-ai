"""Embed audio as vectors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from skai.core import Backend, BaseEmbedder


class AudioEmbedder(BaseEmbedder):
    """Map every audio sample to a fixed-size vector, learning centering and a projection from the audio in `fit`.

    Args:
        dimensions:
            The number of dimensions of the vectors. Backends that can shorten embeddings are asked to, and other
            embeddings are projected with PCA learned in `fit`, or cut when `fit` gets no audio. The full embedding is
            kept when it is `None`.

        center:
            Whether to subtract the mean embedding of the training audio.

        normalize:
            Whether to scale every vector to unit length.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of audio samples submitted to the backend at once.

        on_error:
            `'raise'` raises when a sample fails, and `'ignore'` returns a row of NaN for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        n_features_out_ (int):
            The number of dimensions of the vectors.

        mean_ (np.ndarray):
            The mean embedding of the training audio, before any projection.

        components_ (np.ndarray | None):
            The PCA projection to `dimensions`, or `None` when no projection is learned.

    Examples:
        >>> import numpy as np
        >>> from skai.embedding import AudioEmbedder
        >>> audio = [(np.full(1600, value), 16000) for value in (0.1, 0.2, 0.3)]
        >>> AudioEmbedder(dimensions=4, backend='dummy').fit_transform(audio).shape
        (3, 4)
    """

    task: ClassVar[str] = 'embed_audio'
    input_kind: ClassVar[Literal['audio']] = 'audio'

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
