"""Embed videos as vectors."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import NDArray
from sklearn.utils._param_validation import Interval

from skai.core import Backend, BaseEmbedder, EmbeddingRequest, MultiRequest, VideoFrames


class VideoEmbedder(BaseEmbedder):
    """Map every video to the mean embedding of frames sampled evenly over it.

    The frames are embedded with the image embedding model of the backend, so with a joint text and image model the
    vectors are comparable to those of a `TextEmbedder`, and texts can retrieve videos.

    Args:
        n_frames:
            The number of frames sampled from every video.

        dimensions:
            The number of dimensions of the vectors. Backends that can shorten embeddings are asked to, and other
            embeddings are projected with PCA learned in `fit`, or cut when `fit` gets no videos. The full embedding is
            kept when it is `None`.

        center:
            Whether to subtract the mean embedding of the training videos.

        normalize:
            Whether to scale every vector to unit length.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of videos submitted to the backend at once.

        on_error:
            `'raise'` raises when a video fails, and `'ignore'` returns a row of NaN for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        n_features_out_ (int):
            The number of dimensions of the vectors.

        mean_ (np.ndarray):
            The mean embedding of the training videos, before any projection.

        components_ (np.ndarray | None):
            The PCA projection to `dimensions`, or `None` when no projection is learned.

    Examples:
        >>> import numpy as np
        >>> from skai.core import Video
        >>> from skai.embedding import VideoEmbedder
        >>> videos = [Video(np.full((8, 8, 8, 3), value, dtype=np.uint8), fps=4) for value in (0, 100, 200)]
        >>> VideoEmbedder(n_frames=4, backend='dummy').fit_transform(videos).shape
        (3, 8)
    """

    operations: ClassVar[dict[str, str | None]] = {'embed': 'embed_image'}
    input_kind: ClassVar[Literal['video']] = 'video'
    _parameter_constraints: ClassVar[dict] = {
        **BaseEmbedder._parameter_constraints,
        'n_frames': [Interval(Integral, 1, None, closed='left')],
    }

    def __init__(
        self,
        *,
        n_frames: int = 8,
        dimensions: int | None = None,
        center: bool = True,
        normalize: bool = False,
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.n_frames = n_frames
        self.dimensions = dimensions
        self.center = center
        self.normalize = normalize
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _build_request(self, sample: VideoFrames, seed: int) -> MultiRequest:
        """Build one image embedding request per sampled frame."""
        assert self.backend_ is not None
        native = self.dimensions if self.backend_.shortens_embeddings else None
        return MultiRequest(EmbeddingRequest(frame, native) for frame in sample.frames)

    def _complete(self, result: Any, sample: VideoFrames) -> NDArray[np.float64]:
        """Average the embeddings of the frames."""
        return np.mean([np.asarray(embedding, dtype=float) for embedding in result], axis=0)
