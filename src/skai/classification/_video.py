"""Classify videos with zero-shot or few-shot prediction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral
from typing import ClassVar, Literal

from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from skai.core import Backend, BaseClassifier


class VideoClassifier(BaseClassifier):
    """Assign labels to videos, zero-shot or few-shot, from frames sampled evenly over each video.

    With `labels` and no targets, the classifier predicts zero-shot. With targets, it learns the classes from them and
    passes training videos to the backend as examples, one per class by default. Backends that accept audio also
    receive the audio track.

    Args:
        labels:
            The candidate labels, as a list, or as a dict that maps every label to a description.

        instructions:
            Extra guidance on how to assign the labels.

        n_frames:
            The number of frames sampled from every video.

        k_shot:
            How many training videos are passed to the backend as examples. `None` passes one per class, `0` passes
            none, an integer selects that many at random, and an array of indices selects those videos.

        adaptation:
            `'examples'` passes examples to the backend at prediction. `'finetune'` trains a model on all the training
            data in `fit`, on backends that support fine-tuning, and sends later requests to it.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of videos submitted to the backend at once.

        on_error:
            `'raise'` raises when a video fails, and `'ignore'` returns a missing value for it and warns.

        random_state:
            Controls the selection of examples and the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        classes_ (np.ndarray):
            The class labels.

        multilabel_ (bool):
            Whether the targets are a multilabel indicator matrix.

        examples_ (tuple):
            The labeled examples passed to the backend.

        model_ (str):
            The id of the fine-tuned model. It is set only with `adaptation='finetune'`.

    Examples:
        >>> import numpy as np
        >>> from skai.classification import VideoClassifier
        >>> from skai.core import Video
        >>> video = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10)
        >>> classifier = VideoClassifier(labels=['cooking', 'sports'], n_frames=4, backend='dummy').fit()
        >>> classifier.predict([video, video]).shape
        (2,)
    """

    operations: ClassVar[dict[str, str | None]] = {'classify': 'classify_video'}
    input_kind: ClassVar[Literal['video']] = 'video'
    _parameter_constraints: ClassVar[dict] = {
        **BaseClassifier._parameter_constraints,
        'n_frames': [Interval(Integral, 1, None, closed='left')],
    }

    def __init__(
        self,
        *,
        labels: list | dict | None = None,
        instructions: str | None = None,
        n_frames: int = 8,
        k_shot: int | ArrayLike | None = None,
        adaptation: Literal['examples', 'finetune'] = 'examples',
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.labels = labels
        self.instructions = instructions
        self.n_frames = n_frames
        self.k_shot = k_shot
        self.adaptation = adaptation
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state
