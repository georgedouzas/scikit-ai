"""Classify images with zero-shot or few-shot prediction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from numpy.typing import ArrayLike

from skai.core import Backend, BaseClassifier


class ImageClassifier(BaseClassifier):
    """Assign labels to images with zero-shot or few-shot prediction.

    With `labels` and no targets, the classifier predicts zero-shot. With targets, it learns the classes from them and
    passes `k_shot` of the training images to the backend as labeled examples.

    Args:
        labels:
            The candidate labels, as a list, or as a dict that maps every label to a description.

        instructions:
            Extra guidance on how to assign the labels.

        k_shot:
            How many training images are passed to the backend as examples. `None` or `0` passes none, an integer
            selects that many at random, and an array of indices selects those images.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of images submitted to the backend at once.

        on_error:
            `'raise'` raises when an image fails, and `'ignore'` returns a missing value for it and warns.

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

    Examples:
        >>> from PIL import Image
        >>> from skai.image import ImageClassifier
        >>> classifier = ImageClassifier(labels=['cat', 'dog'], backend='dummy').fit()
        >>> classifier.predict_proba([Image.new('RGB', (32, 32))]).shape
        (1, 2)
    """

    task: ClassVar[str] = 'classify_image'
    input_kind: ClassVar[Literal['image']] = 'image'

    def __init__(
        self,
        *,
        labels: list | dict | None = None,
        instructions: str | None = None,
        k_shot: int | ArrayLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 8,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.labels = labels
        self.instructions = instructions
        self.k_shot = k_shot
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state
