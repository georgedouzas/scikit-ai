"""Classify audio with zero-shot or few-shot prediction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from numpy.typing import ArrayLike

from skai.core import Backend, BaseClassifier


class AudioClassifier(BaseClassifier):
    """Assign labels to audio, such as sound events, music genres or emotions, zero-shot or few-shot.

    With `labels` and no targets, the classifier predicts zero-shot. With targets, it learns the classes from them and
    passes training samples to the backend as examples, one per class by default.

    Args:
        labels:
            The candidate labels, as a list, or as a dict that maps every label to a description.

        instructions:
            Extra guidance on how to assign the labels.

        k_shot:
            How many training samples are passed to the backend as examples. `None` passes one per class, `0` passes
            none, an integer selects that many at random, and an array of indices selects those samples.

        adaptation:
            `'examples'` passes examples to the backend at prediction. `'finetune'` trains a model on all the training
            data in `fit`, on backends that support fine-tuning, and sends later requests to it.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of samples submitted to the backend at once.

        on_error:
            `'raise'` raises when a sample fails, and `'ignore'` returns a missing value for it and warns.

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
        >>> from skai.classification import AudioClassifier
        >>> classifier = AudioClassifier(labels=['dog bark', 'siren'], backend='dummy').fit()
        >>> classifier.predict_proba([(np.zeros(16000), 16000), (np.ones(8000), 16000)]).shape
        (2, 2)
    """

    task: ClassVar[str] = 'classify_audio'
    input_kind: ClassVar[Literal['audio']] = 'audio'

    def __init__(
        self,
        *,
        labels: list | dict | None = None,
        instructions: str | None = None,
        k_shot: int | ArrayLike | None = None,
        adaptation: Literal['examples', 'finetune'] = 'examples',
        backend: str | Backend | None = None,
        batch_size: int = 8,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.labels = labels
        self.instructions = instructions
        self.k_shot = k_shot
        self.adaptation = adaptation
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state
