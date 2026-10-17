"""Classify texts with zero-shot or few-shot prediction."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from numpy.typing import ArrayLike

from skai.core import Backend, BaseClassifier


class TextClassifier(BaseClassifier):
    """Assign labels to texts with zero-shot or few-shot prediction.

    With `labels` and no targets, the classifier predicts zero-shot. With targets, it learns the classes from them and
    passes `k_shot` of the training texts to the backend as labeled examples.

    Args:
        labels:
            The candidate labels, as a list, or as a dict that maps every label to a description.

        instructions:
            Extra guidance on how to assign the labels.

        k_shot:
            How many training texts are passed to the backend as examples. `None` or `0` passes none, an integer
            selects that many at random, and an array of indices selects those texts.

        adaptation:
            `'examples'` passes examples to the backend at prediction. `'finetune'` trains a model on all the training
            data in `fit`, on backends that support fine-tuning, and sends later requests to it.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of texts submitted to the backend at once.

        on_error:
            `'raise'` raises when a text fails, and `'ignore'` returns a missing value for it and warns.

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
        >>> from skai.classification import TextClassifier
        >>> classifier = TextClassifier(labels=['positive', 'negative'], backend='dummy').fit()
        >>> classifier.predict(['I love it.', 'I hate it.']).shape
        (2,)
    """

    task: ClassVar[str] = 'classify_text'
    input_kind: ClassVar[Literal['text']] = 'text'

    def __init__(
        self,
        *,
        labels: list | dict | None = None,
        instructions: str | None = None,
        k_shot: int | ArrayLike | None = None,
        adaptation: Literal['examples', 'finetune'] = 'examples',
        backend: str | Backend | None = None,
        batch_size: int = 16,
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
