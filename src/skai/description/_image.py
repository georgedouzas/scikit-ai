"""Describe images in natural language."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from numpy.typing import ArrayLike
from PIL import Image
from sklearn.utils._param_validation import StrOptions

from skai.core import Backend, BaseSupervisedGenerator, DescriptionRequest


class ImageDescriptor(BaseSupervisedGenerator):
    """Return one natural-language description per image, learning style from reference descriptions.

    `fit(X, y)` with reference descriptions as `y` passes `k_shot` image and description pairs to the backend as
    examples, so the descriptions follow their style, length and vocabulary.

    Args:
        detail:
            `'brief'` for one sentence or `'detailed'` for a paragraph.

        instructions:
            Extra guidance on what to describe, such as `'Focus on the clothes.'`.

        k_shot:
            How many reference pairs are passed to the backend as examples. `None` passes all of them up to
            `max_examples`, `0` passes none, an integer selects that many at random, and an array of indices selects
            those pairs.

        max_examples:
            The maximum number of examples when `k_shot` is `None`.

        use_y:
            `'auto'` learns from `y` in `fit` but not in `fit_transform`, which pipelines use for intermediate steps.
            `True` always learns from `y`, and `False` never does.

        adaptation:
            `'examples'` passes examples to the backend at prediction. `'finetune'` trains a model on all the training
            data in `fit`, on backends that support fine-tuning, and sends later requests to it.

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

        examples_ (tuple):
            The image and reference description pairs passed to the backend.

        model_ (str):
            The id of the fine-tuned model. It is set only with `adaptation='finetune'`.

    Examples:
        >>> from PIL import Image
        >>> from skai.description import ImageDescriptor
        >>> descriptor = ImageDescriptor(backend='dummy').fit()
        >>> descriptor.predict([Image.new('RGB', (64, 32))])[0]
        'A brief description of a 64x32 image.'
    """

    task: ClassVar[str] = 'describe_image'
    input_kind: ClassVar[Literal['image']] = 'image'
    _parameter_constraints: ClassVar[dict] = {
        **BaseSupervisedGenerator._parameter_constraints,
        'adaptation': [StrOptions({'examples', 'finetune'})],
        'detail': [StrOptions({'brief', 'detailed'})],
        'instructions': [str, None],
    }

    def __init__(
        self,
        *,
        detail: Literal['brief', 'detailed'] = 'brief',
        instructions: str | None = None,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 4,
        use_y: Literal['auto'] | bool = 'auto',
        adaptation: Literal['examples', 'finetune'] = 'examples',
        backend: str | Backend | None = None,
        batch_size: int = 8,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.detail = detail
        self.instructions = instructions
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.adaptation = adaptation
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _build_request(self, sample: Image.Image, seed: int) -> DescriptionRequest:
        """Build the description request of one image."""
        return DescriptionRequest(sample, self.detail, self.instructions, self.examples_)
