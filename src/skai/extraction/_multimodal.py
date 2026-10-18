"""Extract structured records from multimodal samples."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral
from typing import ClassVar, Literal

from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval

from skai.core import Backend, BaseExtractor


class MultimodalExtractor(BaseExtractor):
    """Extract a record of declared fields from every sample, learning from reference records in `fit`.

    A sample is a dict, or a DataFrame row, of named texts, images, audio and videos, all sent in one request.

    A field is declared as a description, as a dict of subfields for a nested record, or as a one-element list of
    either for a list of values or records. `predict` returns one dict per sample whose keys are exactly the field
    names, with `None` for absent fields, and `transform` returns one column per top-level field.

    Args:
        fields:
            The fields to extract, as a dict that maps every name to its declaration.

        instructions:
            Extra guidance on how to extract the fields.

        n_frames:
            The number of frames sampled from every video.

        k_shot:
            How many training samples are passed to the backend as examples with their records. `None` passes up to
            `max_examples`, `0` passes none, an integer selects that many at random, and an array of indices selects
            those samples.

        max_examples:
            The maximum number of examples when `k_shot` is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of samples submitted to the backend at once.

        on_error:
            `'raise'` raises when a sample fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the selection of examples and the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        examples_ (tuple):
            The pairs of a sample and its reference record passed to the backend.

    Examples:
        >>> from skai.extraction import MultimodalExtractor
        >>> from PIL import Image
        >>> extractor = MultimodalExtractor(fields={'product': 'product name'}, backend='dummy').fit()
        >>> list(extractor.predict([{'photo': Image.new('RGB', (8, 8)), 'notes': 'SkaiBox'}])[0])
        ['product']
    """

    input_kind: ClassVar[Literal['multimodal']] = 'multimodal'
    _parameter_constraints: ClassVar[dict] = {
        **BaseExtractor._parameter_constraints,
        'n_frames': [Interval(Integral, 1, None, closed='left')],
    }

    def __init__(
        self,
        *,
        fields: dict | None = None,
        instructions: str | None = None,
        n_frames: int = 8,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 8,
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.fields = fields
        self.instructions = instructions
        self.n_frames = n_frames
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state
