"""Extract structured fields from texts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils.validation import check_is_fitted

from skai.core import Backend, BaseSupervisedGenerator, ExtractionRequest, to_object_array


class TextExtractor(BaseSupervisedGenerator):
    """Extract named fields from every text, as records or as feature columns.

    `predict` returns one dict per text, whose keys are exactly the field names, with `None` for a field the text does
    not contain. `transform` returns the same values as a two-dimensional array in the order of `fields`, with the field
    names as feature names, so the fields can feed a `ColumnTransformer` or a DataFrame. `fit(X, y)` with reference
    records as `y` passes `k_shot` text and record pairs to the backend as examples.

    Args:
        fields:
            The fields to extract, mapping every field name to a description. It is required.

        k_shot:
            How many reference pairs are passed to the backend as examples. `None` passes all of them up to
            `max_examples`, `0` passes none, an integer selects that many at random, and an array of indices selects
            those pairs.

        max_examples:
            The maximum number of examples when `k_shot` is `None`.

        use_y:
            `'auto'` learns from `y` in `fit` but not in `fit_transform`, which pipelines use for intermediate steps.
            `True` always learns from `y`, and `False` never does.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of texts submitted to the backend at once.

        on_error:
            `'raise'` raises when a text fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the selection of examples and the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        examples_ (tuple):
            The text and reference record pairs passed to the backend.

    Examples:
        >>> from skai.text import TextExtractor
        >>> extractor = TextExtractor(fields={'name': 'person name', 'city': 'city'}, backend='dummy').fit()
        >>> sorted(extractor.predict(['The name is Ada.'])[0])
        ['city', 'name']
        >>> extractor.get_feature_names_out().tolist()
        ['name', 'city']
    """

    task: ClassVar[str] = 'extract_fields'
    input_kind: ClassVar[Literal['text']] = 'text'
    _parameter_constraints: ClassVar[dict] = {
        **BaseSupervisedGenerator._parameter_constraints,
        'fields': [dict],
    }

    def __init__(
        self,
        *,
        fields: dict[str, str] | None = None,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 4,
        use_y: Literal['auto'] | bool = 'auto',
        backend: str | Backend | None = None,
        batch_size: int = 16,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.fields = fields
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _build_request(self, sample: str, seed: int) -> ExtractionRequest:
        """Build the extraction request of one text."""
        assert self.fields is not None
        return ExtractionRequest(sample, tuple(self.fields.items()), self.examples_)

    def _finalize(self, results: list[Any]) -> NDArray:
        """Return one record per text, with every field and `None` for missing ones."""
        assert self.fields is not None
        names = list(self.fields)
        return to_object_array(
            None if record is None else {name: record.get(name) for name in names} for record in results
        )

    def transform(self, X: ArrayLike) -> NDArray:
        """Return the extracted fields as columns.

        Args:
            X:
                The texts.

        Returns:
            An object array of shape `(n_texts, n_fields)`. A failed text has a row of `None`.
        """
        names = list(self.get_feature_names_out())
        records = self.predict(X)
        columns = np.empty((len(records), len(names)), dtype=object)
        for row, record in enumerate(records):
            columns[row] = [None if record is None else record[name] for name in names]
        return columns

    def get_feature_names_out(self, input_features: ArrayLike | None = None) -> NDArray[np.object_]:
        """Return the field names, in the order of `fields`.

        Args:
            input_features:
                Ignored.

        Returns:
            The field names.
        """
        check_is_fitted(self, 'backend_')
        assert self.fields is not None
        return np.asarray(list(self.fields), dtype=object)
