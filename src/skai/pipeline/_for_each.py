"""Apply an estimator to every item of list-valued samples."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import Tags
from sklearn.utils.validation import check_is_fitted

from skai.core import InvalidSampleError, to_object_array, to_samples


def _flatten(X: ArrayLike) -> tuple[list[Any], list[int]]:
    """Return the items of all list-valued samples and the length of every sample."""
    items: list[Any] = []
    lengths: list[int] = []
    for index, sample in enumerate(to_samples(X)):
        if not isinstance(sample, list | tuple | np.ndarray):
            msg = f'Sample {index} must be a list of items, but got {type(sample).__name__}.'
            raise InvalidSampleError(msg)
        items.extend(sample)
        lengths.append(len(sample))
    return items, lengths


def _split(output: Any, lengths: list[int]) -> NDArray[np.object_]:
    """Split the outputs of all items back into one list per sample."""
    if hasattr(output, 'to_numpy') and not isinstance(output, np.ndarray):
        output = output.to_numpy()
    values = list(output)
    lists = []
    start = 0
    for length in lengths:
        lists.append(values[start : start + length])
        start += length
    return to_object_array(lists)


class ForEach(TransformerMixin, BaseEstimator):
    """Apply an estimator to every item of list-valued samples, such as the regions of a segmenter.

    The items of all samples are flattened and processed in one call of the estimator, so a backend receives them as
    one batch, and the outputs are split back into lists of the original lengths. `fit` fits the estimator on all the
    items with `y=None`.

    Args:
        estimator:
            The estimator or pipeline applied to every item.

    Attributes:
        estimator_ (BaseEstimator):
            The fitted estimator.

    Examples:
        >>> from skai.generation import TextToTextGenerator
        >>> from skai.pipeline import ForEach
        >>> generator = TextToTextGenerator(instructions='Shorten.', max_length=1, backend='dummy')
        >>> ForEach(generator).fit_transform([['One two', 'Three four'], []]).tolist()
        [['One', 'Three'], []]
    """

    def __init__(self, estimator: BaseEstimator) -> None:
        self.estimator = estimator

    def __sklearn_tags__(self) -> Tags:
        """Declare that the samples are lists."""
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        return tags

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit the estimator on the items of all samples.

        Args:
            X:
                The list-valued samples.

            y:
                Ignored, since items have no targets.

        Returns:
            The fitted transformer.
        """
        items, _ = _flatten(X)
        self.estimator_ = clone(self.estimator).fit(items or None)
        return self

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None, **fit_params: Any) -> NDArray[np.object_]:
        """Fit the estimator on the items of all samples and return their outputs, running every item once.

        Args:
            X:
                The list-valued samples.

            y:
                Ignored, since items have no targets.

            **fit_params:
                Ignored.

        Returns:
            One list of outputs per sample.
        """
        items, lengths = _flatten(X)
        self.estimator_ = clone(self.estimator)
        if not items:
            self.estimator_.fit(None)
            return to_object_array([] for _ in lengths)
        estimator = self.estimator_
        output = estimator.fit_transform(items) if hasattr(estimator, 'fit_transform') else None
        if output is None:
            output = estimator.fit(items).predict(items)
        return _split(output, lengths)

    def _apply(self, X: ArrayLike, method: str) -> NDArray[np.object_]:
        """Run a method of the estimator on the items of all samples."""
        check_is_fitted(self, 'estimator_')
        items, lengths = _flatten(X)
        if not items:
            return to_object_array([] for _ in lengths)
        return _split(getattr(self.estimator_, method)(items), lengths)

    def transform(self, X: ArrayLike) -> NDArray[np.object_]:
        """Transform the items of all samples.

        Args:
            X:
                The list-valued samples.

        Returns:
            One list of outputs per sample.
        """
        return self._apply(X, 'transform' if hasattr(self.estimator_, 'transform') else 'predict')

    def predict(self, X: ArrayLike) -> NDArray[np.object_]:
        """Predict on the items of all samples.

        Args:
            X:
                The list-valued samples.

        Returns:
            One list of predictions per sample, or of outputs when the estimator only transforms.
        """
        check_is_fitted(self, 'estimator_')
        return self._apply(X, 'predict' if hasattr(self.estimator_, 'predict') else 'transform')
