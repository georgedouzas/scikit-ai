"""Keep the best items of list-valued samples."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections.abc import Callable
from numbers import Integral, Real
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils import Tags
from sklearn.utils._param_validation import HasMethods, Interval
from sklearn.utils.validation import check_is_fitted

from skai.core import to_object_array

from ._for_each import _flatten


class TopK(TransformerMixin, BaseEstimator):
    """Keep the `k` items of every list-valued sample with the highest scores, or those above `threshold`.

    The kept items stay in their original order. A score comes from a callable on one item, or from the
    `predict_proba` column of `label` of a classifier, which `fit` fits zero-shot on the items. The items of all
    samples are scored in one call.

    Args:
        scorer:
            A callable that scores one item, or a classifier with `predict_proba`, such as a `VideoClassifier` with
            `labels`.

        label:
            The class whose probability is the score of a classifier.

        k:
            The number of items to keep per sample. All items above `threshold` are kept when it is `None`.

        threshold:
            The minimum score of a kept item.

    Attributes:
        scorer_ (Callable | BaseEstimator):
            The callable, or the fitted classifier.

    Examples:
        >>> from skai.pipeline import TopK
        >>> TopK(len, k=2).fit_transform([['a', 'abc', 'ab', 'abcd']]).tolist()
        [['abc', 'abcd']]
    """

    _parameter_constraints: ClassVar[dict] = {
        'scorer': [callable, HasMethods(['predict_proba'])],
        'label': [None, str, Integral],
        'k': [Interval(Integral, 1, None, closed='left'), None],
        'threshold': [Real, None],
    }

    def __init__(
        self,
        scorer: Callable[[Any], float] | BaseEstimator,
        *,
        label: str | int | None = None,
        k: int | None = None,
        threshold: float | None = None,
    ) -> None:
        self.scorer = scorer
        self.label = label
        self.k = k
        self.threshold = threshold

    def __sklearn_tags__(self) -> Tags:
        """Declare that the samples are lists."""
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        return tags

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Fit a classifier scorer zero-shot, without targets.

        Args:
            X:
                The list-valued samples.

            y:
                Ignored, since items have no targets.

        Returns:
            The fitted transformer.

        Raises:
            ValueError:
                If neither `k` nor `threshold` is given, or a classifier is given without `label`.
        """
        self._validate_params()
        if self.k is None and self.threshold is None:
            msg = 'TopK needs `k` or `threshold`.'
            raise ValueError(msg)
        if hasattr(self.scorer, 'predict_proba'):
            if self.label is None:
                msg = 'TopK needs `label` to score items with a classifier.'
                raise ValueError(msg)
            self.scorer_ = clone(self.scorer).fit()
        else:
            self.scorer_ = self.scorer
        return self

    def _scores(self, items: list[Any]) -> NDArray[np.float64]:
        """Score every item."""
        if not hasattr(self.scorer_, 'predict_proba'):
            return np.asarray([float(self.scorer_(item)) for item in items])
        column = list(self.scorer_.classes_).index(self.label)
        return np.nan_to_num(self.scorer_.predict_proba(items)[:, column], nan=-np.inf)

    def transform(self, X: ArrayLike) -> NDArray[np.object_]:
        """Keep the best items of every sample.

        Args:
            X:
                The list-valued samples.

        Returns:
            One list of kept items per sample, in their original order.
        """
        check_is_fitted(self, 'scorer_')
        items, lengths = _flatten(X)
        scores = self._scores(items) if items else np.zeros(0)
        kept = []
        start = 0
        for length in lengths:
            indices = np.arange(start, start + length)
            if self.threshold is not None:
                indices = indices[scores[indices] > self.threshold]
            if self.k is not None:
                indices = np.sort(indices[np.argsort(-scores[indices], kind='stable')[: self.k]])
            kept.append([items[index] for index in indices])
            start += length
        return to_object_array(kept)
