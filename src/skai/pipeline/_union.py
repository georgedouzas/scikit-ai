"""Run named branches on the same samples and return multimodal samples."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.metaestimators import _BaseComposition
from sklearn.utils.validation import check_is_fitted

from skai.core import to_object_array, to_samples

PASSTHROUGH = 'passthrough'
TABLE_NDIM = 2


def _rows(output: Any, count: int) -> list[Any]:
    """Split the output of a branch into one value per sample, unwrapping a single column."""
    if hasattr(output, 'to_numpy') and not isinstance(output, np.ndarray):
        output = output.to_numpy()
    if isinstance(output, np.ndarray) and output.ndim == TABLE_NDIM and output.shape[1] == 1:
        output = output[:, 0]
    values = list(output)
    if len(values) != count:
        msg = f'A branch returned {len(values)} outputs for {count} samples.'
        raise ValueError(msg)
    return values


def _apply(branch: BaseEstimator, X: ArrayLike) -> Any:
    """Return the transform of a fitted branch, or its predictions when it cannot transform."""
    return branch.transform(X) if hasattr(branch, 'transform') else branch.predict(X)


class MediaUnion(TransformerMixin, _BaseComposition):
    """Run named branches on the same samples and return one multimodal sample, a dict keyed by branch name, per sample.

    It is modeled on `FeatureUnion`: every branch is an estimator, a pipeline or `'passthrough'`, branches are fitted
    through `fit_transform` in pipelines, and their parameters are routed as `<branch>__<parameter>`. A branch that
    cannot transform, such as one ending in a classifier, contributes its predictions. Instead of
    concatenating columns, it keeps the output of every branch under its name, so later steps such as
    `MultimodalToVideoGenerator` receive several inputs per sample.

    Args:
        transformer_list:
            The branches, as `(name, estimator)` pairs. An estimator can be `'passthrough'` to keep the input.

    Attributes:
        transformers_ (list):
            The fitted branches, as `(name, estimator)` pairs.

    Examples:
        >>> from skai.generation import TextToTextGenerator
        >>> from skai.pipeline import MediaUnion
        >>> summary = TextToTextGenerator(instructions='Summarize.', max_length=1, backend='dummy')
        >>> union = MediaUnion([('text', 'passthrough'), ('summary', summary)])
        >>> union.fit_transform(['Hello world'])[0]
        {'text': 'Hello world', 'summary': 'Hello'}
    """

    _required_parameters: ClassVar[list[str]] = ['transformer_list']

    def __init__(self, transformer_list: list[tuple[str, BaseEstimator | str]]) -> None:
        self.transformer_list = transformer_list

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Return the parameters of the union and, with `deep`, of its branches.

        Args:
            deep:
                Whether to include the parameters of the branches as `<branch>__<parameter>`.

        Returns:
            The parameters.
        """
        return self._get_params('transformer_list', deep=deep)

    def set_params(self, **params: Any) -> Self:
        """Set the parameters of the union or of its branches.

        Args:
            **params:
                The parameters, with `<branch>` to replace a branch and `<branch>__<parameter>` for its parameters.

        Returns:
            The union.
        """
        self._set_params('transformer_list', **params)
        return self

    def _check(self) -> None:
        """Check the branch names."""
        names = [name for name, _ in self.transformer_list]
        self._validate_names(names)

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Fit every branch on the samples.

        Args:
            X:
                The samples.

            y:
                The targets, passed to every branch.

        Returns:
            The fitted union.
        """
        self._check()
        self.transformers_ = [
            (name, branch if branch == PASSTHROUGH else clone(branch).fit(X, y))
            for name, branch in self.transformer_list
        ]
        return self

    def fit_transform(self, X: ArrayLike, y: ArrayLike | None = None, **fit_params: Any) -> NDArray[np.object_]:
        """Fit every branch through its `fit_transform` and return the multimodal samples.

        Args:
            X:
                The samples.

            y:
                The targets, passed to every branch.

            **fit_params:
                Ignored.

        Returns:
            One dict per sample, keyed by branch name.
        """
        self._check()
        samples = to_samples(X)
        self.transformers_ = []
        outputs = {}
        for name, branch in self.transformer_list:
            if branch == PASSTHROUGH:
                self.transformers_.append((name, branch))
                outputs[name] = samples
                continue
            fitted = clone(branch)
            output = fitted.fit_transform(X, y) if hasattr(fitted, 'transform') else fitted.fit(X, y).predict(X)
            outputs[name] = _rows(output, len(samples))
            self.transformers_.append((name, fitted))
        return self._join(outputs, len(samples))

    def transform(self, X: ArrayLike) -> NDArray[np.object_]:
        """Run every branch and return the multimodal samples.

        Args:
            X:
                The samples.

        Returns:
            One dict per sample, keyed by branch name.
        """
        check_is_fitted(self, 'transformers_')
        samples = to_samples(X)
        outputs = {
            name: samples if branch == PASSTHROUGH else _rows(_apply(branch, X), len(samples))
            for name, branch in self.transformers_
        }
        return self._join(outputs, len(samples))

    def predict(self, X: ArrayLike) -> NDArray[np.object_]:
        """Run every branch and return the multimodal samples, so the union can be the final step of a pipeline.

        Args:
            X:
                The samples.

        Returns:
            One dict per sample, keyed by branch name.
        """
        return self.transform(X)

    @staticmethod
    def _join(outputs: dict[str, list[Any]], count: int) -> NDArray[np.object_]:
        """Join the outputs of the branches into one dict per sample."""
        return to_object_array({name: values[index] for name, values in outputs.items()} for index in range(count))
