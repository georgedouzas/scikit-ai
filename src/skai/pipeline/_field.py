"""Pick fields of multimodal samples and regions."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import Tags

from skai.core import to_object_array, to_samples


class Field(TransformerMixin, BaseEstimator):
    """Return one field of every multimodal sample or region, or a dict of several fields.

    It is stateless, so it can sit anywhere in a pipeline, for example after an extractor's `predict` or inside
    `ForEach` to pick the part of every region that the next step needs.

    Args:
        name:
            The name of the field, or a list of names to return dicts of those fields.

    Examples:
        >>> from skai.pipeline import Field
        >>> Field('visual').fit_transform([{'visual': 'a boat', 'narration': 'At sea.'}]).tolist()
        ['a boat']
    """

    _parameter_constraints: ClassVar[dict] = {'name': [str, list]}

    def __init__(self, name: str | list[str]) -> None:
        self.name = name

    def __sklearn_tags__(self) -> Tags:
        """Declare that the transformer is stateless."""
        tags = super().__sklearn_tags__()
        tags.requires_fit = False
        tags.input_tags.two_d_array = False
        return tags

    def fit(self, X: ArrayLike | None = None, y: ArrayLike | None = None) -> Self:
        """Validate the parameters.

        Args:
            X:
                Ignored.

            y:
                Ignored.

        Returns:
            The transformer.
        """
        self._validate_params()
        return self

    def transform(self, X: ArrayLike) -> NDArray[np.object_]:
        """Pick the field of every sample.

        Args:
            X:
                Dicts, such as multimodal samples, extracted records or regions.

        Returns:
            The value of the field per sample, or a dict of the fields when `name` is a list.

        Raises:
            KeyError:
                If a sample lacks a field.
        """
        names = self.name if isinstance(self.name, list) else [self.name]
        values = []
        for index, sample in enumerate(to_samples(X, rows=True)):
            missing = [name for name in names if not isinstance(sample, dict) or name not in sample]
            if missing:
                msg = f'Sample {index} has no field `{missing[0]}`.'
                raise KeyError(msg)
            picked: Any = {name: sample[name] for name in names} if isinstance(self.name, list) else sample[self.name]
            values.append(picked)
        return to_object_array(values)
