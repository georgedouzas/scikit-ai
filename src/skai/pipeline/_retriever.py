"""Index the items of samples and find the nearest ones for queries."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.base import BaseEstimator, clone
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import Tags
from sklearn.utils._param_validation import HasMethods, StrOptions
from sklearn.utils.validation import check_is_fitted

from skai.core import to_object_array

VECTOR = 'vector'


def _records(output: Any) -> list[dict[str, Any]]:
    """Turn the output of the item pipeline into records with a vector and the index of their sample."""
    if hasattr(output, 'to_numpy') and not isinstance(output, np.ndarray):
        output = output.to_numpy()
    records = []
    for index, value in enumerate(output):
        if isinstance(value, list):
            for item in value:
                if not isinstance(item, dict) or VECTOR not in item:
                    msg = f'Every item of sample {index} must be a record with a `{VECTOR}` key.'
                    raise ValueError(msg)
                records.append({'sample': index, **item})
        else:
            records.append({'sample': index, VECTOR: value})
    return records


class Retriever(BaseEstimator):
    """Index the items of samples with an item pipeline and find the nearest items of queries.

    The item pipeline returns, per sample, either one vector, or a list of records with a `vector` key, such as the
    shots of a video with their times and embeddings. `fit` indexes every item with the index of its sample, and
    `kneighbors` embeds queries with the query pipeline, which is fitted without data, and returns the records of the
    nearest items. With a joint text and image model, texts find images, shots or videos.

    Args:
        items:
            The pipeline that turns every sample into a vector or a list of records with vectors.

        queries:
            The pipeline that turns every query into a vector.

        metric:
            The distance between vectors.

    Attributes:
        items_ (BaseEstimator):
            The fitted item pipeline.

        queries_ (BaseEstimator):
            The fitted query pipeline.

        records_ (list):
            The records of the indexed items, without their vectors.

        index_ (NearestNeighbors):
            The index of the item vectors.

    Examples:
        >>> from skai.embedding import TextEmbedder
        >>> from skai.pipeline import Retriever
        >>> retriever = Retriever(TextEmbedder(center=False, backend='dummy'), TextEmbedder(backend='dummy'))
        >>> retriever.fit(['a goal', 'a corner', 'a foul']).kneighbors(['a goal'], n_neighbors=1)[0][0]['sample']
        0
    """

    _parameter_constraints: ClassVar[dict] = {
        'items': [HasMethods(['fit', 'transform'])],
        'queries': [HasMethods(['fit', 'transform'])],
        'metric': [StrOptions({'cosine', 'euclidean'})],
    }

    def __init__(self, items: BaseEstimator, queries: BaseEstimator, *, metric: str = 'cosine') -> None:
        self.items = items
        self.queries = queries
        self.metric = metric

    def __sklearn_tags__(self) -> Tags:
        """Declare that samples are media, not a numeric matrix."""
        tags = super().__sklearn_tags__()
        tags.input_tags.two_d_array = False
        return tags

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> Self:
        """Index the items of the samples.

        Args:
            X:
                The samples, such as videos.

            y:
                Ignored.

        Returns:
            The fitted retriever.

        Raises:
            ValueError:
                If the samples have no items.
        """
        self._validate_params()
        self.items_ = clone(self.items)
        records = _records(self.items_.fit_transform(X))
        if not records:
            msg = 'The samples have no items to index.'
            raise ValueError(msg)
        vectors = np.stack([np.asarray(record.pop(VECTOR), dtype=float) for record in records])
        self.records_ = records
        self.index_ = NearestNeighbors(metric=self.metric).fit(vectors)
        self.queries_ = clone(self.queries).fit(None)
        return self

    def kneighbors(self, queries: ArrayLike, n_neighbors: int = 5) -> NDArray[np.object_]:
        """Find the nearest items of every query.

        Args:
            queries:
                The queries, such as texts.

            n_neighbors:
                The number of items per query, at most the number of indexed items.

        Returns:
            One list of item records per query, nearest first, each with its `sample` index and `distance`.
        """
        check_is_fitted(self, 'index_')
        vectors = np.asarray(self.queries_.transform(queries), dtype=float)
        count = min(n_neighbors, len(self.records_))
        distances, indices = self.index_.kneighbors(vectors, n_neighbors=count)
        return to_object_array(
            [{**self.records_[index], 'distance': float(distance)} for index, distance in zip(row, gaps, strict=True)]
            for row, gaps in zip(indices, distances, strict=True)
        )
