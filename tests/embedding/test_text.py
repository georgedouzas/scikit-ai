"""Test the text embedder."""

import numpy as np
import pytest

from skai.backends import DummyBackend
from skai.embedding import TextEmbedder


def test_fit_transform_returns_fixed_dimensions():
    """Return a float array of shape `(n_texts, dimensions)`."""
    embeddings = TextEmbedder(dimensions=64, backend='dummy').fit_transform([f'text {index}' for index in range(10)])
    assert embeddings.shape == (10, 64)
    assert embeddings.dtype == float


def test_set_output_returns_named_columns():
    """Return a DataFrame with named columns when pandas output is set."""
    embedder = TextEmbedder(dimensions=2, backend='dummy').set_output(transform='pandas')
    frame = embedder.fit_transform(['a', 'b'])
    assert list(frame.columns) == ['textembedder0', 'textembedder1']


CORPUS = [f'sentence number {index}' for index in range(20)]


class LongEmbeddingBackend(DummyBackend):
    """Return 16-dimensional embeddings and ignore requests for fewer dimensions."""

    shortens_embeddings = False

    def embed_text(self, request):
        """Return a 16-dimensional deterministic vector whose first entry varies the most."""
        vector = np.random.default_rng(len(request.sample) * 7 + ord(request.sample[-1])).standard_normal(16)
        vector[0] *= 10
        return vector


def test_fit_centers_embeddings():
    """Return embeddings whose mean over the training texts is zero."""
    embeddings = TextEmbedder(backend='dummy').fit(CORPUS).transform(CORPUS)
    np.testing.assert_allclose(embeddings.mean(axis=0), 0, atol=1e-12)
    raw = TextEmbedder(center=False, backend='dummy').fit(CORPUS).transform(CORPUS)
    assert not np.allclose(raw.mean(axis=0), 0)


def test_fit_learns_pca_when_backend_cannot_shorten():
    """Project to the directions of largest variance instead of cutting the first entries."""
    embedder = TextEmbedder(dimensions=4, backend=LongEmbeddingBackend()).fit(CORPUS)
    assert embedder.components_.shape == (4, 16)
    projected = embedder.transform(CORPUS)
    assert projected.shape == (20, 4)
    variances = projected.var(axis=0)
    assert list(variances) == sorted(variances, reverse=True)
    assert abs(embedder.components_[0, 0]) > 0.9


def test_fit_uses_native_shortening():
    """Ask a backend that shortens embeddings for the dimensions and learn no projection."""
    embedder = TextEmbedder(dimensions=4, backend='dummy').fit(CORPUS)
    assert embedder.components_ is None
    assert embedder.transform(['x']).shape == (1, 4)


def test_fit_rejects_projection_with_too_few_texts():
    """Ask for at least `dimensions` texts to learn a projection."""
    with pytest.raises(ValueError, match='at least 4 samples'):
        TextEmbedder(dimensions=4, backend=LongEmbeddingBackend()).fit(CORPUS[:3])
