"""Test the text embedder."""

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
