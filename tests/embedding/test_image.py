"""Test the image embedder."""

import numpy as np
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

from skai.embedding import ImageEmbedder, TextEmbedder

IMAGES = [Image.new('RGB', (8, 8), (value, 0, 0)) for value in range(0, 250, 50)]


def test_fit_transform_returns_fixed_dimensions():
    """Return a float array of shape `(n_images, dimensions)`."""
    assert ImageEmbedder(dimensions=32, backend='dummy').fit_transform(IMAGES).shape == (5, 32)


def test_embeddings_cluster_images():
    """Assign every image to one of two clusters."""
    pipeline = make_pipeline(ImageEmbedder(backend='dummy'), KMeans(n_clusters=2, n_init=2, random_state=0))
    labels = pipeline.fit_predict(IMAGES)
    assert set(labels) <= {0, 1}


def test_shared_space_has_equal_dimensions():
    """Return vectors of equal length from text and image embedders on the same backend and model."""
    text = TextEmbedder(normalize=True, backend='dummy:joint').fit().transform(['a red square'])
    image = ImageEmbedder(normalize=True, backend='dummy:joint').fit().transform(IMAGES[:1])
    assert text.shape == image.shape
    assert -1 <= float(text[0] @ image[0]) <= 1
    np.testing.assert_allclose(np.linalg.norm(image, axis=1), 1)
