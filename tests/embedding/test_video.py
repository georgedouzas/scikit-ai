"""Test the video embedder."""

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from skai.backends import DummyBackend
from skai.core import Video
from skai.embedding import ImageEmbedder, VideoEmbedder

VIDEOS = [Video(np.full((8, 8, 8, 3), value, dtype=np.uint8), fps=4) for value in (0, 100, 200)]


def recording_backend():
    """Return a dummy backend that records its image embedding requests in `requests`."""
    requests = []

    class RecordingBackend(DummyBackend):
        """Record the requests, also across clones."""

        def embed_image(self, request):
            """Record the request and embed it like the dummy backend."""
            requests.append(request)
            return super().embed_image(request)

    backend = RecordingBackend()
    backend.requests = requests
    return backend


def test_video_vectors():
    """Return a float array with one row per video."""
    embeddings = VideoEmbedder(n_frames=4, backend='dummy').fit_transform(VIDEOS)
    assert embeddings.shape == (3, 8)
    assert embeddings.dtype == float


def test_embedding_is_the_mean_of_frame_embeddings():
    """Average the image embeddings of `n_frames` evenly sampled frames."""
    backend = recording_backend()
    embedder = VideoEmbedder(n_frames=4, center=False, backend=backend).fit()
    backend.requests.clear()
    raw = embedder.transform(VIDEOS[1:2])
    assert len(backend.requests) == 4
    frames = [request.sample for request in backend.requests]
    expected = ImageEmbedder(center=False, backend='dummy').fit().transform(frames).mean(axis=0)
    np.testing.assert_allclose(raw[0], expected)


def test_dimensions_center_and_normalize():
    """Shorten, center and normalize the vectors like the other embedders."""
    embedder = VideoEmbedder(n_frames=2, dimensions=3, backend='dummy').fit(VIDEOS)
    embeddings = embedder.transform(VIDEOS)
    assert embeddings.shape == (3, 3)
    np.testing.assert_allclose(embeddings.mean(axis=0), 0, atol=1e-12)
    normalized = VideoEmbedder(n_frames=2, normalize=True, backend='dummy').fit(VIDEOS).transform(VIDEOS)
    np.testing.assert_allclose(np.linalg.norm(normalized, axis=1), 1)


def test_fit_without_videos_probes_dimensions():
    """Find the dimensions from a probe video when `fit` gets no videos."""
    embedder = VideoEmbedder(backend='dummy').fit()
    assert embedder.n_features_out_ == 8
    assert embedder.get_feature_names_out().tolist() == [f'videoembedder{index}' for index in range(8)]


def test_clone_and_not_fitted():
    """Clone with the same parameters and refuse to transform before `fit`."""
    embedder = clone(VideoEmbedder(n_frames=3, backend='dummy'))
    assert embedder.get_params()['n_frames'] == 3
    with pytest.raises(NotFittedError):
        embedder.transform(VIDEOS)
