"""Test the audio embedder."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from skai.embedding import AudioEmbedder

AUDIO = [(np.full(1600, value / 10, dtype=np.float32), 16000) for value in range(8)]


def test_audio_vectors_feed_a_classifier():
    """Predict one label per audio sample through the embeddings."""
    labels = ['quiet'] * 4 + ['loud'] * 4
    pipeline = make_pipeline(AudioEmbedder(backend='dummy'), LogisticRegression())
    assert pipeline.fit(AUDIO, labels).predict(AUDIO).shape == (8,)


def test_fit_without_audio_probes_dimensions():
    """Find the dimensions from a silent probe when `fit` gets no audio."""
    assert AudioEmbedder(backend='dummy').fit().n_features_out_ == 8
