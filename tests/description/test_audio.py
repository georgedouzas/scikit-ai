"""Test the audio descriptor."""

import numpy as np
from sklearn.pipeline import make_pipeline

from skai.classification import TextClassifier
from skai.description import AudioDescriptor

AUDIO = [(np.zeros(size, dtype=np.float32), 16000) for size in (8000, 16000, 24000, 32000)]


def test_predict_describes_sounds():
    """Return one description string per audio sample."""
    descriptions = AudioDescriptor(backend='dummy').fit().predict(AUDIO[:2])
    assert descriptions.tolist() == [
        'A brief description of 0.50 seconds of audio.',
        'A brief description of 1.00 seconds of audio.',
    ]


def test_descriptions_feed_text_classifier():
    """Classify audio through its descriptions in a pipeline."""
    pipeline = make_pipeline(AudioDescriptor(backend='dummy'), TextClassifier(backend='dummy'))
    predictions = pipeline.fit(AUDIO, ['short', 'short', 'long', 'long']).predict(AUDIO)
    assert set(predictions) <= {'short', 'long'}


def test_fit_keeps_reference_examples():
    """Keep audio and reference description pairs as examples."""
    descriptor = AudioDescriptor(k_shot=1, random_state=0, backend='dummy').fit(AUDIO[:2], ['Rain.', 'Wind.'])
    assert len(descriptor.examples_) == 1
