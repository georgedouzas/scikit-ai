"""Test the audio classifier."""

import numpy as np

from skai.classification import AudioClassifier

AUDIO = [(np.full(1600, value, dtype=np.float32), 16000) for value in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6)]


def test_predict_zero_shot_sound_events():
    """Predict only the given labels and score two columns."""
    classifier = AudioClassifier(labels=['dog bark', 'siren'], backend='dummy').fit()
    assert set(classifier.predict(AUDIO[:2])) <= {'dog bark', 'siren'}
    assert classifier.predict_proba(AUDIO[:2]).shape == (2, 2)


def test_fit_selects_audio_examples():
    """Keep `k_shot` training samples with their labels as examples."""
    classifier = AudioClassifier(k_shot=2, random_state=0, backend='dummy').fit(AUDIO, ['a', 'b'] * 3)
    assert len(classifier.examples_) == 2
    assert all(labels in {('a',), ('b',)} for _, labels in classifier.examples_)
