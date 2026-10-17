"""Test the image classifier."""

import pytest
from PIL import Image

from skai.classification import ImageClassifier

IMAGES = [Image.new('RGB', (8, 8), color) for color in ('red', 'green', 'blue')]


def test_predict_zero_shot_uses_labels():
    """Predict only the given labels and score two columns."""
    classifier = ImageClassifier(labels=['cat', 'dog'], backend='dummy').fit(None)
    assert set(classifier.predict(IMAGES)) <= {'cat', 'dog'}
    assert classifier.predict_proba(IMAGES).shape == (3, 2)


def test_fit_rejects_targets_outside_labels():
    """List the target values that are not in `labels`."""
    with pytest.raises(ValueError, match=r"not in `labels`: \['bird'\]"):
        ImageClassifier(labels=['cat', 'dog'], backend='dummy').fit(IMAGES, ['cat', 'dog', 'bird'])


def test_fit_with_targets_inside_labels_uses_all_labels():
    """Keep every label as a class when the targets use only some of them."""
    classifier = ImageClassifier(labels=['cat', 'dog', 'bird'], backend='dummy').fit(IMAGES[:2], ['cat', 'dog'])
    assert classifier.classes_.tolist() == ['bird', 'cat', 'dog']
