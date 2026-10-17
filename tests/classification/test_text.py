"""Test the text classifier."""

import numpy as np
import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skai.classification import TextClassifier

REVIEWS = [
    'This is the worst thing I have ever bought.',
    'It does the job, nothing special.',
    'I am extremely happy with this purchase!',
    'Very poor quality and bad support.',
    'Excellent product! Will buy again.',
    'Disappointed. I expected more.',
]
SENTIMENTS = ['negative', 'neutral', 'positive', 'negative', 'positive', 'negative']


def test_fit_predict_few_shot_returns_known_classes():
    """Predict one of the learned classes for every review with few-shot examples."""
    classifier = TextClassifier(k_shot=4, random_state=0, backend='dummy').fit(REVIEWS, SENTIMENTS)
    predictions = classifier.predict(['Superb quality!', 'Not impressed at all.'])
    assert predictions.shape == (2,)
    assert set(predictions) <= set(classifier.classes_)
    assert len(classifier.examples_) == 4


def test_fit_with_label_descriptions_keeps_label_keys():
    """Use the keys of a label dict as classes and its values as descriptions."""
    labels = {0: 'negative', 1: 'neutral', 2: 'positive'}
    classifier = TextClassifier(labels=labels, backend='dummy').fit(REVIEWS[:3], [0, 1, 2])
    assert classifier.classes_.tolist() == [0, 1, 2]
    assert set(classifier.predict(REVIEWS)) <= {0, 1, 2}


def test_fit_with_multilabel_targets_predicts_indicator_matrix():
    """Predict an indicator matrix of shape `(m, k)` for multilabel targets."""
    y = np.array([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [0, 1]])
    classifier = TextClassifier(backend='dummy').fit(REVIEWS, y)
    assert classifier.predict(REVIEWS[:4]).shape == (4, 2)
    assert classifier.classes_.tolist() == [0, 1]


@pytest.mark.parametrize(
    'params',
    [{'k_shot': -1}, {'labels': 'positive'}, {'instructions': 3}, {'on_error': 'skip'}, {'backend': 3}],
)
def test_fit_rejects_invalid_parameters(params):
    """Report invalid parameters at `fit`."""
    with pytest.raises(InvalidParameterError):
        TextClassifier(**params).fit(REVIEWS, SENTIMENTS)
