"""Test the image generator."""

import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skai.text import ImageGenerator


def test_predict_returns_requested_size():
    """Return one image of the requested size per prompt."""
    images = ImageGenerator(size=(512, 512), backend='dummy').fit().predict(['a cat', 'a dog'])
    assert [image.size for image in images] == [(512, 512)] * 2


@pytest.mark.parametrize('size', [(0, 5), (1, 2, 3), (1.5, 2)])
def test_fit_rejects_invalid_size(size):
    """Reject sizes that are not a pair of positive integers."""
    with pytest.raises(InvalidParameterError, match='size'):
        ImageGenerator(size=size, backend='dummy').fit()


def test_predict_differs_by_seed():
    """Return different images for different seeds and equal images for equal seeds."""
    first = ImageGenerator(size=(4, 4), random_state=0, backend='dummy').fit().predict(['a cat'])[0]
    second = ImageGenerator(size=(4, 4), random_state=0, backend='dummy').fit().predict(['a cat'])[0]
    other = ImageGenerator(size=(4, 4), random_state=1, backend='dummy').fit().predict(['a cat'])[0]
    assert first.tobytes() == second.tobytes()
    assert first.tobytes() != other.tobytes()
