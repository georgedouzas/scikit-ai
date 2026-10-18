"""Test the image segmenter."""

import pytest
from PIL import Image
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import InvalidParameterError

from skai.core import InvalidSampleError
from skai.segmentation import ImageSegmenter

PHOTO = Image.new('RGB', (64, 48), (90, 160, 220))


def test_regions_feed_later_steps():
    """Return regions whose media are PIL images."""
    regions = ImageSegmenter(labels=['person'], backend='dummy').fit().predict([PHOTO])[0]
    assert regions
    assert all(isinstance(region['media'], Image.Image) for region in regions)
    assert all({'label', 'score', 'media', 'box'} <= set(region) for region in regions)


def test_object_regions_are_crops_of_their_boxes():
    """Name the regions by `labels` and crop the image to every box."""
    regions = ImageSegmenter(labels=['person', 'dog'], backend='dummy').fit().predict([PHOTO])[0]
    assert [region['label'] for region in regions] == ['person', 'dog']
    for region in regions:
        left, top, right, bottom = region['box']
        assert 0 <= left < right <= PHOTO.width
        assert 0 <= top < bottom <= PHOTO.height
        assert region['media'].size == (right - left, bottom - top)
        assert 'mask' not in region


def test_mask_for_an_edit():
    """Return a grayscale mask of the photo's size for every region described by `prompt`."""
    regions = ImageSegmenter(by='mask', prompt='the sky', backend='dummy').fit().predict([PHOTO, PHOTO])
    assert len(regions) == 2
    for region in regions[0]:
        assert region['label'] == 'the sky'
        assert region['mask'].size == PHOTO.size
        assert region['mask'].mode == 'L'
        left, top, right, bottom = region['box']
        assert region['mask'].getpixel((left, top)) == 255
        assert region['media'].mode == 'RGBA'
        assert region['media'].size == (right - left, bottom - top)


def test_invalid_by_names_the_parameter_and_choices():
    """Reject a `by` that images do not offer and name the valid values."""
    with pytest.raises(InvalidParameterError, match="'by' parameter") as error:
        ImageSegmenter(by='shot', backend='dummy').fit()
    assert all(repr(value) in str(error.value) for value in ['mask', 'object'])


def test_rejects_list_valued_samples():
    """Reject a list sample and point to `ForEach`."""
    with pytest.raises(InvalidSampleError, match='ForEach'):
        ImageSegmenter(labels=['a'], backend='dummy').fit().predict([[PHOTO, PHOTO]])


def test_clone_and_not_fitted():
    """Clone with the same parameters and refuse to predict before `fit`."""
    segmenter = ImageSegmenter(by='mask', labels=['cat'], backend='dummy')
    cloned = clone(segmenter)
    assert cloned.get_params() == segmenter.get_params()
    with pytest.raises(NotFittedError):
        cloned.predict([PHOTO])
