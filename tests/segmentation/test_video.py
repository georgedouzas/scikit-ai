"""Test the video segmenter."""

import sys

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError
from sklearn.utils._param_validation import InvalidParameterError

from skai.core import InvalidSampleError, Video
from skai.segmentation import VideoSegmenter

FPS = 4
SHOTS = np.concatenate(
    [np.zeros((4, 8, 8, 3)), np.full((4, 8, 8, 3), 255), np.zeros((4, 8, 8, 3))],
).astype(np.uint8)
TWO_CUTS = Video(SHOTS, fps=FPS)
STILL = Video(np.full((8, 8, 8, 3), 100, dtype=np.uint8), fps=FPS)
PROVIDERS = ('openai', 'transformers', 'litellm', 'torch', 'replicate', 'diffusers')


def test_shots_cover_the_video():
    """Return three shots in time order whose spans cover a video with two cuts."""
    regions = VideoSegmenter(by='shot').fit().predict([TWO_CUTS])[0]
    spans = [(region['start'], region['end']) for region in regions]
    assert spans == [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)]
    assert spans[0][0] == 0.0
    assert spans[-1][1] == TWO_CUTS.duration
    for region in regions:
        assert region['label'] == 'shot'
        assert isinstance(region['media'], Video)
        assert region['media'].frames.shape[0] == 4


def test_still_video_is_one_shot():
    """Return one shot for a video without cuts, and one shot when the threshold is too high for any cut."""
    assert len(VideoSegmenter().fit().predict([STILL])[0]) == 1
    assert len(VideoSegmenter(threshold=1.0).fit().predict([TWO_CUTS])[0]) == 1


def test_offline_shots_import_no_provider():
    """Find shots with `backend=None` without resolving a backend or importing a provider."""
    before = set(sys.modules)
    segmenter = VideoSegmenter(by='shot', backend=None).fit()
    assert segmenter.backend_ is None
    assert len(segmenter.predict([TWO_CUTS])[0]) == 3
    new = set(sys.modules) - before
    assert not [name for name in new if name.split('.')[0] in PROVIDERS]


def test_moments_match_the_prompt():
    """Return the moments the backend finds for `prompt` with their clips, and require a prompt."""
    regions = VideoSegmenter(by='moment', prompt='a goal', n_frames=4, backend='dummy').fit().predict([TWO_CUTS])[0]
    assert len(regions) == 1
    region = regions[0]
    assert region['label'] == 'moment'
    assert 0 <= region['start'] < region['end'] <= TWO_CUTS.duration
    assert isinstance(region['media'], Video)
    with pytest.raises(ValueError, match='needs `prompt`'):
        VideoSegmenter(by='moment', backend='dummy').fit()


def test_invalid_by_names_the_parameter_and_choices():
    """Reject a `by` that videos do not offer and name the valid values."""
    with pytest.raises(InvalidParameterError, match="'by' parameter") as error:
        VideoSegmenter(by='speaker').fit()
    assert all(repr(value) in str(error.value) for value in ['moment', 'shot'])


def test_rejects_list_valued_samples():
    """Reject a list sample and point to `ForEach`."""
    with pytest.raises(InvalidSampleError, match='ForEach'):
        VideoSegmenter().fit().predict([[STILL, STILL]])


def test_clone_and_not_fitted():
    """Clone with the same parameters and refuse to predict before `fit`."""
    segmenter = VideoSegmenter(by='moment', prompt='a goal', threshold=0.3, n_frames=4)
    cloned = clone(segmenter)
    assert cloned.get_params() == segmenter.get_params()
    with pytest.raises(NotFittedError):
        cloned.predict([STILL])
