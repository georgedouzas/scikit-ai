"""Test the video generator."""

import pytest
from sklearn.utils._param_validation import InvalidParameterError

from skai.core import Video
from skai.generation import VideoGenerator


def test_predict_returns_requested_duration_and_size():
    """Return one video of the requested duration and size."""
    video = VideoGenerator(duration=2, size=(320, 240), fps=8, backend='dummy').fit().predict(['a boat at sea'])[0]
    assert isinstance(video, Video)
    assert video.duration == 2.0
    assert video.frames.shape[1:3] == (240, 320)


def test_fit_rejects_invalid_size():
    """Reject sizes that are not a pair of positive integers."""
    with pytest.raises(InvalidParameterError, match='size'):
        VideoGenerator(size=(0, 10), backend='dummy').fit()
