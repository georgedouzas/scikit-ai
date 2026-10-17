"""Test the video classifier."""

import numpy as np

from skai.classification import VideoClassifier
from skai.core import Video

VIDEOS = [Video(np.full((10, 8, 8, 3), value, dtype=np.uint8), fps=5) for value in (10, 20, 30)]


def test_predict_zero_shot_video_labels():
    """Return one label from `labels` per video."""
    predictions = VideoClassifier(labels=['cooking', 'sports'], n_frames=4, backend='dummy').fit().predict(VIDEOS)
    assert len(predictions) == 3
    assert set(predictions) <= {'cooking', 'sports'}


def test_predict_multilabel_videos():
    """Return an indicator matrix with one column per label."""
    y = np.array([[1, 0, 1], [0, 1, 0], [1, 1, 0]])
    classifier = VideoClassifier(labels=['a', 'b', 'c'], n_frames=2, backend='dummy').fit(VIDEOS, y)
    assert classifier.predict(VIDEOS).shape == (3, 3)
