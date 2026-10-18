"""Test the video classifier."""

import numpy as np

from skai.backends import DummyBackend
from skai.classification import VideoClassifier
from skai.core import Video, VideoFrames

VIDEOS = [Video(np.full((10, 8, 8, 3), value, dtype=np.uint8), fps=5) for value in (10, 20, 30)]


def recording_backend():
    """Return a dummy backend that records its video classification requests in `requests`."""
    requests = []

    class RecordingBackend(DummyBackend):
        """Record the requests, also across clones."""

        def classify_video(self, request):
            """Record the request and score it like the dummy backend."""
            requests.append(request)
            return super().classify_video(request)

    backend = RecordingBackend()
    backend.requests = requests
    return backend


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


def test_declares_a_classify_operation():
    """Run the classification as the `classify_video` backend task."""
    assert VideoClassifier.operations == {'classify': 'classify_video'}
    assert VideoClassifier(labels=['a'], backend='dummy').fit().task_ == 'classify_video'


def test_predict_sends_n_frames_sampled_frames():
    """Read every video through the base reader as `n_frames` evenly sampled frames."""
    backend = recording_backend()
    VideoClassifier(labels=['a', 'b'], n_frames=4, batch_size=1, backend=backend).fit().predict(VIDEOS[:2])
    assert len(backend.requests) == 2
    for request in backend.requests:
        assert isinstance(request.sample, VideoFrames)
        assert len(request.sample.frames) == 4
        assert request.sample.duration == 2.0
