"""Test the multimodal classifier."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from skai.backends import DummyBackend
from skai.classification import MultimodalClassifier
from skai.core import InvalidSampleError, Video, VideoFrames

PHOTO = Image.new('RGB', (16, 12), (10, 120, 200))
POSTS = [
    {'image': PHOTO, 'caption': 'hello'},
    {'image': Image.new('RGB', (16, 12), (200, 0, 0)), 'caption': 'buy now'},
    {'image': Image.new('RGB', (16, 12), (0, 200, 0)), 'caption': 'nice day'},
]


def recording_backend():
    """Return a dummy backend that records its multimodal classification requests in `requests`."""
    requests = []

    class RecordingBackend(DummyBackend):
        """Record the requests, also across clones."""

        def classify_multimodal(self, request):
            """Record the request and score it like the dummy backend."""
            requests.append(request)
            return super().classify_multimodal(request)

    backend = RecordingBackend()
    backend.requests = requests
    return backend


def test_post_with_image_and_caption():
    """Return one label from `labels` and send every named part in one request."""
    backend = recording_backend()
    classifier = MultimodalClassifier(labels=['safe', 'unsafe'], backend=backend).fit()
    predictions = classifier.predict([{'image': PHOTO, 'caption': 'hello'}])
    assert predictions.shape == (1,)
    assert predictions[0] in {'safe', 'unsafe'}
    assert len(backend.requests) == 1
    sample = backend.requests[0].sample
    assert set(sample) == {'image', 'caption'}
    assert isinstance(sample['image'], Image.Image)
    assert sample['caption'] == 'hello'


def test_predict_reads_dataframe_rows_as_samples():
    """Read every DataFrame row as one sample keyed by column."""
    backend = recording_backend()
    frame = pd.DataFrame({'image': [post['image'] for post in POSTS], 'caption': [post['caption'] for post in POSTS]})
    classifier = MultimodalClassifier(labels=['safe', 'unsafe'], backend=backend).fit()
    assert classifier.predict(frame).shape == (3,)
    assert classifier.predict_proba(frame).shape == (3, 2)
    assert [request.sample['caption'] for request in backend.requests[:3]] == ['hello', 'buy now', 'nice day']


def test_fit_with_targets_selects_multimodal_examples():
    """Keep `k_shot` samples with their labels as examples."""
    classifier = MultimodalClassifier(k_shot=2, random_state=0, backend='dummy').fit(POSTS, ['safe', 'unsafe', 'safe'])
    assert classifier.classes_.tolist() == ['safe', 'unsafe']
    assert len(classifier.examples_) == 2
    assert all(set(sample) == {'image', 'caption'} for sample, _ in classifier.examples_)


def test_videos_are_sampled_to_n_frames():
    """Sample `n_frames` frames from the video parts of a sample."""
    backend = recording_backend()
    video = Video(np.zeros((10, 8, 8, 3), dtype=np.uint8), fps=5)
    MultimodalClassifier(labels=['a', 'b'], n_frames=3, backend=backend).fit().predict([{'clip': video, 'note': 'x'}])
    clip = backend.requests[0].sample['clip']
    assert isinstance(clip, VideoFrames)
    assert len(clip.frames) == 3


def test_predict_rejects_samples_that_are_not_dicts():
    """Reject a sample that is not a dict of named media."""
    classifier = MultimodalClassifier(labels=['a', 'b'], backend='dummy').fit()
    with pytest.raises(InvalidSampleError, match='dict of named media'):
        classifier.predict(['just a text'])


def test_predict_rejects_list_valued_samples():
    """Reject a list sample and point to `ForEach`."""
    classifier = MultimodalClassifier(labels=['a', 'b'], backend='dummy').fit()
    with pytest.raises(InvalidSampleError, match='ForEach'):
        classifier.predict([[POSTS[0], POSTS[1]]])


def test_clone_and_not_fitted():
    """Clone with the same parameters and refuse to predict before `fit`."""
    classifier = MultimodalClassifier(labels=['a', 'b'], n_frames=3, backend='dummy')
    cloned = clone(classifier)
    assert cloned.get_params() == classifier.get_params()
    assert cloned.get_params()['n_frames'] == 3
    with pytest.raises(NotFittedError):
        cloned.predict(POSTS)
