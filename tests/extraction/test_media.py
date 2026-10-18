"""Test the image, audio, video and multimodal extractors."""

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.utils._param_validation import InvalidParameterError

from skai.backends import DummyBackend
from skai.core import Video, VideoFrames
from skai.extraction import AudioExtractor, ImageExtractor, MultimodalExtractor, VideoExtractor

IMAGES = [Image.new('RGB', (16, 12), color) for color in ('red', 'green')]
AUDIO = [(np.full(1600, value, dtype=np.float32), 16000) for value in (0.1, 0.2)]
VIDEOS = [Video(np.full((10, 8, 8, 3), value, dtype=np.uint8), fps=5) for value in (10, 200)]
POSTS = [{'photo': IMAGES[0], 'notes': 'SkaiBox, 20 euros'}, {'photo': IMAGES[1], 'notes': 'SkaiCam, 50 euros'}]


def recording_backend():
    """Return a dummy backend that records its extraction requests in `requests`."""
    requests = []

    class RecordingBackend(DummyBackend):
        """Record the requests, also across clones."""

        def extract_fields(self, request):
            """Record the request and extract like the dummy backend."""
            requests.append(request)
            return super().extract_fields(request)

    backend = RecordingBackend()
    backend.requests = requests
    return backend


@pytest.mark.parametrize(
    ('extractor_class', 'samples'),
    [
        (ImageExtractor, IMAGES),
        (AudioExtractor, AUDIO),
        (VideoExtractor, VIDEOS),
        (MultimodalExtractor, POSTS),
    ],
)
def test_predict_returns_one_record_per_sample(extractor_class, samples):
    """Return one dict per sample with exactly the field names."""
    fields = {'color': 'main color', 'items': [{'name': 'item name'}]}
    records = extractor_class(fields=fields, backend='dummy').fit().predict(samples)
    assert len(records) == len(samples)
    for record in records:
        assert set(record) == {'color', 'items'}
        assert isinstance(record['items'], list)
        assert all(set(item) == {'name'} for item in record['items'])


@pytest.mark.parametrize('extractor_class', [ImageExtractor, AudioExtractor, VideoExtractor, MultimodalExtractor])
def test_fit_requires_fields(extractor_class):
    """Reject a missing `fields` at `fit`."""
    with pytest.raises(InvalidParameterError, match='fields'):
        extractor_class(backend='dummy').fit()


def test_video_extractor_sends_n_frames():
    """Send `n_frames` evenly sampled frames of every video."""
    backend = recording_backend()
    VideoExtractor(fields={'place': 'where'}, n_frames=3, backend=backend).fit().predict(VIDEOS[:1])
    sample = backend.requests[0].sample
    assert isinstance(sample, VideoFrames)
    assert len(sample.frames) == 3


def test_multimodal_extractor_reads_dataframe_rows():
    """Read every DataFrame row as one sample whose parts are named by column."""
    backend = recording_backend()
    frame = pd.DataFrame({'photo': [post['photo'] for post in POSTS], 'notes': [post['notes'] for post in POSTS]})
    extractor = MultimodalExtractor(fields={'product': 'product name', 'price': 'price'}, backend=backend).fit()
    table = extractor.set_output(transform='pandas').transform(frame)
    assert list(table.columns) == ['product', 'price']
    assert table.shape == (2, 2)
    assert set(backend.requests[0].sample) == {'photo', 'notes'}
    assert isinstance(backend.requests[0].sample['photo'], Image.Image)


def test_image_extractor_learns_reference_records():
    """Keep image and record pairs as examples."""
    extractor = ImageExtractor(fields={'color': 'main color'}, backend='dummy').fit(IMAGES[:1], [{'color': 'red'}])
    ((image, record),) = extractor.examples_
    assert isinstance(image, Image.Image)
    assert record == {'color': 'red'}
