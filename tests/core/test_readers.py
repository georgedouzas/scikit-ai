"""Test reading samples of every input kind, including multimodal dicts and DataFrame rows."""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from skai.backends import DummyBackend
from skai.classification import MultimodalClassifier, TextClassifier, VideoClassifier
from skai.core import (
    Audio,
    InvalidSampleError,
    MediaReadError,
    Video,
    VideoFrames,
    input_kind_of,
    read_input,
    read_multimodal,
    read_video_frames,
    to_samples,
)
from skai.generation import MultimodalToVideoGenerator, TextToTextGenerator


class RecordingBackend(DummyBackend):
    """Keep every request the backend receives."""

    requests: ClassVar[list] = []

    def run(self, task, requests):
        """Keep the requests and run them like the dummy backend."""
        RecordingBackend.requests.extend(requests)
        return super().run(task, requests)


@pytest.fixture(autouse=True)
def clear_requests():
    """Start every test with no recorded requests."""
    RecordingBackend.requests = []


@pytest.fixture
def video(audio):
    """Return a one-second video of 10 frames with a soundtrack."""
    return Video(np.zeros((10, 8, 8, 3), dtype=np.uint8), fps=10, audio=audio)


def test_to_samples_reads_dataframe_rows_as_dicts():
    """Turn every DataFrame row into a dict keyed by column only when `rows=True`."""
    frame = pd.DataFrame({'title': ['a', 'b'], 'views': [1, 2]})
    assert to_samples(frame, rows=True) == [{'title': 'a', 'views': 1}, {'title': 'b', 'views': 2}]
    with pytest.raises(ValueError, match='one column of samples'):
        to_samples(frame)
    assert to_samples(pd.Series(['x', 'y']), rows=True) == ['x', 'y']
    assert to_samples(None, rows=True) == []


def test_read_multimodal_reads_every_part_by_its_kind(tmp_path, image, png_bytes, audio):
    """Read images, audio and texts from paths, bytes, arrays and tuples."""
    (tmp_path / 'cover.png').write_bytes(png_bytes)
    audio.save(tmp_path / 'voice.wav')
    sample = {
        'title': 'A title',
        'cover': str(tmp_path / 'cover.png'),
        'voice': tmp_path / 'voice.wav',
        'thumbnail': png_bytes,
        'jingle': audio.to_bytes(),
        'pixels': np.asarray(image),
        'track': (audio.samples, audio.sample_rate),
        'photo': image,
        'views': 3,
    }
    read = read_multimodal(sample)
    assert read['title'] == 'A title'
    assert isinstance(read['cover'], Image.Image)
    assert read['cover'].size == (32, 24)
    assert isinstance(read['voice'], Audio)
    assert isinstance(read['thumbnail'], Image.Image)
    assert isinstance(read['jingle'], Audio)
    assert read['pixels'].size == (32, 24)
    assert read['track'].duration == pytest.approx(1.0)
    assert read['photo'] is image
    assert read['views'] == 3
    assert input_kind_of(read) == 'multimodal'


def test_read_multimodal_reads_strings_as_texts_unless_they_name_files_or_urls(tmp_path, server, png_bytes):
    """Read a string as a text unless it is a URL or the path of an existing media file."""
    directory, url = server
    (directory / 'cover.png').write_bytes(png_bytes)
    (tmp_path / 'notes.txt').write_text('notes')
    read = read_multimodal(
        {
            'missing': str(tmp_path / 'missing.png'),
            'sentence': 'See photo.png for details.',
            'text_file': str(tmp_path / 'notes.txt'),
            'remote': f'{url}/cover.png',
        },
    )
    assert read['missing'] == str(tmp_path / 'missing.png')
    assert read['sentence'] == 'See photo.png for details.'
    assert read['text_file'] == str(tmp_path / 'notes.txt')
    assert isinstance(read['remote'], Image.Image)


def test_read_multimodal_samples_videos_only_with_n_frames(video):
    """Keep videos whole without `n_frames` and sample frames with it."""
    assert read_multimodal({'clip': video})['clip'] is video
    frames = read_multimodal({'clip': video}, n_frames=3)['clip']
    assert isinstance(frames, VideoFrames)
    assert len(frames.frames) == 3
    assert frames.audio is not None


def test_read_multimodal_accepts_pandas_rows():
    """Read a pandas row as a dict keyed by column."""
    row = pd.DataFrame({'title': ['a'], 'body': ['b']}).iloc[0]
    assert read_multimodal(row) == {'title': 'a', 'body': 'b'}


def test_read_multimodal_names_the_missing_key_and_index():
    """Name the index of the sample and the first missing required key."""
    with pytest.raises(InvalidSampleError, match=r'Sample 4 has no `audio` key\. The keys \[.video., .audio.\]'):
        read_multimodal({'video': 'clip.mp4'}, index=4, required=('video', 'audio'))
    with pytest.raises(InvalidSampleError, match='Sample 2 must be a dict of named media, but got str'):
        read_multimodal('just text', index=2)


def test_read_input_reads_each_kind(image, audio, video):
    """Read a sample of every input kind, sampling video frames only with `n_frames`."""
    assert read_input('text', 'hi', 0) == 'hi'
    assert read_input('image', np.asarray(image), 0).size == (32, 24)
    assert read_input('audio', (audio.samples, audio.sample_rate), 0).duration == pytest.approx(1.0)
    assert read_input('video', video, 0) is video
    assert len(read_input('video', video, 0, n_frames=2).frames) == 2
    assert read_input('multimodal', {'a': 'b'}, 0) == {'a': 'b'}
    with pytest.raises(InvalidSampleError, match='Sample 7 must be a string'):
        read_input('text', image, 7)


def test_read_video_frames_samples_evenly_and_keeps_frames(video):
    """Sample evenly spaced frames, and return already sampled frames unchanged."""
    frames = read_video_frames(video, 0, 5)
    assert frames.timestamps == (0.0, 0.2, 0.4, 0.6, 0.8)
    assert frames.duration == 1.0
    assert read_video_frames(frames, 0, 2) is frames


def test_read_video_frames_reports_undecodable_videos():
    """Name the index of a video that cannot be decoded."""
    with pytest.raises(MediaReadError, match='Sample 3'):
        read_video_frames(b'not a video', 3, 2)


def test_input_kind_of_names_every_kind(image, audio, video):
    """Name the input kind of every read sample."""
    frames = read_video_frames(video, 0, 2)
    assert [input_kind_of(sample) for sample in ['t', image, audio, frames, video, {'a': 1}]] == [
        'text',
        'image',
        'audio',
        'video',
        'video',
        'multimodal',
    ]


def test_components_sample_video_frames_with_n_frames(video):
    """Send `n_frames` sampled frames of every video to the backend."""
    VideoClassifier(labels=['a', 'b'], n_frames=3, backend=RecordingBackend()).fit().predict([video])
    sample = RecordingBackend.requests[-1].sample
    assert isinstance(sample, VideoFrames)
    assert len(sample.frames) == 3


def test_multimodal_components_read_dataframe_rows(image):
    """Send every DataFrame row as one dict sample with its parts read."""
    X = pd.DataFrame({'title': ['Free money', 'Weather'], 'cover': [image, image]})
    predictions = MultimodalClassifier(labels=['fake', 'real'], backend=RecordingBackend()).fit().predict(X)
    assert predictions.shape == (2,)
    samples = [request.sample for request in RecordingBackend.requests]
    assert samples[0]['title'] == 'Free money'
    assert samples[1]['cover'] is image


def test_components_name_missing_required_keys(video):
    """Name the missing key and the index of the sample for the operation of a component."""
    generator = MultimodalToVideoGenerator(operation='mux').fit()
    with pytest.raises(InvalidSampleError, match='Sample 1 has no `audio` key'):
        generator.predict([{'video': video, 'audio': video.audio}, {'video': video}])


def test_list_valued_samples_suggest_for_each():
    """Reject a list-valued sample and suggest wrapping the step in `ForEach`."""
    with pytest.raises(InvalidSampleError, match=r'Sample 1 is a list.*TextClassifier.*`ForEach`'):
        TextClassifier(labels=['a', 'b'], backend='dummy').fit().predict(['ok', ['a', 'b']])
    generator = TextToTextGenerator(instructions='Summarize.', backend='dummy').fit()
    with pytest.raises(InvalidSampleError, match='ForEach'):
        generator.predict([['one', 'two']])
