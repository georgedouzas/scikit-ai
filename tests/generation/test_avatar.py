"""Test the talking avatar generator."""

import numpy as np
import pytest
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.utils._param_validation import InvalidParameterError

from skai.core import Audio, Video
from skai.generation import SpeechSynthesizer, TalkingAvatarGenerator


def test_predict_returns_one_video_per_audio(tmp_path, png_bytes):
    """Return one video per audio sample, as long as the audio."""
    hello = Audio(np.zeros(16000, dtype=np.float32), 16000).save(tmp_path / 'hello.wav')
    bye = Audio(np.zeros(8000, dtype=np.float32), 16000).save(tmp_path / 'bye.wav')
    (tmp_path / 'me.png').write_bytes(png_bytes)
    generator = TalkingAvatarGenerator(avatar=tmp_path / 'me.png', backend='dummy').fit()
    videos = generator.predict([hello, bye])
    assert all(isinstance(video, Video) for video in videos)
    assert [video.duration for video in videos] == pytest.approx([1.0, 0.5], abs=1 / 25)
    assert videos[0].audio is not None


def test_fit_requires_avatar():
    """Reject a missing `avatar` at `fit`."""
    with pytest.raises(InvalidParameterError, match='avatar'):
        TalkingAvatarGenerator(backend='dummy').fit()


def test_synthesized_speech_is_animated_in_pipeline():
    """Return one video per text from a synthesizer followed by the avatar generator."""
    pipeline = make_pipeline(
        SpeechSynthesizer(backend='dummy'),
        TalkingAvatarGenerator(avatar=Image.new('RGB', (16, 16)), backend='dummy'),
    )
    videos = pipeline.fit(None).predict(['Hello there', 'Goodbye'])
    assert len(videos) == 2
    assert all(isinstance(video, Video) for video in videos)
