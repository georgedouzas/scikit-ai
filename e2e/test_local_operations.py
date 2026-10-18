"""Run the local operations, which need no provider, on the media that OpenAI generated, as a sanity pass."""

import numpy as np
import pytest
from PIL import Image

from skai.core import Audio, Video
from skai.generation import (
    AudioToAudioGenerator,
    ImageToImageGenerator,
    MultimodalToAudioGenerator,
    MultimodalToVideoGenerator,
    VideoToAudioGenerator,
    VideoToImageGenerator,
    VideoToVideoGenerator,
)
from skai.segmentation import AudioSegmenter, TextSegmenter, VideoSegmenter


def test_local_video_operations(apple_video, scenes_video, shared_speech):
    """Extract the track and a frame, join, mux and subtitle videos, and split them at cuts."""
    track = VideoToAudioGenerator().fit().predict([apple_video])[0]
    assert track.duration == pytest.approx(shared_speech.duration, abs=0.1)
    frame = VideoToImageGenerator(operation='frame', at=1.0).fit().predict([apple_video])[0]
    assert frame.size == (320, 320)
    joined = VideoToVideoGenerator(operation='concatenate').fit().predict([[apple_video, scenes_video]])[0]
    assert joined.duration == pytest.approx(apple_video.duration + scenes_video.duration, abs=0.2)
    silent = Video(apple_video.frames, fps=apple_video.fps)
    muxed = MultimodalToVideoGenerator(operation='mux').fit().predict([{'video': silent, 'audio': shared_speech}])[0]
    assert muxed.audio.duration == pytest.approx(shared_speech.duration, abs=0.1)
    transcript = [{'start': 0.0, 'end': 1.0, 'text': 'Good morning'}]
    subtitler = MultimodalToVideoGenerator(operation='subtitle').fit()
    subtitled = subtitler.predict([{'video': muxed, 'transcript': transcript}])[0]
    assert 'Good morning' in subtitled.subtitles
    cut = Video(np.concatenate([np.zeros((8, 32, 32, 3), np.uint8), np.full((8, 32, 32, 3), 255, np.uint8)]), fps=8)
    shots = VideoSegmenter(by='shot').fit().predict([cut])[0]
    assert [(shot['start'], shot['end']) for shot in shots] == [(0.0, 1.0), (1.0, 2.0)]


def test_local_audio_operations(shared_speech):
    """Join and mix audio, and split it at its silences."""
    joined = AudioToAudioGenerator(operation='concatenate', gap=1.0).fit().predict([[shared_speech, shared_speech]])[0]
    assert joined.duration == pytest.approx(2 * shared_speech.duration + 1.0, abs=0.1)
    rate = shared_speech.sample_rate
    music = Audio((0.1 * np.sin(np.arange(2 * rate) / 10)).astype(np.float32), rate)
    mixer = MultimodalToAudioGenerator(duck_under=['music'], duck_for='voice')
    mixed = mixer.fit().predict([{'voice': shared_speech, 'music': music}])[0]
    assert mixed.duration == pytest.approx(max(shared_speech.duration, music.duration), abs=0.1)
    parts = AudioSegmenter(by='silence').fit().predict([joined])[0]
    assert len(parts) >= 2
    assert all(part['start'] < part['end'] for part in parts)


def test_local_image_and_text_operations(shared_images):
    """Lay images out as a collage and split a text into sentences."""
    collage = ImageToImageGenerator(operation='collage', columns=2, tile=128).fit().predict([shared_images])[0]
    assert isinstance(collage, Image.Image)
    assert collage.size == (256, 128)
    sentences = TextSegmenter(by='sentence').fit().predict(['The apple is red. The car is blue.'])[0]
    assert [sentence['media'].strip() for sentence in sentences] == ['The apple is red.', 'The car is blue.']
