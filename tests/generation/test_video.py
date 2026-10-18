"""Test the generators that return videos."""

from typing import ClassVar

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.utils._param_validation import InvalidParameterError

from skai.backends import DummyBackend
from skai.core import Audio, InvalidSampleError, Video
from skai.generation import (
    AudioToVideoGenerator,
    ImageToVideoGenerator,
    MultimodalToVideoGenerator,
    TextToAudioGenerator,
    TextToVideoGenerator,
    VideoToVideoGenerator,
)

RATE = 16000


def clip(seconds, fps=8, size=(16, 12), audio=None, value=0):
    """Return a video of one gray level."""
    width, height = size
    frames = np.full((round(seconds * fps), height, width, 3), value, dtype=np.uint8)
    return Video(frames, fps, audio)


def speech(seconds):
    """Return a quiet tone of a length."""
    times = np.arange(round(seconds * RATE)) / RATE
    return Audio((0.1 * np.sin(2 * np.pi * 220 * times)).astype(np.float32), RATE)


class RecordingBackend(DummyBackend):
    """Keep every request the backend receives."""

    requests: ClassVar[list] = []

    def run(self, task, requests):
        """Keep the requests and run them like the dummy backend."""
        RecordingBackend.requests.extend(requests)
        return super().run(task, requests)


class SizedBackend(DummyBackend):
    """Take portraits of 512 pixels."""

    avatar_size = 512


@pytest.fixture(autouse=True)
def clear_requests():
    """Start every test with no recorded requests."""
    RecordingBackend.requests = []


def test_requested_duration_and_size():
    """Return one video of the requested duration whose frames have the requested size."""
    videos = TextToVideoGenerator(duration=2, size=(320, 240), fps=8, backend='dummy').fit().predict(['a boat at sea'])
    assert len(videos) == 1
    assert isinstance(videos[0], Video)
    assert videos[0].duration == 2.0
    assert videos[0].frames.shape[1:3] == (240, 320)


def test_text_to_video_sends_style_and_negative_prompt():
    """Send the style and the negative prompt with every request."""
    generator = TextToVideoGenerator(
        duration=1,
        size=(8, 8),
        style='anime',
        negative_prompt='text',
        backend=RecordingBackend(),
    )
    generator.fit().predict(['a boat', 'a car'])
    assert [(request.style, request.negative_prompt) for request in RecordingBackend.requests] == [
        ('anime', 'text'),
    ] * 2


def test_text_to_video_rejects_invalid_size():
    """Reject sizes that are not a pair of positive integers."""
    with pytest.raises(InvalidParameterError, match='size'):
        TextToVideoGenerator(size=(0, 10), backend='dummy').fit()


def test_animate_images():
    """Return one video of the requested duration per image, with the motion instructions in every request."""
    animator = ImageToVideoGenerator(instructions='slow zoom', duration=2, backend=RecordingBackend()).fit()
    video = animator.predict([Image.new('RGB', (32, 24), 'teal')])[0]
    assert video.duration == 2.0
    assert video.frames[0].shape[:2] == (24, 32)
    assert RecordingBackend.requests[-1].motion_prompt == 'slow zoom'


def test_talking_head_follows_the_audio(tmp_path, png_bytes):
    """Return one video per audio sample, as long as the audio and with it as the track."""
    hello = Audio(np.zeros(16000, dtype=np.float32), 16000).save(tmp_path / 'hello.wav')
    bye = Audio(np.zeros(8000, dtype=np.float32), 16000).save(tmp_path / 'bye.wav')
    (tmp_path / 'me.png').write_bytes(png_bytes)
    videos = AudioToVideoGenerator(avatar=tmp_path / 'me.png', backend='dummy').fit().predict([hello, bye])
    assert all(isinstance(video, Video) for video in videos)
    assert [video.duration for video in videos] == pytest.approx([1.0, 0.5], abs=1 / 25)
    assert videos[0].audio is not None


def test_talking_head_requires_avatar():
    """Reject a missing `avatar` at `fit`."""
    with pytest.raises(InvalidParameterError, match='avatar'):
        AudioToVideoGenerator(backend='dummy').fit()


def test_prepared_portrait():
    """Convert the portrait to RGB, crop it to a centered square and resize it to the backend's input size."""
    generator = AudioToVideoGenerator(avatar=Image.new('RGBA', (640, 480)), backend=SizedBackend()).fit()
    assert (generator.avatar_.size, generator.avatar_.mode) == ((512, 512), 'RGB')
    portrait = Image.new('RGB', (60, 40), 'black')
    portrait.paste((255, 255, 255), (10, 0, 50, 40))
    centered = AudioToVideoGenerator(avatar=portrait, backend='dummy').fit().avatar_
    assert centered.size == (64, 64)
    assert centered.getpixel((0, 0)) == (255, 255, 255)


def test_speech_is_animated_in_a_pipeline():
    """Return one video per text from a speech generator followed by a talking head."""
    pipeline = make_pipeline(
        TextToAudioGenerator(backend='dummy'),
        AudioToVideoGenerator(avatar=Image.new('RGB', (16, 16)), backend='dummy'),
    )
    videos = pipeline.fit(None).predict(['Hello there', 'Goodbye'])
    assert len(videos) == 2
    assert all(isinstance(video, Video) for video in videos)


def test_reframe_follows_the_subject():
    """Crop every video to the aspect ratio, detecting the subject on every sampled frame."""
    generator = VideoToVideoGenerator(aspect_ratio=9 / 16, subject='dog', n_frames=4, backend=RecordingBackend()).fit()
    reframed = generator.predict([clip(2, size=(64, 36))])[0]
    height, width = reframed.frames.shape[1:3]
    assert height == 36
    assert width == pytest.approx(36 * 9 / 16, abs=1)
    assert reframed.duration == 2.0
    assert len(RecordingBackend.requests) == 4
    assert all(request.labels == ('dog',) for request in RecordingBackend.requests)


def test_concatenate_clips():
    """Join a list of clips into one video of the summed durations, keeping the order."""
    clips = [clip(1, value=10), clip(0.5, value=20), clip(0.25, value=30)]
    joined = VideoToVideoGenerator(operation='concatenate').fit().predict([clips])[0]
    assert joined.duration == pytest.approx(1.75)
    assert joined.frames[0, 0, 0, 0] == 10
    assert joined.frames[-1, 0, 0, 0] == 30


def test_lip_sync():
    """Return one video as long as the speech, with the speech as its track."""
    generator = MultimodalToVideoGenerator(operation='lip_sync', backend='dummy').fit()
    video = generator.predict([{'video': clip(1), 'audio': speech(2.5)}])[0]
    assert isinstance(video, Video)
    assert video.duration == pytest.approx(2.5, abs=1 / 8)
    assert video.audio.duration == pytest.approx(2.5)


def test_mux_replaces_the_soundtrack():
    """Replace the soundtrack of the video, cut or padded to its length, without a backend."""
    muxer = MultimodalToVideoGenerator(operation='mux').fit()
    assert muxer.backend_ is None
    videos = muxer.predict([{'video': clip(2), 'audio': speech(3)}, {'video': clip(2), 'audio': speech(1)}])
    assert [video.duration for video in videos] == [2.0, 2.0]
    assert [video.audio.duration for video in videos] == pytest.approx([2.0, 2.0])


def test_subtitles_burned_or_attached():
    """Draw the transcript on the frames, or attach it in SRT format with `burn=False`."""
    sample = {'video': clip(2), 'transcript': [{'start': 0.0, 'end': 1.0, 'text': 'Hello there'}]}
    burned = MultimodalToVideoGenerator(operation='subtitle').fit().predict([sample])[0]
    assert burned.frames[:8].any()
    attached = MultimodalToVideoGenerator(operation='subtitle', burn=False).fit().predict([sample])[0]
    assert not attached.frames.any()
    assert '00:00:00,000 --> 00:00:01,000' in attached.subtitles
    assert 'Hello there' in attached.subtitles
    plain = (
        MultimodalToVideoGenerator(operation='subtitle', burn=False)
        .fit()
        .predict([{'video': clip(1), 'transcript': 'Hi.'}])
    )
    assert 'Hi.' in plain[0].subtitles


def test_subtitle_file_is_written_next_to_the_video(tmp_path):
    """Write the SRT file next to the MP4 file of a path output."""
    generator = MultimodalToVideoGenerator(operation='subtitle', burn=False, output_format='path', output_dir=tmp_path)
    (path,) = generator.fit().predict([{'video': clip(1), 'transcript': 'Hi.'}])
    assert str(path).endswith('.mp4')
    assert 'Hi.' in (tmp_path / path).with_suffix('.srt').read_text(encoding='utf-8')


@pytest.mark.parametrize(('operation', 'key'), [('mux', 'audio'), ('lip_sync', 'audio'), ('subtitle', 'transcript')])
def test_missing_key(operation, key):
    """Name the missing key and the index of the sample."""
    generator = MultimodalToVideoGenerator(operation=operation, backend='dummy').fit()
    samples = [{'video': clip(1), 'audio': speech(1), 'transcript': 'Hi.'}, {'video': clip(1)}]
    with pytest.raises(InvalidSampleError, match=rf'Sample 1.*`{key}`'):
        generator.predict(samples)


def test_dataframe_rows_are_samples():
    """Read every row of a DataFrame as one multimodal sample."""
    frame = pd.DataFrame({'video': [clip(1), clip(2)], 'audio': [speech(1), speech(1)]})
    videos = MultimodalToVideoGenerator(operation='mux').fit().predict(frame)
    assert [video.duration for video in videos] == [1.0, 2.0]
