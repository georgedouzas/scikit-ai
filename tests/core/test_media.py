"""Test the audio and video containers."""

import numpy as np
import pytest

from skai.core import Audio, Video


def test_audio_reports_duration_and_repr(audio):
    """Report the duration in seconds and a short description."""
    assert audio.duration == 1.0
    assert repr(audio) == 'Audio(duration=1.00s, sample_rate=16000)'


def test_audio_to_mono_mixes_channels(audio):
    """Average the channels of stereo audio and keep mono audio unchanged."""
    stereo = Audio(np.stack([audio.samples, np.zeros_like(audio.samples)], axis=1), audio.sample_rate)
    assert stereo.to_mono().samples.shape == (16000,)
    np.testing.assert_allclose(stereo.to_mono().samples, audio.samples / 2)
    assert audio.to_mono() is audio


def test_audio_round_trips_through_bytes_and_files(tmp_path, audio):
    """Decode the WAV bytes and files that the audio writes."""
    decoded = Audio.from_bytes(audio.to_bytes())
    saved = Audio.from_bytes(audio.save(tmp_path / 'audio.wav').read_bytes())
    for copy in (decoded, saved):
        assert copy.sample_rate == audio.sample_rate
        np.testing.assert_allclose(copy.samples, audio.samples, atol=1e-4)


def test_video_requires_a_representation():
    """Reject a video without frames and frame rate or encoded data."""
    with pytest.raises(ValueError, match='Provide either'):
        Video(np.zeros((1, 2, 2, 3), dtype=np.uint8))


def test_video_round_trips_through_bytes(tmp_path, audio):
    """Encode frames and audio as MP4 and decode them back."""
    frames = np.full((25, 16, 16, 3), 120, dtype=np.uint8)
    video = Video(frames, fps=25, audio=audio)
    assert repr(video) == 'Video(duration=1.00s, fps=25)'
    decoded = Video(data=video.save(tmp_path / 'video.mp4').read_bytes())
    assert repr(decoded).startswith('Video(encoded_bytes=')
    assert decoded.frames.shape == (25, 16, 16, 3)
    assert decoded.fps == 25
    assert decoded.duration == 1.0
    assert decoded.audio is not None
    assert decoded.audio.duration == pytest.approx(1.0, abs=0.1)
    assert abs(int(decoded.frames[0, 0, 0, 0]) - 120) < 5


def test_video_without_audio_decodes_to_no_track():
    """Decode a video without an audio track to `audio=None`."""
    encoded = Video(np.zeros((5, 8, 8, 3), dtype=np.uint8), fps=5).to_bytes()
    assert Video(data=encoded).audio is None


def test_video_audio_track_is_mono(audio):
    """Return the audio track mixed to mono."""
    stereo = Audio(np.stack([audio.samples, audio.samples], axis=1), audio.sample_rate)
    track = Video(np.zeros((25, 4, 4, 3), dtype=np.uint8), fps=25, audio=stereo).audio_track()
    assert track.samples.ndim == 1
    assert track.duration == 1.0


def test_video_without_audio_gives_silence():
    """Return silence as long as the video when it has no audio track."""
    track = Video(np.zeros((50, 4, 4, 3), dtype=np.uint8), fps=25).audio_track()
    assert track.duration == 2.0
    assert not track.samples.any()
