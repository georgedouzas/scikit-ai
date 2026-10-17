"""Test the video transcriber."""

import numpy as np

from skai.core import Audio, Video
from skai.transcription import VideoTranscriber

SPEAKING = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10, audio=Audio(np.zeros(32000), 16000))
SILENT = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10)


def test_predict_transcribes_audio_track():
    """Return the transcript of the audio track."""
    assert VideoTranscriber(backend='dummy').fit().predict([SPEAKING]).tolist() == ['Speech of 2.00 seconds.']


def test_timestamps_follow_the_video():
    """Measure segment times from the start of the video."""
    records = VideoTranscriber(timestamps='segment', backend='dummy').fit().predict([SPEAKING])[0]
    assert records == [{'start': 0.0, 'end': 2.0, 'text': 'Speech of 2.00 seconds.'}]


def test_silent_video_gives_empty_transcript():
    """Return an empty transcript for a video without an audio track."""
    assert VideoTranscriber(backend='dummy').fit().predict([SILENT, SPEAKING]).tolist()[0] == ''
