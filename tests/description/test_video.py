"""Test the video descriptor."""

import numpy as np

from skai.core import Audio, Video
from skai.description import VideoDescriptor


def test_predict_describes_videos():
    """Return one description per video from its sampled frames."""
    video = Video(np.zeros((40, 8, 8, 3), dtype=np.uint8), fps=10, audio=Audio(np.zeros(64000), 16000))
    assert VideoDescriptor(n_frames=8, backend='dummy').fit().predict([video]).tolist() == [
        'A brief description of 8 frames of a 4.00 second video.',
    ]


def test_predict_describes_video_without_audio():
    """Describe a video without an audio track from its frames alone."""
    video = Video(np.zeros((20, 8, 8, 3), dtype=np.uint8), fps=10)
    assert VideoDescriptor(n_frames=4, backend='dummy').fit().predict([video])[0].startswith('A brief description')
