"""Components that transcribe the speech of audio and videos."""

from ._speech import SpeechTranscriber
from ._video import VideoTranscriber

__all__: list[str] = [
    'SpeechTranscriber',
    'VideoTranscriber',
]
