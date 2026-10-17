"""Components that assign labels to texts, images, audio and videos."""

from ._audio import AudioClassifier
from ._image import ImageClassifier
from ._text import TextClassifier
from ._video import VideoClassifier

__all__: list[str] = [
    'AudioClassifier',
    'ImageClassifier',
    'TextClassifier',
    'VideoClassifier',
]
