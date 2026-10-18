"""Components that assign labels to texts, images, audio, videos and multimodal samples."""

from ._audio import AudioClassifier
from ._image import ImageClassifier
from ._multimodal import MultimodalClassifier
from ._text import TextClassifier
from ._video import VideoClassifier

__all__: list[str] = [
    'AudioClassifier',
    'ImageClassifier',
    'MultimodalClassifier',
    'TextClassifier',
    'VideoClassifier',
]
