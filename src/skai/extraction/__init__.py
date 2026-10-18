"""Components that extract structured records from texts, images, audio, videos and multimodal samples."""

from ._audio import AudioExtractor
from ._image import ImageExtractor
from ._multimodal import MultimodalExtractor
from ._text import TextExtractor
from ._video import VideoExtractor

__all__: list[str] = [
    'AudioExtractor',
    'ImageExtractor',
    'MultimodalExtractor',
    'TextExtractor',
    'VideoExtractor',
]
