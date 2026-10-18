"""Components that map texts, images, audio and videos to vectors."""

from ._audio import AudioEmbedder
from ._image import ImageEmbedder
from ._text import TextEmbedder
from ._video import VideoEmbedder

__all__: list[str] = [
    'AudioEmbedder',
    'ImageEmbedder',
    'TextEmbedder',
    'VideoEmbedder',
]
