"""Components that describe images, audio and videos in natural language."""

from ._audio import AudioDescriptor
from ._image import ImageDescriptor
from ._video import VideoDescriptor

__all__: list[str] = [
    'AudioDescriptor',
    'ImageDescriptor',
    'VideoDescriptor',
]
