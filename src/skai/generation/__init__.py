"""Components that generate texts, images, audio and videos from every input kind."""

from ._audio import AudioToAudioGenerator, MultimodalToAudioGenerator, TextToAudioGenerator, VideoToAudioGenerator
from ._image import ImageToImageGenerator, MultimodalToImageGenerator, TextToImageGenerator, VideoToImageGenerator
from ._text import (
    AudioToTextGenerator,
    ImageToTextGenerator,
    MultimodalToTextGenerator,
    TextToTextGenerator,
    VideoToTextGenerator,
)
from ._video import (
    AudioToVideoGenerator,
    ImageToVideoGenerator,
    MultimodalToVideoGenerator,
    TextToVideoGenerator,
    VideoToVideoGenerator,
)

__all__: list[str] = [
    'AudioToAudioGenerator',
    'AudioToTextGenerator',
    'AudioToVideoGenerator',
    'ImageToImageGenerator',
    'ImageToTextGenerator',
    'ImageToVideoGenerator',
    'MultimodalToAudioGenerator',
    'MultimodalToImageGenerator',
    'MultimodalToTextGenerator',
    'MultimodalToVideoGenerator',
    'TextToAudioGenerator',
    'TextToImageGenerator',
    'TextToTextGenerator',
    'TextToVideoGenerator',
    'VideoToAudioGenerator',
    'VideoToImageGenerator',
    'VideoToTextGenerator',
    'VideoToVideoGenerator',
]
