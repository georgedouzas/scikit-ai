"""Components that generate and edit images, speech and video."""

from ._animator import ImageAnimator
from ._avatar import TalkingAvatarGenerator
from ._editor import ImageEditor
from ._image import ImageGenerator
from ._speech import SpeechSynthesizer
from ._video import VideoGenerator

__all__: list[str] = [
    'ImageAnimator',
    'ImageEditor',
    'ImageGenerator',
    'SpeechSynthesizer',
    'TalkingAvatarGenerator',
    'VideoGenerator',
]
