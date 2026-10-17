"""Components that generate images, speech and video."""

from ._avatar import TalkingAvatarGenerator
from ._image import ImageGenerator
from ._speech import SpeechSynthesizer

__all__: list[str] = ['ImageGenerator', 'SpeechSynthesizer', 'TalkingAvatarGenerator']
