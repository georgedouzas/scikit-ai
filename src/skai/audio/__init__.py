"""Components whose input is audio."""

from ._avatar import TalkingAvatarGenerator
from ._transcriber import SpeechTranscriber

__all__: list[str] = ['SpeechTranscriber', 'TalkingAvatarGenerator']
