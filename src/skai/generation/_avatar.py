"""Animate a portrait so that it speaks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os import PathLike
from pathlib import PurePath
from typing import Any, ClassVar, Literal

import numpy as np
from PIL import Image

from skai.core import Audio, AvatarRequest, Backend, BaseMediaGenerator, read_image


class TalkingAvatarGenerator(BaseMediaGenerator):
    """Animate a fixed portrait so that it speaks every audio sample, with lip-synced video and the audio as its track.

    Args:
        avatar:
            The portrait to animate, as a path, a URL, bytes, a PIL image or a NumPy array. It is required.

        output_format:
            `'object'` returns `Video` objects, `'bytes'` returns MP4 bytes, and `'path'` writes MP4 files and returns
            their paths. `'bytes'` and `'path'` need the `video` extra for videos that are not already encoded.

        output_dir:
            The directory for `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The Replicate backend with an
            open-source talking-head model is used when it is `None`.

        batch_size:
            The number of audio samples submitted to the backend at once.

        on_error:
            `'raise'` raises when a sample fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        avatar_ (PIL.Image.Image):
            The portrait read from `avatar`, converted to RGB, cropped to a centered square and resized to the input
            size of the backend.

    Examples:
        >>> import numpy as np
        >>> from PIL import Image
        >>> from skai.generation import TalkingAvatarGenerator
        >>> generator = TalkingAvatarGenerator(avatar=Image.new('RGB', (64, 64)), backend='dummy').fit()
        >>> video = generator.predict([(np.zeros(32000), 16000)])[0]
        >>> video.duration
        2.0
    """

    task: ClassVar[str] = 'animate_avatar'
    input_kind: ClassVar[Literal['audio']] = 'audio'
    output_kind: ClassVar[Literal['video']] = 'video'
    default_backend: ClassVar[str] = 'replicate'
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'avatar': [str, bytes, Image.Image, np.ndarray, PurePath],
    }

    def __init__(
        self,
        *,
        avatar: str | bytes | Image.Image | np.ndarray | PurePath | None = None,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 2,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.avatar = avatar
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Read the portrait, crop it to a centered square and resize it to the input size of the backend."""
        portrait = read_image(self.avatar).convert('RGB')
        side = min(portrait.size)
        left, top = (portrait.width - side) // 2, (portrait.height - side) // 2
        portrait = portrait.crop((left, top, left + side, top + side))
        size = self.backend_.avatar_size
        self.avatar_ = portrait if size is None else portrait.resize((size, size))

    def _build_request(self, sample: Audio, seed: int) -> Any:
        """Build the avatar request of one audio sample."""
        return AvatarRequest(self.avatar_, sample, seed)
