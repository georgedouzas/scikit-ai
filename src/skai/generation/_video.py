"""Generate videos from texts, images, audio, videos and multimodal samples."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral, Real
from os import PathLike
from pathlib import PurePath
from typing import Any, ClassVar, Literal

import numpy as np
from PIL import Image
from sklearn.utils._param_validation import Interval

from skai.core import (
    AnimationRequest,
    Audio,
    AvatarRequest,
    Backend,
    BaseMediaGenerator,
    InvalidSampleError,
    LipSyncRequest,
    MultiRequest,
    Region,
    RegionRequest,
    Video,
    VideoGenerationRequest,
    check_size,
    concatenate_video,
    crop_to_aspect,
    mux,
    read_image,
    read_video,
    sample_frames,
    subtitle,
)


class TextToVideoGenerator(BaseMediaGenerator):
    """Create one video per prompt, of a duration and size.

    Args:
        operation:
            `'generate'`, the only operation.

        duration:
            The length of every video in seconds.

        size:
            The width and height of every video in pixels.

        fps:
            The number of frames per second, or `None` for the model's default.

        style:
            A description of the visual style.

        negative_prompt:
            A description of what the videos must not show.

        output_format:
            `'object'` returns `Video` objects, `'bytes'` MP4 bytes, and `'path'` the paths of new MP4 files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of prompts submitted to the backend at once.

        on_error:
            `'raise'` raises when a prompt fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

    Examples:
        >>> from skai.generation import TextToVideoGenerator
        >>> generator = TextToVideoGenerator(duration=2, size=(320, 240), fps=8, backend='dummy').fit()
        >>> generator.predict(['a boat at sea'])[0].duration
        2.0
    """

    input_kind: ClassVar[Literal['text']] = 'text'
    output_kind: ClassVar[Literal['video']] = 'video'
    operations: ClassVar[dict[str, str | None]] = {'generate': 'generate_video'}
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'duration': [Interval(Real, 0, None, closed='neither')],
        'size': [tuple],
        'fps': [Interval(Integral, 1, None, closed='left'), None],
        'style': [str, None],
        'negative_prompt': [str, None],
    }

    def __init__(
        self,
        *,
        operation: Literal['generate'] = 'generate',
        duration: float = 4.0,
        size: tuple[int, int] = (1280, 720),
        fps: int | None = None,
        style: str | None = None,
        negative_prompt: str | None = None,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 1,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.duration = duration
        self.size = size
        self.fps = fps
        self.style = style
        self.negative_prompt = negative_prompt
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Check the size."""
        check_size(self.size, type(self).__name__)

    def _build_request(self, sample: str, seed: int) -> VideoGenerationRequest:
        """Build the video generation request of one prompt."""
        size = check_size(self.size, type(self).__name__)
        return VideoGenerationRequest(
            sample,
            float(self.duration),
            size,
            self.fps,
            self.style,
            self.negative_prompt,
            seed,
        )


class ImageToVideoGenerator(BaseMediaGenerator):
    """Turn every image into a short video that starts from it, with optional instructions for the motion.

    Args:
        operation:
            `'animate'`, the only operation.

        instructions:
            A description of the motion.

        duration:
            The length of every video in seconds.

        fps:
            The number of frames per second, or `None` for the model's default.

        output_format:
            `'object'` returns `Video` objects, `'bytes'` MP4 bytes, and `'path'` the paths of new MP4 files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of images submitted to the backend at once.

        on_error:
            `'raise'` raises when an image fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

    Examples:
        >>> from PIL import Image
        >>> from skai.generation import ImageToVideoGenerator
        >>> generator = ImageToVideoGenerator(duration=1, fps=8, backend='dummy').fit()
        >>> generator.predict([Image.new('RGB', (64, 48))])[0].duration
        1.0
    """

    input_kind: ClassVar[Literal['image']] = 'image'
    output_kind: ClassVar[Literal['video']] = 'video'
    operations: ClassVar[dict[str, str | None]] = {'animate': 'animate_image'}
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'instructions': [str, None],
        'duration': [Interval(Real, 0, None, closed='neither')],
        'fps': [Interval(Integral, 1, None, closed='left'), None],
    }

    def __init__(
        self,
        *,
        operation: Literal['animate'] = 'animate',
        instructions: str | None = None,
        duration: float = 2.0,
        fps: int | None = None,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 1,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.instructions = instructions
        self.duration = duration
        self.fps = fps
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _build_request(self, sample: Image.Image, seed: int) -> AnimationRequest:
        """Build the animation request of one image."""
        return AnimationRequest(sample, self.instructions, float(self.duration), self.fps, seed)


class AudioToVideoGenerator(BaseMediaGenerator):
    """Make a portrait speak every audio sample, as a talking head.

    `fit` prepares `avatar_` from `avatar`: converted to RGB, center-cropped to a square, and resized to the input size
    of the backend.

    Args:
        operation:
            `'talking_head'`, the only operation.

        avatar:
            The portrait, as a path, URL, bytes, PIL image or array.

        output_format:
            `'object'` returns `Video` objects, `'bytes'` MP4 bytes, and `'path'` the paths of new MP4 files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. SadTalker on Replicate is used
            when it is `None`.

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
            The prepared portrait.

    Examples:
        >>> import numpy as np
        >>> from PIL import Image
        >>> from skai.generation import AudioToVideoGenerator
        >>> generator = AudioToVideoGenerator(avatar=Image.new('RGB', (640, 480)), backend='dummy').fit()
        >>> generator.predict([(np.zeros(16000), 16000)])[0].duration
        1.0
    """

    input_kind: ClassVar[Literal['audio']] = 'audio'
    output_kind: ClassVar[Literal['video']] = 'video'
    default_backend: ClassVar[str] = 'replicate'
    operations: ClassVar[dict[str, str | None]] = {'talking_head': 'animate_avatar'}
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'avatar': [str, bytes, Image.Image, np.ndarray, PurePath],
    }

    def __init__(
        self,
        *,
        operation: Literal['talking_head'] = 'talking_head',
        avatar: str | bytes | Image.Image | np.ndarray | PurePath | None = None,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 2,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.avatar = avatar
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Read the portrait, crop it to a centered square and resize it to the input size of the backend."""
        assert self.backend_ is not None
        portrait = read_image(self.avatar).convert('RGB')
        side = min(portrait.size)
        left, top = (portrait.width - side) // 2, (portrait.height - side) // 2
        portrait = portrait.crop((left, top, left + side, top + side))
        size = self.backend_.avatar_size
        self.avatar_ = portrait if size is None else portrait.resize((size, size))

    def _build_request(self, sample: Audio, seed: int) -> AvatarRequest:
        """Build the avatar request of one audio sample."""
        return AvatarRequest(self.avatar_, sample, seed)


class VideoToVideoGenerator(BaseMediaGenerator):
    """Return one video per video, reframed to an aspect ratio following its main subject, or a joined list of clips.

    Reframing detects `subject` in `n_frames` sampled frames with the backend and crops every frame around the
    detected centers, interpolated over time. Concatenation joins a list of clips locally, at the size and frame rate
    of the first clip.

    Args:
        operation:
            `'reframe'` or `'concatenate'`.

        aspect_ratio:
            The width divided by the height of reframed videos, such as `9 / 16` for vertical video.

        subject:
            The kind of subject that reframing follows.

        n_frames:
            The number of frames in which reframing detects the subject.

        output_format:
            `'object'` returns `Video` objects, `'bytes'` MP4 bytes, and `'path'` the paths of new MP4 files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`. Concatenation needs no backend.

        batch_size:
            The number of videos submitted to the backend at once.

        on_error:
            `'raise'` raises when a video fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend | None):
            The resolved backend, or `None` for concatenation.

    Examples:
        >>> import numpy as np
        >>> from skai.core import Video
        >>> from skai.generation import VideoToVideoGenerator
        >>> clip = Video(np.zeros((4, 8, 8, 3), dtype=np.uint8), fps=4)
        >>> VideoToVideoGenerator(operation='concatenate').fit().predict([[clip, clip, clip]])[0].duration
        3.0
    """

    input_kind: ClassVar[Literal['video']] = 'video'
    output_kind: ClassVar[Literal['video']] = 'video'
    operations: ClassVar[dict[str, str | None]] = {'reframe': 'detect_objects', 'concatenate': None}
    list_operations: ClassVar[frozenset[str]] = frozenset({'concatenate'})
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'aspect_ratio': [Interval(Real, 0, None, closed='neither')],
        'subject': [str],
        'n_frames': [Interval(Integral, 1, None, closed='left')],
    }

    def __init__(
        self,
        *,
        operation: Literal['reframe', 'concatenate'] = 'reframe',
        aspect_ratio: float = 9 / 16,
        subject: str = 'person',
        n_frames: int = 8,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 2,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.aspect_ratio = aspect_ratio
        self.subject = subject
        self.n_frames = n_frames
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _read_one(self, sample: Any, index: int) -> Video:
        """Read the whole video."""
        return read_video(sample, index)

    def _build_request(self, sample: Video, seed: int) -> MultiRequest:
        """Build one detection request per sampled frame."""
        frames = sample_frames(sample, self.n_frames).frames
        return MultiRequest(RegionRequest(frame, (self.subject,)) for frame in frames)

    def _complete(self, result: list[list[Region]], sample: Video) -> Video:
        """Crop the video around the most confident subject of every sampled frame."""
        width = sample.frames.shape[2]
        centers = [
            (best.box[0] + best.box[2]) / 2 / width
            for regions in result
            if regions
            for best in [max(regions, key=lambda region: region.score)]
        ]
        return crop_to_aspect(sample, float(self.aspect_ratio), centers)

    def _run_local(self, sample: list[Video] | Video, seed: int) -> Video:
        """Join a list of clips."""
        return concatenate_video(sample if isinstance(sample, list) else [sample])


class MultimodalToVideoGenerator(BaseMediaGenerator):
    """Return one video per multimodal sample: lip-synced to new speech, with a new soundtrack, or subtitled.

    A sample is a dict, or a DataFrame row, of named parts. `'lip_sync'` makes `video` speak `audio` with the backend,
    `'mux'` replaces the soundtrack of `video` with `audio`, and `'subtitle'` shows `transcript` as subtitles, both
    locally. A transcript is a string, or a list of timed records with `start`, `end` and `text`, such as the output
    of `AudioSegmenter`.

    Args:
        operation:
            `'lip_sync'`, `'mux'` or `'subtitle'`.

        burn:
            Whether subtitles are drawn on the frames. Otherwise they are attached as `subtitles` in SRT format and
            written next to the video.

        output_format:
            `'object'` returns `Video` objects, `'bytes'` MP4 bytes, and `'path'` the paths of new MP4 files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. An open-source model on
            Replicate is used for lip sync when it is `None`. Muxing and subtitles need no backend.

        batch_size:
            The number of samples submitted to the backend at once.

        on_error:
            `'raise'` raises when a sample fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend | None):
            The resolved backend, or `None` for muxing and subtitles.

    Examples:
        >>> import numpy as np
        >>> from skai.core import Video
        >>> from skai.generation import MultimodalToVideoGenerator
        >>> sample = {'video': Video(np.zeros((8, 8, 8, 3), dtype=np.uint8), fps=4), 'audio': (np.zeros(8000), 16000)}
        >>> MultimodalToVideoGenerator(operation='mux').fit().predict([sample])[0].audio.duration
        2.0
    """

    input_kind: ClassVar[Literal['multimodal']] = 'multimodal'
    output_kind: ClassVar[Literal['video']] = 'video'
    default_backends: ClassVar[dict[str, str]] = {'lip_sync': 'replicate'}
    operations: ClassVar[dict[str, str | None]] = {'lip_sync': 'lip_sync', 'mux': None, 'subtitle': None}
    required_keys: ClassVar[dict[str, tuple[str, ...]]] = {
        'lip_sync': ('video', 'audio'),
        'mux': ('video', 'audio'),
        'subtitle': ('video', 'transcript'),
    }
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'burn': ['boolean'],
    }

    def __init__(
        self,
        *,
        operation: Literal['lip_sync', 'mux', 'subtitle'] = 'lip_sync',
        burn: bool = True,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 1,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.burn = burn
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _read_one(self, sample: Any, index: int) -> dict[str, Any]:
        """Read the parts of one sample, checking that the video is a video and the audio is audio."""
        parts = super()._read_one(sample, index)
        video = parts['video']
        if not isinstance(video, Video):
            video = read_video(video, index) if isinstance(video, bytes | str | PurePath) else video
        if not isinstance(video, Video):
            msg = f'Sample {index}: the `video` part must be a video, but got {type(video).__name__}.'
            raise InvalidSampleError(msg)
        parts['video'] = video
        if 'audio' in parts and isinstance(parts['audio'], Video):
            parts['audio'] = parts['audio'].audio_track()
        if self.operation != 'subtitle' and not isinstance(parts['audio'], Audio):
            msg = f'Sample {index}: the `audio` part must be audio, but got {type(parts["audio"]).__name__}.'
            raise InvalidSampleError(msg)
        return parts

    def _build_request(self, sample: dict[str, Any], seed: int) -> LipSyncRequest:
        """Build the lip sync request of one sample."""
        return LipSyncRequest(sample['video'], sample['audio'], seed)

    def _run_local(self, sample: dict[str, Any], seed: int) -> Video:
        """Replace the soundtrack or add the subtitles of one sample."""
        if self.operation == 'mux':
            return mux(sample['video'], sample['audio'])
        return subtitle(sample['video'], sample['transcript'], self.burn)
