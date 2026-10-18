"""Generate audio from texts, audio, videos and multimodal samples."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Real
from os import PathLike
from pathlib import PurePath
from typing import Any, ClassVar, Literal

import numpy as np
from sklearn.utils._param_validation import Interval, InvalidParameterError, StrOptions

from skai.core import (
    Audio,
    AudioGenerationRequest,
    AudioOperationRequest,
    Backend,
    BaseMediaGenerator,
    InvalidSampleError,
    SpeechRequest,
    UnsupportedBackendError,
    Video,
    concatenate_audio,
    extract_audio,
    list_backends,
    mix_audio,
    place_audio,
    read_audio,
    read_video,
)

JOIN_GAP_SECONDS = 0.3
MAX_VOICE_SECONDS = 30.0
VoiceSource = str | Audio | tuple | bytes | PurePath | list


def _join_recordings(recordings: list[Audio]) -> Audio:
    """Join recordings into one mono reference at the first recording's rate, with short gaps, capped in length."""
    rate = recordings[0].sample_rate
    gap = np.zeros(round(JOIN_GAP_SECONDS * rate), dtype=np.float32)
    parts = []
    for recording in recordings:
        parts += [recording.resample(rate).samples.astype(np.float32), gap]
    samples = np.concatenate(parts[:-1])[: round(MAX_VOICE_SECONDS * rate)]
    return Audio(samples, rate)


def _learn_voice(voice: VoiceSource | None, owner: str) -> str | Audio | None:
    """Return a voice name as is, or the joined reference audio of one recording or a list of recordings."""
    if voice is None or (isinstance(voice, str) and not PurePath(voice).suffix):
        return voice
    if isinstance(voice, list) and not voice:
        msg = f'The `voice` parameter of {owner} must hold at least one recording.'
        raise ValueError(msg)
    recordings = voice if isinstance(voice, list) else [voice]
    return _join_recordings([read_audio(recording, index) for index, recording in enumerate(recordings)])


def _as_track(value: Any, key: str, index: int) -> Audio:
    """Return a named track as audio, placing a list of timed records at their starts."""
    if not isinstance(value, list):
        return _as_audio(value, key, index)
    clips = []
    for record in value:
        if not isinstance(record, dict) or 'start' not in record or 'media' not in record:
            msg = f'Sample {index}: every record of the `{key}` track needs `start` and `media`.'
            raise InvalidSampleError(msg)
        media = record['media']
        clip = _as_audio(read_audio(media, index) if isinstance(media, tuple | bytes) else media, key, index)
        clips.append((float(record['start']), clip))
    return place_audio(clips)


def _as_audio(value: Any, key: str, index: int) -> Audio:
    """Return a named part of a multimodal sample as audio, taking the track of a video."""
    if isinstance(value, Video):
        return value.audio_track()
    if isinstance(value, Audio):
        return value
    msg = f'Sample {index}: the `{key}` part must be audio or a video, but got {type(value).__name__}.'
    raise InvalidSampleError(msg)


class TextToAudioGenerator(BaseMediaGenerator):
    """Return one audio sample per text: the text spoken, or music or a sound it describes.

    With `operation='speech'`, `voice` is a voice name of the backend, or one recording or a list of recordings of the
    voice to clone, which `fit` joins into `voice_`.

    Args:
        operation:
            `'speech'`, `'music'` or `'sound'`.

        voice:
            A voice name, or one recording or a list of recordings of the target voice.

        language:
            The language of the text.

        duration:
            The length of music and sounds in seconds.

        output_format:
            `'object'` returns `Audio` objects, `'bytes'` WAV bytes, and `'path'` the paths of new WAV files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of texts submitted to the backend at once.

        on_error:
            `'raise'` raises when a text fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        voice_ (str | Audio | None):
            The voice name, or the joined reference audio of the voice to clone.

    Examples:
        >>> from skai.generation import TextToAudioGenerator
        >>> generator = TextToAudioGenerator(operation='music', duration=5, backend='dummy').fit()
        >>> generator.predict(['calm piano'])[0].duration
        5.0
    """

    input_kind: ClassVar[Literal['text']] = 'text'
    output_kind: ClassVar[Literal['audio']] = 'audio'
    operations: ClassVar[dict[str, str | None]] = {
        'speech': 'synthesize_speech',
        'music': 'generate_music',
        'sound': 'generate_sound',
    }
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'voice': [str, bytes, tuple, list, Audio, PurePath, None],
        'language': [str, None],
        'duration': [Interval(Real, 0, None, closed='neither')],
    }

    def __init__(
        self,
        *,
        operation: Literal['speech', 'music', 'sound'] = 'speech',
        voice: VoiceSource | None = None,
        language: str | None = None,
        duration: float = 10.0,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 16,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.voice = voice
        self.language = language
        self.duration = duration
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Learn the voice and check that the backend can clone it."""
        assert self.backend_ is not None
        self.voice_ = _learn_voice(self.voice, type(self).__name__) if self.operation == 'speech' else None
        if isinstance(self.voice_, Audio) and not self.backend_.supports_voice_cloning:
            backends = list_backends(self.task_, lambda backend_class: backend_class.supports_voice_cloning)
            msg = f'{type(self.backend_).__name__} cannot clone voices. Backends that can: {backends}.'
            raise UnsupportedBackendError(msg)

    def _build_request(self, sample: str, seed: int) -> Any:
        """Build the request of one text for the operation."""
        if self.operation == 'speech':
            return SpeechRequest(sample, self.voice_, self.language, seed)
        return AudioGenerationRequest(sample, float(self.duration), seed)


class AudioToAudioGenerator(BaseMediaGenerator):
    """Return one audio sample per audio sample: a separated stem, a denoised copy, another voice, or a joined list.

    Args:
        operation:
            `'separate'` keeps `stem`, `'enhance'` removes noise, `'convert_voice'` says the same speech in `voice`, and
            `'concatenate'` joins a list of audio samples with `gap` seconds between them, locally.

        stem:
            `'vocals'` or `'accompaniment'`, the stem kept by a separation.

        voice:
            One recording or a list of recordings of the target voice. Required with `operation='convert_voice'`.

        gap:
            The seconds of silence between joined audio samples.

        output_format:
            `'object'` returns `Audio` objects, `'bytes'` WAV bytes, and `'path'` the paths of new WAV files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`, and an open-source model on Replicate for voice conversion. Concatenation needs
            no backend.

        batch_size:
            The number of audio samples submitted to the backend at once.

        on_error:
            `'raise'` raises when a sample fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend | None):
            The resolved backend, or `None` for concatenation.

        voice_ (Audio | None):
            The joined reference audio of the target voice.

    Examples:
        >>> import numpy as np
        >>> from skai.generation import AudioToAudioGenerator
        >>> song = (np.zeros(32000), 16000)
        >>> AudioToAudioGenerator(operation='separate', backend='dummy').fit().predict([song])[0].duration
        2.0
    """

    input_kind: ClassVar[Literal['audio']] = 'audio'
    output_kind: ClassVar[Literal['audio']] = 'audio'
    operations: ClassVar[dict[str, str | None]] = {
        'separate': 'separate_audio',
        'enhance': 'enhance_audio',
        'convert_voice': 'convert_voice',
        'concatenate': None,
    }
    default_backends: ClassVar[dict[str, str]] = {'convert_voice': 'replicate'}
    list_operations: ClassVar[frozenset[str]] = frozenset({'concatenate'})
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'stem': [StrOptions({'vocals', 'accompaniment'})],
        'voice': [bytes, tuple, list, Audio, PurePath, str, None],
        'gap': [Interval(Real, 0, None, closed='left')],
    }

    def __init__(
        self,
        *,
        operation: Literal['separate', 'enhance', 'convert_voice', 'concatenate'] = 'separate',
        stem: Literal['vocals', 'accompaniment'] = 'vocals',
        voice: VoiceSource | None = None,
        gap: float = 0.0,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.stem = stem
        self.voice = voice
        self.gap = gap
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Learn the target voice of voice conversion."""
        self.voice_: Audio | None = None
        if self.operation != 'convert_voice':
            return
        if self.voice is None:
            msg = "The 'voice' parameter of AudioToAudioGenerator is required with `operation='convert_voice'`."
            raise InvalidParameterError(msg)
        voice = _learn_voice(self.voice, type(self).__name__)
        if not isinstance(voice, Audio):
            msg = 'The `voice` parameter of AudioToAudioGenerator must be one recording or a list of recordings.'
            raise InvalidParameterError(msg)
        self.voice_ = voice

    def _build_request(self, sample: Audio, seed: int) -> AudioOperationRequest:
        """Build the request of one audio sample for the operation."""
        return AudioOperationRequest(sample, self.stem, self.voice_, seed)

    def _run_local(self, sample: list[Audio] | Audio, seed: int) -> Audio:
        """Join a list of audio samples."""
        return concatenate_audio(sample if isinstance(sample, list) else [sample], float(self.gap))


class VideoToAudioGenerator(BaseMediaGenerator):
    """Return the mono audio track of every video, or silence as long as a video without one, locally.

    Args:
        operation:
            `'extract'`, the only operation.

        output_format:
            `'object'` returns `Audio` objects, `'bytes'` WAV bytes, and `'path'` the paths of new WAV files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        on_error:
            `'raise'` raises when a video fails, and `'ignore'` returns `None` for it and warns.

    Attributes:
        backend_ (None):
            Always `None`, since extraction runs locally.

    Examples:
        >>> import numpy as np
        >>> from skai.core import Video
        >>> from skai.generation import VideoToAudioGenerator
        >>> video = Video(np.zeros((8, 8, 8, 3), dtype=np.uint8), fps=4)
        >>> VideoToAudioGenerator().fit().predict([video])[0].duration
        2.0
    """

    input_kind: ClassVar[Literal['video']] = 'video'
    output_kind: ClassVar[Literal['audio']] = 'audio'
    operations: ClassVar[dict[str, str | None]] = {'extract': None}

    def __init__(
        self,
        *,
        operation: Literal['extract'] = 'extract',
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        on_error: Literal['raise', 'ignore'] = 'raise',
    ) -> None:
        self.operation = operation
        self.output_format = output_format
        self.output_dir = output_dir
        self.on_error = on_error

    def _read_one(self, sample: Any, index: int) -> Video:
        """Read the whole video."""
        return read_video(sample, index)

    def _run_local(self, sample: Video, seed: int) -> Audio:
        """Return the audio track of one video."""
        return extract_audio(sample)


class MultimodalToAudioGenerator(BaseMediaGenerator):
    """Mix the named tracks of every multimodal sample, or speak its text in its own voice.

    With `operation='mix'` (the default), a sample is a dict, or a DataFrame row, of named tracks, mixed locally into
    one track as long as the longest. A track is audio, a video whose audio track is used, or a list of timed records
    with `start` and `media`, such as segmenter regions, each placed at its start. With `operation='speech'`, the
    backend speaks the `text` of every sample in its `voice`, a voice name or reference audio of the voice to clone.

    Args:
        operation:
            `'mix'` or `'speech'`.

        volumes:
            The gain of every track by name, one when it is missing.

        duck_under:
            The names of the tracks to lower while `duck_for` sounds, such as `['music']`.

        duck_for:
            The name of the track whose sound lowers the others, such as `'voice'`.

        duck_level:
            The gain of the lowered tracks while `duck_for` sounds.

        language:
            The language of the texts spoken with `operation='speech'`.

        output_format:
            `'object'` returns `Audio` objects, `'bytes'` WAV bytes, and `'path'` the paths of new WAV files.

        output_dir:
            The directory of `'path'` outputs. A new temporary directory is used when it is `None`.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`. Mixing needs no backend.

        batch_size:
            The number of samples submitted to the backend at once.

        on_error:
            `'raise'` raises when a sample fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend | None):
            The resolved backend, or `None` for mixing.

    Examples:
        >>> import numpy as np
        >>> from skai.generation import MultimodalToAudioGenerator
        >>> sample = {'voice': (np.zeros(16000), 16000), 'music': (np.zeros(32000), 16000)}
        >>> MultimodalToAudioGenerator(duck_under=['music'], duck_for='voice').fit().predict([sample])[0].duration
        2.0
    """

    input_kind: ClassVar[Literal['multimodal']] = 'multimodal'
    output_kind: ClassVar[Literal['audio']] = 'audio'
    operations: ClassVar[dict[str, str | None]] = {'mix': None, 'speech': 'synthesize_speech'}
    required_keys: ClassVar[dict[str, tuple[str, ...]]] = {'speech': ('text',)}
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'volumes': [dict, None],
        'duck_under': [list, None],
        'duck_for': [str, None],
        'duck_level': [Interval(Real, 0, 1, closed='both')],
        'language': [str, None],
    }

    def __init__(
        self,
        *,
        operation: Literal['mix', 'speech'] = 'mix',
        volumes: dict[str, float] | None = None,
        duck_under: list[str] | None = None,
        duck_for: str | None = None,
        duck_level: float = 0.25,
        language: str | None = None,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 16,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.volumes = volumes
        self.duck_under = duck_under
        self.duck_for = duck_for
        self.duck_level = duck_level
        self.language = language
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _read_one(self, sample: Any, index: int) -> dict[str, Any]:
        """Read every named part, as tracks for mixing or as a text and a voice for speech."""
        parts = super()._read_one(sample, index)
        if self.operation == 'speech':
            voice = parts.get('voice')
            if voice is not None and not isinstance(voice, str):
                parts['voice'] = _as_audio(voice, 'voice', index)
            return parts
        if self.duck_for is not None and self.duck_for not in parts:
            msg = f'Sample {index} has no `{self.duck_for}` key, which `duck_for` names.'
            raise InvalidSampleError(msg)
        return {key: _as_track(value, key, index) for key, value in parts.items()}

    def _build_request(self, sample: dict[str, Any], seed: int) -> SpeechRequest:
        """Build the speech request of one sample, checking that the backend can clone a reference voice."""
        assert self.backend_ is not None
        voice = sample.get('voice')
        if isinstance(voice, Audio) and not self.backend_.supports_voice_cloning:
            backends = list_backends(self.task_, lambda backend_class: backend_class.supports_voice_cloning)
            msg = f'{type(self.backend_).__name__} cannot clone voices. Backends that can: {backends}.'
            raise UnsupportedBackendError(msg)
        return SpeechRequest(str(sample['text']), voice, self.language, seed)

    def _run_local(self, sample: dict[str, Audio], seed: int) -> Audio:
        """Mix the tracks of one sample."""
        return mix_audio(
            sample,
            volumes=self.volumes,
            duck_under=self.duck_under,
            duck_for=self.duck_for,
            duck_level=float(self.duck_level),
        )
