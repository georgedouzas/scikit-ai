"""Speak texts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from os import PathLike
from pathlib import PurePath
from typing import Any, ClassVar, Literal

import numpy as np

from skai.core import (
    Audio,
    Backend,
    BaseMediaGenerator,
    SpeechRequest,
    UnsupportedBackendError,
    list_backends,
    read_audio,
)

JOIN_GAP_SECONDS = 0.3
MAX_VOICE_SECONDS = 30.0


def _join_recordings(recordings: list[Audio]) -> Audio:
    """Join recordings into one mono reference at the first recording's rate, with short gaps, capped in length."""
    rate = recordings[0].sample_rate
    gap = np.zeros(round(JOIN_GAP_SECONDS * rate), dtype=np.float32)
    parts = []
    for recording in recordings:
        mono = recording.to_mono()
        times = np.arange(round(mono.duration * rate)) / rate
        source_times = np.arange(mono.samples.shape[0]) / mono.sample_rate
        parts += [np.interp(times, source_times, mono.samples).astype(np.float32), gap]
    samples = np.concatenate(parts[:-1])[: round(MAX_VOICE_SECONDS * rate)]
    return Audio(samples, rate)


class SpeechSynthesizer(BaseMediaGenerator):
    """Convert every text into spoken audio.

    Args:
        voice:
            A string is a voice name of the backend. Reference audio of a voice to clone, given as one recording or a
            list of recordings (`Audio` objects, `(samples, sample_rate)` tuples, bytes or `pathlib.Path`s), needs a
            backend that clones voices. `fit` joins the recordings into one reference of at most 30 seconds. The
            backend's default voice is used when it is `None`.

        language:
            The language of the texts.

        output_format:
            `'object'` returns `Audio` objects, `'bytes'` returns WAV bytes, and `'path'` writes WAV files and returns
            their paths.

        output_dir:
            The directory for `'path'` outputs. A new temporary directory is used when it is `None`.

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
            The voice name, or the reference audio joined from the recordings in `voice`.

    Examples:
        >>> from skai.generation import SpeechSynthesizer
        >>> synthesizer = SpeechSynthesizer(backend='dummy').fit()
        >>> speech = synthesizer.predict(['Hello', 'Goodbye'])
        >>> len(speech), speech[0].sample_rate
        (2, 16000)
    """

    task: ClassVar[str] = 'synthesize_speech'
    input_kind: ClassVar[Literal['text']] = 'text'
    output_kind: ClassVar[Literal['audio']] = 'audio'
    _parameter_constraints: ClassVar[dict] = {
        **BaseMediaGenerator._parameter_constraints,
        'voice': [str, bytes, tuple, list, Audio, PurePath, None],
        'language': [str, None],
    }

    def __init__(
        self,
        *,
        voice: str | Audio | tuple | bytes | PurePath | list | None = None,
        language: str | None = None,
        output_format: Literal['object', 'bytes', 'path'] = 'object',
        output_dir: str | PathLike | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 16,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.voice = voice
        self.language = language
        self.output_format = output_format
        self.output_dir = output_dir
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Read the reference audio of `voice` and check that the backend can clone voices."""
        if self.voice is None or isinstance(self.voice, str):
            self.voice_: str | Audio | None = self.voice
            return
        if isinstance(self.voice, list) and not self.voice:
            msg = 'The `voice` parameter of SpeechSynthesizer must hold at least one recording.'
            raise ValueError(msg)
        if not self.backend_.supports_voice_cloning:
            backends = list_backends(self.task, lambda backend_class: backend_class.supports_voice_cloning)
            msg = f'{type(self.backend_).__name__} cannot clone voices. Backends that can: {backends}.'
            raise UnsupportedBackendError(msg)
        recordings = self.voice if isinstance(self.voice, list) else [self.voice]
        self.voice_ = _join_recordings([read_audio(recording, index) for index, recording in enumerate(recordings)])

    def _build_request(self, sample: str, seed: int) -> Any:
        """Build the speech request of one text."""
        return SpeechRequest(sample, self.voice_, self.language, seed)
