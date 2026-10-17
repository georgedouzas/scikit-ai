"""Transcribe speech."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar, Literal

from numpy.typing import ArrayLike, NDArray
from sklearn.utils._param_validation import StrOptions

from skai.core import (
    Audio,
    Backend,
    BaseGenerator,
    TranscriptionRequest,
    UnsupportedBackendError,
    list_backends,
    to_object_array,
)


class SpeechTranscriber(BaseGenerator):
    """Return one transcript per audio sample.

    With `timestamps`, `predict` returns timed records, and `transform` still returns the plain text so the
    transcriber can feed a text component in a pipeline.

    Args:
        language:
            The language of the speech, such as `'en'`. It is detected when it is `None`.

        timestamps:
            `None` for plain text, `'segment'` for timed segments or `'word'` for timed words.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of audio samples submitted to the backend at once.

        on_error:
            `'raise'` raises when a sample fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

    Examples:
        >>> import numpy as np
        >>> from skai.transcription import SpeechTranscriber
        >>> transcriber = SpeechTranscriber(backend='dummy').fit()
        >>> transcriber.predict([(np.zeros(16000), 16000)])[0]
        'Speech of 1.00 seconds.'
    """

    task: ClassVar[str] = 'transcribe'
    input_kind: ClassVar[Literal['audio']] = 'audio'
    _parameter_constraints: ClassVar[dict] = {
        **BaseGenerator._parameter_constraints,
        'language': [str, None],
        'timestamps': [StrOptions({'segment', 'word'}), None],
    }

    def __init__(
        self,
        *,
        language: str | None = None,
        timestamps: Literal['segment', 'word'] | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 8,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.language = language
        self.timestamps = timestamps
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Check that the backend can return the requested timestamps."""
        if self.timestamps is None or self.timestamps in self.backend_.timestamp_granularities:
            return
        granularity = self.timestamps
        backends = list_backends(self.task, lambda backend_class: granularity in backend_class.timestamp_granularities)
        name = type(self.backend_).__name__
        msg = f'{name} cannot return `{granularity}` timestamps. Backends that can: {backends}.'
        raise UnsupportedBackendError(msg)

    def _build_request(self, sample: Audio, seed: int) -> TranscriptionRequest:
        """Build the transcription request of one audio sample."""
        return TranscriptionRequest(sample, self.language, self.timestamps)

    def _finalize(self, results: list[Any]) -> NDArray:
        """Return the transcripts, or their timed records when timestamps are requested."""
        if self.timestamps is None:
            return to_object_array(None if transcript is None else transcript.text for transcript in results)
        return to_object_array(
            None if transcript is None else [vars(segment) for segment in transcript.segments] for transcript in results
        )

    def transform(self, X: ArrayLike) -> NDArray:
        """Return the plain text of every transcript, so the transcriber can be an intermediate pipeline step.

        Args:
            X:
                The audio samples.

        Returns:
            The transcripts, in input order.
        """
        return to_object_array(None if transcript is None else transcript.text for transcript in self._run_task(X))
