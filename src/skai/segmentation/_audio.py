"""Split audio into timed speech, speaker turns, sounds and moments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral, Real
from typing import Any, ClassVar, Literal

from sklearn.utils._param_validation import Interval

from skai.core import (
    Audio,
    Backend,
    BaseSegmenter,
    DiarizationRequest,
    Moment,
    MomentRequest,
    Transcript,
    TranscriptionRequest,
    UnsupportedBackendError,
    cut_audio,
    list_backends,
    sound_spans,
)

GRANULARITIES: dict[str, Literal['segment', 'word']] = {'speech': 'segment', 'word': 'word'}


class AudioSegmenter(BaseSegmenter):
    """Split every audio sample into timed regions.

    Every region is a dict with `label`, `score`, `start` and `end` in seconds, and `media`, the audio of the region.
    `by='speech'` returns timed speech with `text`, `by='word'` timed words, `by='speaker'` speaker turns with `speaker`
    and `text`,
    `by='silence'` the parts between silences, found locally, and `by='moment'` the parts that match `prompt`.

    With `by='speech'` or `by='word'`, `fit(X, y)` with reference transcripts as `y` learns a vocabulary of the
    domain terms they contain, which the backend receives as context for spelling.

    Args:
        by:
            `'speech'`, `'word'`, `'speaker'`, `'silence'` or `'moment'`.

        prompt:
            A description of the moments to find with `by='moment'`.

        language:
            The language of the speech. It is detected when it is `None`.

        threshold:
            The RMS level below which audio counts as silent with `by='silence'`.

        min_silence:
            The shortest silence in seconds that separates two parts with `by='silence'`.

        max_vocabulary:
            The maximum number of terms in the vocabulary, the most frequent first.

        use_y:
            `'auto'` learns from `y` in `fit` but not in `fit_transform`, which pipelines use for intermediate steps.
            `True` always learns from `y`, and `False` never does.

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
        backend_ (Backend | None):
            The resolved backend, or `None` with `by='silence'`.

        vocabulary_ (tuple):
            The domain terms learned from the reference transcripts.

    Examples:
        >>> import numpy as np
        >>> from skai.segmentation import AudioSegmenter
        >>> regions = AudioSegmenter(by='speaker', backend='dummy').fit().predict([(np.zeros(16000), 16000)])[0]
        >>> [region['speaker'] for region in regions]
        ['SPEAKER_0', 'SPEAKER_1']
    """

    input_kind: ClassVar[Literal['audio']] = 'audio'
    operations: ClassVar[dict[str, str | None]] = {
        'speech': 'transcribe',
        'word': 'transcribe',
        'speaker': 'diarize',
        'silence': None,
        'moment': 'find_moments',
    }
    _parameter_constraints: ClassVar[dict] = {
        **BaseSegmenter._parameter_constraints,
        'prompt': [str, None],
        'language': [str, None],
        'threshold': [Interval(Real, 0, None, closed='left')],
        'min_silence': [Interval(Real, 0, None, closed='left')],
        'max_vocabulary': [Interval(Integral, 1, None, closed='left')],
    }

    def __init__(
        self,
        *,
        by: Literal['speech', 'word', 'speaker', 'silence', 'moment'] = 'speech',
        prompt: str | None = None,
        language: str | None = None,
        threshold: float = 0.01,
        min_silence: float = 0.3,
        max_vocabulary: int = 50,
        use_y: Literal['auto'] | bool = 'auto',
        backend: str | Backend | None = None,
        batch_size: int = 8,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.by = by
        self.prompt = prompt
        self.language = language
        self.threshold = threshold
        self.min_silence = min_silence
        self.max_vocabulary = max_vocabulary
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Check that moments have a prompt and that the backend returns the timestamps of timed speech."""
        if self.by == 'moment' and not self.prompt:
            msg = 'AudioSegmenter needs `prompt` with `by="moment"`.'
            raise ValueError(msg)
        granularity = GRANULARITIES.get(self.by)
        if granularity is None or self.backend_ is None or granularity in self.backend_.timestamp_granularities:
            return
        backends = list_backends(self.task_, lambda backend_class: granularity in backend_class.timestamp_granularities)
        msg = f'{type(self.backend_).__name__} cannot return `{granularity}` timestamps. Backends that can: {backends}.'
        raise UnsupportedBackendError(msg)

    def _learn(self, samples: list[Any], references: list[Any]) -> None:
        """Learn the vocabulary of the reference transcripts with `by='speech'`."""
        self.examples_ = ()
        self._learn_vocabulary(references if self.by in GRANULARITIES else [])

    def _build_request(self, sample: Audio, seed: int) -> Any:
        """Build the request of one audio sample."""
        if self.by in GRANULARITIES:
            return TranscriptionRequest(sample, self.language, GRANULARITIES[self.by], self.vocabulary_)
        if self.by == 'speaker':
            return DiarizationRequest(sample, self.language, self.vocabulary_)
        assert self.prompt is not None
        return MomentRequest(sample, self.prompt)

    def _complete(self, result: Transcript | list[Moment], sample: Audio) -> list[dict[str, Any]]:
        """Turn timed segments or moments into records with the audio of every region."""
        if isinstance(result, Transcript):
            segments = result.segments or ()
            records = []
            for segment in segments:
                record: dict[str, Any] = {'label': self.by, 'score': 1.0, 'start': segment.start, 'end': segment.end}
                if self.by == 'speaker':
                    record['speaker'] = segment.speaker
                record |= {'text': segment.text, 'media': cut_audio(sample, segment.start, segment.end)}
                records.append(record)
            return records
        return [
            {
                'label': 'moment',
                'score': moment.score,
                'start': moment.start,
                'end': moment.end,
                'media': cut_audio(sample, moment.start, moment.end),
            }
            for moment in result
        ]

    def _run_local(self, sample: Audio, seed: int) -> list[dict[str, Any]]:
        """Split one audio sample at its silences."""
        return [
            {'label': 'sound', 'score': 1.0, 'start': start, 'end': end, 'media': cut_audio(sample, start, end)}
            for start, end in sound_spans(sample, self.threshold, self.min_silence)
        ]
