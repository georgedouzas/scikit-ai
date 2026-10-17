"""Transcribe the speech of videos."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from skai.core import Transcript, TranscriptionRequest, Video

from ._speech import SpeechTranscriber


class VideoTranscriber(SpeechTranscriber):
    """Return one transcript of the audio track of every video.

    Timestamps are measured from the start of the video, and a video without an audio track gives an empty transcript
    without calling the backend. `fit(X, y)` with reference transcripts as `y` learns a vocabulary of domain terms.

    Args:
        language:
            The language of the speech, such as `'en'`. It is detected when it is `None`.

        timestamps:
            `None` for plain text, `'segment'` for timed segments or `'word'` for timed words.

        max_vocabulary:
            The maximum number of terms in the vocabulary, the most frequent first.

        use_y:
            `'auto'` learns from `y` in `fit` but not in `fit_transform`, which pipelines use for intermediate steps.
            `True` always learns from `y`, and `False` never does.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of videos submitted to the backend at once.

        on_error:
            `'raise'` raises when a video fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        vocabulary_ (tuple):
            The domain terms learned from the reference transcripts.

    Examples:
        >>> import numpy as np
        >>> from skai.core import Audio, Video
        >>> from skai.transcription import VideoTranscriber
        >>> speaking = Video(np.zeros((10, 8, 8, 3), dtype=np.uint8), fps=10, audio=Audio(np.zeros(16000), 16000))
        >>> silent = Video(np.zeros((10, 8, 8, 3), dtype=np.uint8), fps=10)
        >>> VideoTranscriber(backend='dummy').fit().predict([speaking, silent]).tolist()
        ['Speech of 1.00 seconds.', '']
    """

    input_kind: ClassVar[Literal['video']] = 'video'  # type: ignore[assignment]  # narrows the input of the parent

    def _empty_result(self) -> Transcript:
        """Return the empty transcript of a video without an audio track."""
        return Transcript('')

    def _build_request(self, sample: Video, seed: int) -> TranscriptionRequest | None:  # type: ignore[override]
        """Build the transcription request of the audio track, or `None` for a video without one."""
        if sample.audio is None:
            return None
        return TranscriptionRequest(sample.audio_track(), self.language, self.timestamps, self.vocabulary_)
