"""Split videos into shots and moments."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral, Real
from typing import Any, ClassVar, Literal

from sklearn.utils._param_validation import Interval

from skai.core import (
    Backend,
    BaseSegmenter,
    Moment,
    MomentRequest,
    Video,
    cut_video,
    read_video,
    sample_frames,
    shot_spans,
)


class VideoSegmenter(BaseSegmenter):
    """Split every video into shots, found locally, or into the moments that match `prompt`.

    Every region is a dict with `label`, `score`, `start` and `end` in seconds, and `media`, the clip of the region.

    Args:
        by:
            `'shot'` splits at cuts, found from the color histograms of consecutive frames without a model.
            `'moment'` asks the backend for the parts that match `prompt`.

        prompt:
            A description of the moments to find with `by='moment'`.

        threshold:
            The histogram distance between consecutive frames, from 0 to 1, above which a cut is found.

        n_frames:
            The number of frames sampled from every video with `by='moment'`.

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
        backend_ (Backend | None):
            The resolved backend, or `None` with `by='shot'`.

    Examples:
        >>> import numpy as np
        >>> from skai.core import Video
        >>> from skai.segmentation import VideoSegmenter
        >>> frames = np.concatenate([np.zeros((4, 8, 8, 3)), np.full((4, 8, 8, 3), 255)]).astype(np.uint8)
        >>> regions = VideoSegmenter(by='shot').fit().predict([Video(frames, fps=4)])[0]
        >>> [(region['start'], region['end']) for region in regions]
        [(0.0, 1.0), (1.0, 2.0)]
    """

    input_kind: ClassVar[Literal['video']] = 'video'
    operations: ClassVar[dict[str, str | None]] = {'shot': None, 'moment': 'find_moments'}
    _parameter_constraints: ClassVar[dict] = {
        **BaseSegmenter._parameter_constraints,
        'prompt': [str, None],
        'threshold': [Interval(Real, 0, 1, closed='both')],
        'n_frames': [Interval(Integral, 1, None, closed='left')],
    }

    def __init__(
        self,
        *,
        by: Literal['shot', 'moment'] = 'shot',
        prompt: str | None = None,
        threshold: float = 0.5,
        n_frames: int = 16,
        backend: str | Backend | None = None,
        batch_size: int = 2,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.by = by
        self.prompt = prompt
        self.threshold = threshold
        self.n_frames = n_frames
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _prepare(self) -> None:
        """Check that moments have a prompt."""
        if self.by == 'moment' and not self.prompt:
            msg = 'VideoSegmenter needs `prompt` with `by="moment"`.'
            raise ValueError(msg)

    def _learn(self, samples: list[Any], references: list[Any]) -> None:
        """Learn nothing, since segmentation takes no references."""
        self.examples_ = ()

    def _read_one(self, sample: Any, index: int) -> Video:
        """Read the whole video, since every region keeps its clip."""
        return read_video(sample, index)

    def _build_request(self, sample: Video, seed: int) -> MomentRequest:
        """Build the moment request of one video from its sampled frames."""
        assert self.prompt is not None
        return MomentRequest(sample_frames(sample, self.n_frames), self.prompt)

    def _complete(self, result: list[Moment], sample: Video) -> list[dict[str, Any]]:
        """Turn the moments into records with their clips."""
        return [
            {
                'label': 'moment',
                'score': moment.score,
                'start': moment.start,
                'end': moment.end,
                'media': cut_video(sample, moment.start, moment.end),
            }
            for moment in result
        ]

    def _run_local(self, sample: Video, seed: int) -> list[dict[str, Any]]:
        """Split one video at its cuts."""
        return [
            {'label': 'shot', 'score': 1.0, 'start': start, 'end': end, 'media': cut_video(sample, start, end)}
            for start, end in shot_spans(sample, self.threshold)
        ]
