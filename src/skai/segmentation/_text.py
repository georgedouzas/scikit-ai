"""Split texts into passages."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral
from typing import Any, ClassVar, Literal

from sklearn.utils._param_validation import Interval

from skai.core import BaseSegmenter, text_spans


class TextSegmenter(BaseSegmenter):
    """Split every text into paragraphs, sentences or chunks, locally and without a model.

    Every region is a dict with `label` (the value of `by`), `score` (one), `start` and `end` character offsets, and
    `media`, the passage itself.

    Args:
        by:
            `'paragraph'` splits at blank lines, `'sentence'` after sentence punctuation, and `'chunk'` into pieces of
            `chunk_size` characters that join back to the text.

        chunk_size:
            The number of characters of a chunk.

    Attributes:
        backend_ (None):
            Always `None`, since splitting runs locally.

    Examples:
        >>> from skai.segmentation import TextSegmenter
        >>> regions = TextSegmenter(by='sentence').fit().predict(['One. Two.'])[0]
        >>> [region['media'] for region in regions]
        ['One. ', 'Two.']
    """

    input_kind: ClassVar[Literal['text']] = 'text'
    operations: ClassVar[dict[str, str | None]] = {'paragraph': None, 'sentence': None, 'chunk': None}
    _parameter_constraints: ClassVar[dict] = {
        **BaseSegmenter._parameter_constraints,
        'chunk_size': [Interval(Integral, 1, None, closed='left')],
    }

    def __init__(self, *, by: Literal['paragraph', 'sentence', 'chunk'] = 'paragraph', chunk_size: int = 1000) -> None:
        self.by = by
        self.chunk_size = chunk_size

    def _run_local(self, sample: str, seed: int) -> list[dict[str, Any]]:
        """Split one text."""
        return [
            {'label': self.by, 'score': 1.0, 'start': start, 'end': end, 'media': sample[start:end]}
            for start, end in text_spans(sample, self.by, self.chunk_size)
        ]
