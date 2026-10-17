"""Summarize texts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral
from typing import Any, ClassVar, Literal

from numpy.typing import NDArray
from sklearn.utils._param_validation import Interval

from skai.core import Backend, BaseGenerator, SummaryRequest, to_object_array


class TextSummarizer(BaseGenerator):
    """Return one summary per text.

    Summaries longer than `max_length` words are cut to `max_length` words.

    Args:
        max_length:
            The maximum number of words of a summary.

        style:
            A description of the style of the summaries, such as `'bullet points'` or `'for a child'`.

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

    Examples:
        >>> from skai.text import TextSummarizer
        >>> summarizer = TextSummarizer(max_length=4, backend='dummy').fit()
        >>> summarizer.predict(['scikit-ai gives media tasks a scikit-learn API.'])[0]
        'scikit-ai gives media tasks'
    """

    task: ClassVar[str] = 'summarize'
    input_kind: ClassVar[Literal['text']] = 'text'
    _parameter_constraints: ClassVar[dict] = {
        **BaseGenerator._parameter_constraints,
        'max_length': [Interval(Integral, 1, None, closed='left')],
        'style': [str, None],
    }

    def __init__(
        self,
        *,
        max_length: int = 100,
        style: str | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 16,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.max_length = max_length
        self.style = style
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _build_request(self, sample: str, seed: int) -> SummaryRequest:
        """Build the summary request of one text."""
        return SummaryRequest(sample, self.max_length, self.style)

    def _finalize(self, results: list[Any]) -> NDArray:
        """Cut every summary to `max_length` words."""
        return to_object_array(
            None if summary is None else ' '.join(summary.split()[: self.max_length]) for summary in results
        )
