"""Summarize texts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from numbers import Integral
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from sklearn.utils._param_validation import Interval, StrOptions

from skai.core import Backend, BaseSupervisedGenerator, SummaryRequest, to_object_array

DEFAULT_MAX_LENGTH = 100
LENGTH_PERCENTILE = 90


class TextSummarizer(BaseSupervisedGenerator):
    """Return one summary per text, learning style and length from reference summaries.

    `fit(X, y)` with reference summaries as `y` passes `k_shot` text and summary pairs to the backend as examples, and
    learns the length of the summaries when `max_length` is `None`. Summaries longer than the length are cut to it.

    Args:
        max_length:
            The maximum number of words of a summary. When it is `None`, it is learned from the references, or
            set to 100 without them.

        style:
            A description of the style of the summaries, such as `'bullet points'` or `'for a child'`.

        k_shot:
            How many reference pairs are passed to the backend as examples. `None` passes all of them up to
            `max_examples`, `0` passes none, an integer selects that many at random, and an array of indices selects
            those pairs.

        max_examples:
            The maximum number of examples when `k_shot` is `None`.

        use_y:
            `'auto'` learns from `y` in `fit` but not in `fit_transform`, which pipelines use for intermediate steps.
            `True` always learns from `y`, and `False` never does.

        adaptation:
            `'examples'` passes examples to the backend at prediction. `'finetune'` trains a model on all the training
            data in `fit`, on backends that support fine-tuning, and sends later requests to it.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of texts submitted to the backend at once.

        on_error:
            `'raise'` raises when a text fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the selection of examples and the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        examples_ (tuple):
            The text and reference summary pairs passed to the backend.

        max_length_ (int):
            The maximum number of words of a summary.

        model_ (str):
            The id of the fine-tuned model. It is set only with `adaptation='finetune'`.

    Examples:
        >>> from skai.text import TextSummarizer
        >>> texts = ['One two three four five six.', 'Seven eight nine ten eleven twelve.']
        >>> summarizer = TextSummarizer(max_length=None, backend='dummy').fit(texts, ['One two.', 'Seven eight.'])
        >>> summarizer.max_length_
        2
        >>> summarizer.predict(['Alpha beta gamma delta.'])[0]
        'Alpha beta'
    """

    task: ClassVar[str] = 'summarize'
    input_kind: ClassVar[Literal['text']] = 'text'
    _parameter_constraints: ClassVar[dict] = {
        **BaseSupervisedGenerator._parameter_constraints,
        'adaptation': [StrOptions({'examples', 'finetune'})],
        'max_length': [Interval(Integral, 1, None, closed='left'), None],
        'style': [str, None],
    }

    def __init__(
        self,
        *,
        max_length: int | None = DEFAULT_MAX_LENGTH,
        style: str | None = None,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 8,
        use_y: Literal['auto'] | bool = 'auto',
        adaptation: Literal['examples', 'finetune'] = 'examples',
        backend: str | Backend | None = None,
        batch_size: int = 16,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.max_length = max_length
        self.style = style
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.adaptation = adaptation
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _learn(self, samples: list[Any], references: list[Any]) -> None:
        """Select the example pairs and learn the summary length from the references."""
        super()._learn(samples, references)
        if self.max_length is not None:
            self.max_length_ = self.max_length
        elif references:
            lengths = [len(str(reference).split()) for reference in references]
            self.max_length_ = max(1, int(np.percentile(lengths, LENGTH_PERCENTILE)))
        else:
            self.max_length_ = DEFAULT_MAX_LENGTH

    def _build_request(self, sample: str, seed: int) -> SummaryRequest:
        """Build the summary request of one text."""
        return SummaryRequest(sample, self.max_length_, self.style, self.examples_)

    def _finalize(self, results: list[Any]) -> NDArray:
        """Cut every summary to the learned maximum length."""
        return to_object_array(
            None if summary is None else ' '.join(summary.split()[: self.max_length_]) for summary in results
        )
