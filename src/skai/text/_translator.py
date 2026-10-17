"""Translate texts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from typing import Any, ClassVar, Literal

from numpy.typing import ArrayLike
from sklearn.utils._param_validation import StrOptions

from skai.core import Backend, BaseSupervisedGenerator, TranslationRequest, find_terms

MIN_TERM_COUNT = 2


def _learn_glossary(sources: list[str], references: list[str]) -> tuple[tuple[str, str], ...]:
    """Map every term that appears in at least two sources to the one term shared by all their translations."""
    counts = Counter(term for source in sources for term in set(find_terms(source)))
    glossary = []
    for term, count in sorted(counts.items()):
        if count < MIN_TERM_COUNT:
            continue
        translations = [
            set(find_terms(reference))
            for source, reference in zip(sources, references, strict=True)
            if term in find_terms(source)
        ]
        shared = set.intersection(*translations)
        if term in shared:
            glossary.append((term, term))
        elif len(shared) == 1:
            glossary.append((term, shared.pop()))
    return tuple(glossary)


class TextTranslator(BaseSupervisedGenerator):
    """Translate every text into a target language, learning from reference translations.

    `fit(X, y)` with reference translations as `y` passes `k_shot` text and translation pairs to the backend as
    examples, and learns a glossary of terms that appear in several texts, so they are translated consistently.

    Args:
        target_language:
            The language to translate into, such as `'French'`. It is required.

        source_language:
            The language of the texts. It is detected when it is `None`.

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
            The text and reference translation pairs passed to the backend.

        glossary_ (tuple):
            Pairs of a term and its required translation.

        model_ (str):
            The id of the fine-tuned model. It is set only with `adaptation='finetune'`.

    Examples:
        >>> from skai.text import TextTranslator
        >>> texts = ['Open SkaiBox now.', 'SkaiBox is ready.']
        >>> translator = TextTranslator(target_language='French', backend='dummy')
        >>> translator.fit(texts, ['Ouvrez SkaiBox.', 'SkaiBox est prête.']).glossary_
        (('SkaiBox', 'SkaiBox'),)
    """

    task: ClassVar[str] = 'translate'
    input_kind: ClassVar[Literal['text']] = 'text'
    _parameter_constraints: ClassVar[dict] = {
        **BaseSupervisedGenerator._parameter_constraints,
        'adaptation': [StrOptions({'examples', 'finetune'})],
        'target_language': [str],
        'source_language': [str, None],
    }

    def __init__(
        self,
        *,
        target_language: str | None = None,
        source_language: str | None = None,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 8,
        use_y: Literal['auto'] | bool = 'auto',
        adaptation: Literal['examples', 'finetune'] = 'examples',
        backend: str | Backend | None = None,
        batch_size: int = 16,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.target_language = target_language
        self.source_language = source_language
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.adaptation = adaptation
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _learn(self, samples: list[Any], references: list[Any]) -> None:
        """Select the example pairs and learn the glossary from the references."""
        super()._learn(samples, references)
        self.glossary_ = _learn_glossary([str(sample) for sample in samples], [str(ref) for ref in references])

    def _build_request(self, sample: str, seed: int) -> TranslationRequest:
        """Build the translation request of one text."""
        assert self.target_language is not None
        return TranslationRequest(
            sample,
            self.target_language,
            self.source_language,
            self.examples_,
            self.glossary_,
        )
