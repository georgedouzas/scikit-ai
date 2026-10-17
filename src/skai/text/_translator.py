"""Translate texts."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import ClassVar, Literal

from skai.core import Backend, BaseGenerator, TranslationRequest


class TextTranslator(BaseGenerator):
    """Translate every text into a target language.

    Args:
        target_language:
            The language to translate into, such as `'French'`. It is required.

        source_language:
            The language of the texts. It is detected when it is `None`.

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
        >>> from skai.text import TextTranslator
        >>> translator = TextTranslator(target_language='French', backend='dummy').fit()
        >>> translator.predict(['Good morning.'])[0]
        '[French] Good morning.'
    """

    task: ClassVar[str] = 'translate'
    input_kind: ClassVar[Literal['text']] = 'text'
    _parameter_constraints: ClassVar[dict] = {
        **BaseGenerator._parameter_constraints,
        'target_language': [str],
        'source_language': [str, None],
    }

    def __init__(
        self,
        *,
        target_language: str | None = None,
        source_language: str | None = None,
        backend: str | Backend | None = None,
        batch_size: int = 16,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.target_language = target_language
        self.source_language = source_language
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _build_request(self, sample: str, seed: int) -> TranslationRequest:
        """Build the translation request of one text."""
        assert self.target_language is not None
        return TranslationRequest(sample, self.target_language, self.source_language)
