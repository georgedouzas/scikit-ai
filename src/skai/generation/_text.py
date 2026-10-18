"""Generate text from texts, images, audio, videos and multimodal samples."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from collections import Counter
from numbers import Integral
from typing import Any, ClassVar, Literal

import numpy as np
from numpy.typing import ArrayLike
from sklearn.utils._param_validation import Interval, InvalidParameterError, StrOptions

from skai.core import (
    Backend,
    BaseSupervisedGenerator,
    ImageTranscriptionRequest,
    TextGenerationRequest,
    Transcript,
    TranscriptionRequest,
    VideoFrames,
    find_terms,
)

LENGTH_PERCENTILE = 90
MIN_TERM_COUNT = 2


def _learn_glossary(sources: list[str], references: list[str]) -> tuple[tuple[str, str], ...]:
    """Map every term that appears in at least two sources to the one term shared by all their references."""
    counts = Counter(term for source in sources for term in set(find_terms(source)))
    glossary = []
    for term, count in sorted(counts.items()):
        if count < MIN_TERM_COUNT:
            continue
        renderings = [
            set(find_terms(reference))
            for source, reference in zip(sources, references, strict=True)
            if term in find_terms(source)
        ]
        shared = set.intersection(*renderings)
        if term in shared:
            glossary.append((term, term))
        elif len(shared) == 1:
            glossary.append((term, shared.pop()))
    return tuple(glossary)


class BaseTextGenerator(BaseSupervisedGenerator):
    """Return one text per sample, by following instructions or by transcribing it.

    With `operation='instruct'`, `fit(X, y)` with reference texts as `y` passes example pairs to the backend, learns the
    maximum length with `max_length='auto'` and, with `learn_terms=True`, a glossary. With `operation='transcribe'` it
    learns a vocabulary of the domain terms of reference transcripts.
    """

    _parameter_constraints: ClassVar[dict] = {
        **BaseSupervisedGenerator._parameter_constraints,
        'instructions': [str, None],
        'max_length': [Interval(Integral, 1, None, closed='left'), StrOptions({'auto'}), None],
        'adaptation': [StrOptions({'examples', 'finetune'})],
        'learn_terms': ['boolean'],
        'language': [str, None],
        'max_vocabulary': [Interval(Integral, 1, None, closed='left')],
        'format': [StrOptions({'text', 'markdown'})],
        'n_frames': [Interval(Integral, 1, None, closed='left')],
    }

    def _prepare(self) -> None:
        """Check that instructed generation has instructions and that only instructed generation is fine-tuned."""
        operation = self._operation()
        if operation == 'instruct' and not getattr(self, 'instructions', None):
            msg = f"The 'instructions' parameter of {type(self).__name__} is required with `operation='instruct'`."
            raise InvalidParameterError(msg)
        if operation != 'instruct' and getattr(self, 'adaptation', 'examples') == 'finetune':
            msg = f"{type(self).__name__} can be fine-tuned only with `operation='instruct'`."
            raise ValueError(msg)

    def _learn(self, samples: list[Any], references: list[Any]) -> None:
        """Learn examples, length and glossary for instructions, or a vocabulary for transcription."""
        self.max_length_: int | None = None
        self.glossary_: tuple[tuple[str, str], ...] = ()
        self.vocabulary_: tuple[str, ...] = ()
        if self._operation() != 'instruct':
            self.examples_ = ()
            self._learn_vocabulary(references)
            return
        super()._learn(samples, references)
        max_length = getattr(self, 'max_length', None)
        if max_length == 'auto':
            if references:
                lengths = [len(str(reference).split()) for reference in references]
                self.max_length_ = max(1, int(np.percentile(lengths, LENGTH_PERCENTILE)))
        else:
            self.max_length_ = max_length
        if getattr(self, 'learn_terms', False) and references:
            self.glossary_ = _learn_glossary([str(sample) for sample in samples], [str(ref) for ref in references])

    def _transcription_request(self, sample: Any) -> Any:
        """Build the transcription request of one sample."""
        return TranscriptionRequest(sample, getattr(self, 'language', None), None, self.vocabulary_)

    def _build_request(self, sample: Any, seed: int) -> Any:
        """Build the request of one sample for the operation."""
        if self._operation() == 'transcribe':
            return self._transcription_request(sample)
        assert self.instructions is not None
        return TextGenerationRequest(sample, self.instructions, self.examples_, self.max_length_, self.glossary_)

    def _empty_result(self) -> str:
        """Return the empty transcript of a sample without speech."""
        return ''

    def _complete(self, result: Any, sample: Any) -> str:
        """Return the text, cut to the maximum length."""
        text = result.text if isinstance(result, Transcript) else str(result)
        return text if self.max_length_ is None else ' '.join(text.split()[: self.max_length_])


class TextToTextGenerator(BaseTextGenerator):
    """Return one text per text by following instructions, such as a summary, a translation or an answer.

    `fit(X, y)` with reference texts as `y` passes `k_shot` text and reference pairs to the backend as examples, learns
    `max_length_` with `max_length='auto'`, and learns a glossary of consistently rendered terms with
    `learn_terms=True`.

    Args:
        instructions:
            What to produce from every text, such as `'Summarize the text.'` or `'Translate into Greek.'`.

        operation:
            `'instruct'`, the only operation.

        max_length:
            The maximum number of words of every output. `'auto'` learns it from the references as the 90th percentile
            of their word counts, and `None` sets no limit.

        learn_terms:
            Whether `fit` learns a glossary of the terms that appear in at least two texts, mapped to the term their
            references share.

        k_shot:
            How many training texts are passed to the backend as examples with their references. `None` passes up to
            `max_examples`, `0` passes none, an integer selects that many at random, and an array of indices selects
            those texts.

        max_examples:
            The maximum number of examples when `k_shot` is `None`.

        adaptation:
            `'examples'` passes examples to the backend at prediction. `'finetune'` trains a model on all the training
            data in `fit`, on backends that support fine-tuning, and sends later requests to it.

        use_y:
            `'auto'` learns from `y` in `fit` but not in `fit_transform`, which pipelines use for intermediate steps.
            `True` always learns from `y`, and `False` never does.

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
            The pairs of a text and its reference passed to the backend.

        max_length_ (int | None):
            The maximum number of words of every output.

        glossary_ (tuple):
            Pairs of a term and its required rendering.

        model_ (str):
            The id of the fine-tuned model. It is set only with `adaptation='finetune'`.

    Examples:
        >>> from skai.generation import TextToTextGenerator
        >>> generator = TextToTextGenerator(instructions='Summarize the text.', max_length=3, backend='dummy').fit()
        >>> generator.predict(['One two three four five.']).tolist()
        ['One two three']
    """

    input_kind: ClassVar[Literal['text']] = 'text'
    operations: ClassVar[dict[str, str | None]] = {'instruct': 'generate_text'}

    def __init__(
        self,
        *,
        instructions: str | None = None,
        operation: Literal['instruct'] = 'instruct',
        max_length: int | Literal['auto'] | None = None,
        learn_terms: bool = False,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 8,
        adaptation: Literal['examples', 'finetune'] = 'examples',
        use_y: Literal['auto'] | bool = 'auto',
        backend: str | Backend | None = None,
        batch_size: int = 16,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.instructions = instructions
        self.operation = operation
        self.max_length = max_length
        self.learn_terms = learn_terms
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.adaptation = adaptation
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state


class ImageToTextGenerator(BaseTextGenerator):
    """Return one text per image, by following instructions or by transcribing the text written in it.

    With `operation='instruct'`, the image is described, answered about or captioned as the instructions ask, and
    `fit(X, y)` with reference texts as `y` passes example pairs to the backend. With `operation='transcribe'`, the
    text of the image is returned exactly, as plain text or as Markdown that keeps the page structure.

    Args:
        instructions:
            What to produce from every image, such as `'Describe the image.'`. Required with `operation='instruct'`.

        operation:
            `'instruct'` or `'transcribe'`.

        format:
            `'text'` or `'markdown'`, the form of transcriptions.

        max_length:
            The maximum number of words of every output. `'auto'` learns it from the references as the 90th percentile
            of their word counts, and `None` sets no limit.

        k_shot:
            How many training images are passed to the backend as examples with their references. `None` passes up to
            `max_examples`, `0` passes none, an integer selects that many at random, and an array of indices selects
            those images.

        max_examples:
            The maximum number of examples when `k_shot` is `None`.

        adaptation:
            `'examples'` passes examples to the backend at prediction. `'finetune'` trains a model on all the training
            data in `fit`, on backends that support fine-tuning, and sends later requests to it.

        use_y:
            `'auto'` learns from `y` in `fit` but not in `fit_transform`, which pipelines use for intermediate steps.
            `True` always learns from `y`, and `False` never does.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of images submitted to the backend at once.

        on_error:
            `'raise'` raises when an image fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the selection of examples and the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        examples_ (tuple):
            The pairs of an image and its reference passed to the backend.

        max_length_ (int | None):
            The maximum number of words of every output.

        model_ (str):
            The id of the fine-tuned model. It is set only with `adaptation='finetune'`.

    Examples:
        >>> from PIL import Image
        >>> from skai.generation import ImageToTextGenerator
        >>> generator = ImageToTextGenerator(instructions='Describe the image.', backend='dummy').fit()
        >>> generator.predict([Image.new('RGB', (64, 48))])[0]
        'A 64x48 image.'
    """

    input_kind: ClassVar[Literal['image']] = 'image'
    operations: ClassVar[dict[str, str | None]] = {'instruct': 'generate_text', 'transcribe': 'transcribe_image'}

    def __init__(
        self,
        *,
        instructions: str | None = None,
        operation: Literal['instruct', 'transcribe'] = 'instruct',
        format: Literal['text', 'markdown'] = 'text',  # noqa: A002 - the name of the output format of transcriptions
        max_length: int | Literal['auto'] | None = None,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 8,
        adaptation: Literal['examples', 'finetune'] = 'examples',
        use_y: Literal['auto'] | bool = 'auto',
        backend: str | Backend | None = None,
        batch_size: int = 8,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.instructions = instructions
        self.operation = operation
        self.format = format
        self.max_length = max_length
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.adaptation = adaptation
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _learn(self, samples: list[Any], references: list[Any]) -> None:
        """Learn examples and length for instructions, and nothing for transcription."""
        if self.operation == 'instruct':
            super()._learn(samples, references)
            return
        self.max_length_ = None
        self.glossary_ = ()
        self.vocabulary_ = ()
        self.examples_ = ()

    def _transcription_request(self, sample: Any) -> ImageTranscriptionRequest:
        """Build the request that transcribes the text of one image."""
        return ImageTranscriptionRequest(sample, self.format == 'markdown')


class AudioToTextGenerator(BaseTextGenerator):
    """Return one text per audio sample, by transcribing the speech or by following instructions.

    With `operation='transcribe'` (the default), `fit(X, y)` with reference transcripts as `y` learns a vocabulary of
    the names, products and technical terms they contain, which the backend receives as context for spelling. With
    `operation='instruct'`, the audio is described, summarized or answered about by an audio-language model, as the
    instructions ask.

    Args:
        operation:
            `'transcribe'` or `'instruct'`.

        instructions:
            What to produce from every audio sample. Required with `operation='instruct'`.

        language:
            The language of the speech, such as `'en'`. It is detected when it is `None`.

        max_vocabulary:
            The maximum number of terms in the vocabulary, the most frequent first.

        max_length:
            The maximum number of words of every output. `'auto'` learns it from the references as the 90th percentile
            of their word counts, and `None` sets no limit.

        k_shot:
            How many training samples are passed to the backend as examples with their references, with
            `operation='instruct'`. `None` passes up to `max_examples`, `0` passes none, an integer selects that many
            at random, and an array of indices selects those samples.

        max_examples:
            The maximum number of examples when `k_shot` is `None`.

        adaptation:
            `'examples'` passes examples to the backend at prediction. `'finetune'` trains a model on all the training
            data in `fit`, on backends that support fine-tuning, with `operation='instruct'`.

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
            Controls the selection of examples and the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        vocabulary_ (tuple):
            The domain terms learned from the reference transcripts.

        examples_ (tuple):
            The pairs of a sample and its reference passed to the backend.

        max_length_ (int | None):
            The maximum number of words of every output.

    Examples:
        >>> import numpy as np
        >>> from skai.generation import AudioToTextGenerator
        >>> AudioToTextGenerator(backend='dummy').fit().predict([(np.zeros(16000), 16000)])[0]
        'Speech of 1.00 seconds.'
    """

    input_kind: ClassVar[Literal['audio']] = 'audio'
    operations: ClassVar[dict[str, str | None]] = {'transcribe': 'transcribe', 'instruct': 'generate_text'}

    def __init__(
        self,
        *,
        operation: Literal['transcribe', 'instruct'] = 'transcribe',
        instructions: str | None = None,
        language: str | None = None,
        max_vocabulary: int = 50,
        max_length: int | Literal['auto'] | None = None,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 8,
        adaptation: Literal['examples', 'finetune'] = 'examples',
        use_y: Literal['auto'] | bool = 'auto',
        backend: str | Backend | None = None,
        batch_size: int = 8,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.operation = operation
        self.instructions = instructions
        self.language = language
        self.max_vocabulary = max_vocabulary
        self.max_length = max_length
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.adaptation = adaptation
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state


class VideoToTextGenerator(BaseTextGenerator):
    """Return one text per video, by following instructions or by transcribing its speech.

    With `operation='instruct'` (the default), the model sees frames sampled evenly over every video, and the audio
    track on backends that accept audio, and produces what the instructions ask. With `operation='transcribe'`, the
    speech of the audio track is transcribed, learning a vocabulary from reference transcripts as
    `AudioToTextGenerator` does, and a video without an audio track gives an empty transcript.

    Args:
        instructions:
            What to produce from every video. Required with `operation='instruct'`.

        operation:
            `'instruct'` or `'transcribe'`.

        n_frames:
            The number of frames sampled from every video.

        language:
            The language of the speech. It is detected when it is `None`.

        max_vocabulary:
            The maximum number of terms in the vocabulary, the most frequent first.

        max_length:
            The maximum number of words of every output. `'auto'` learns it from the references as the 90th percentile
            of their word counts, and `None` sets no limit.

        k_shot:
            How many training videos are passed to the backend as examples with their references, with
            `operation='instruct'`. `None` passes up to `max_examples`, `0` passes none, an integer selects that many
            at random, and an array of indices selects those videos.

        max_examples:
            The maximum number of examples when `k_shot` is `None`.

        adaptation:
            `'examples'` passes examples to the backend at prediction. `'finetune'` trains a model on all the training
            data in `fit`, on backends that support fine-tuning, with `operation='instruct'`.

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
            Controls the selection of examples and the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        vocabulary_ (tuple):
            The domain terms learned from the reference transcripts.

        examples_ (tuple):
            The pairs of a video and its reference passed to the backend.

        max_length_ (int | None):
            The maximum number of words of every output.

    Examples:
        >>> import numpy as np
        >>> from skai.core import Video
        >>> from skai.generation import VideoToTextGenerator
        >>> video = Video(np.zeros((8, 8, 8, 3), dtype=np.uint8), fps=4)
        >>> VideoToTextGenerator(operation='transcribe', backend='dummy').fit().predict([video])[0]
        ''
    """

    input_kind: ClassVar[Literal['video']] = 'video'
    operations: ClassVar[dict[str, str | None]] = {'instruct': 'generate_text', 'transcribe': 'transcribe'}

    def __init__(
        self,
        *,
        instructions: str | None = None,
        operation: Literal['instruct', 'transcribe'] = 'instruct',
        n_frames: int = 8,
        language: str | None = None,
        max_vocabulary: int = 50,
        max_length: int | Literal['auto'] | None = None,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 4,
        adaptation: Literal['examples', 'finetune'] = 'examples',
        use_y: Literal['auto'] | bool = 'auto',
        backend: str | Backend | None = None,
        batch_size: int = 4,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.instructions = instructions
        self.operation = operation
        self.n_frames = n_frames
        self.language = language
        self.max_vocabulary = max_vocabulary
        self.max_length = max_length
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.adaptation = adaptation
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state

    def _transcription_request(self, sample: VideoFrames) -> TranscriptionRequest | None:
        """Build the transcription request of the audio track, or `None` for a video without one."""
        if sample.audio is None:
            return None
        return TranscriptionRequest(sample.audio, self.language, None, self.vocabulary_)


class MultimodalToTextGenerator(BaseTextGenerator):
    """Return one text per multimodal sample by following instructions, with every named part in one request.

    A sample is a dict, or a DataFrame row, of named texts, images, audio and videos. Strings are texts unless they are
    URLs or paths of existing media files. `fit(X, y)` with reference texts as `y` passes example pairs to the
    backend.

    Args:
        instructions:
            What to produce from every sample, such as `'Write a caption.'`.

        operation:
            `'instruct'`, the only operation.

        n_frames:
            The number of frames sampled from every video part.

        max_length:
            The maximum number of words of every output. `'auto'` learns it from the references as the 90th percentile
            of their word counts, and `None` sets no limit.

        k_shot:
            How many training samples are passed to the backend as examples with their references. `None` passes up to
            `max_examples`, `0` passes none, an integer selects that many at random, and an array of indices selects
            those samples.

        max_examples:
            The maximum number of examples when `k_shot` is `None`.

        adaptation:
            `'examples'` passes examples to the backend at prediction. `'finetune'` trains a model on all the training
            data in `fit`, on backends that support fine-tuning, and sends later requests to it.

        use_y:
            `'auto'` learns from `y` in `fit` but not in `fit_transform`, which pipelines use for intermediate steps.
            `True` always learns from `y`, and `False` never does.

        backend:
            The backend, as `'<provider>'`, `'<provider>:<model>'` or a backend object. The local Hugging Face backend
            is used when it is `None`.

        batch_size:
            The number of samples submitted to the backend at once.

        on_error:
            `'raise'` raises when a sample fails, and `'ignore'` returns `None` for it and warns.

        random_state:
            Controls the selection of examples and the seeds sent to the backend.

    Attributes:
        backend_ (Backend):
            The resolved backend.

        examples_ (tuple):
            The pairs of a sample and its reference passed to the backend.

        max_length_ (int | None):
            The maximum number of words of every output.

    Examples:
        >>> from PIL import Image
        >>> from skai.generation import MultimodalToTextGenerator
        >>> generator = MultimodalToTextGenerator(instructions='Write a caption.', backend='dummy').fit()
        >>> generator.predict([{'photo': Image.new('RGB', (8, 8)), 'notes': 'Crete'}])[0]
        'photo: A 8x8 image. notes: Crete'
    """

    input_kind: ClassVar[Literal['multimodal']] = 'multimodal'
    operations: ClassVar[dict[str, str | None]] = {'instruct': 'generate_text'}

    def __init__(
        self,
        *,
        instructions: str | None = None,
        operation: Literal['instruct'] = 'instruct',
        n_frames: int = 8,
        max_length: int | Literal['auto'] | None = None,
        k_shot: int | ArrayLike | None = None,
        max_examples: int | None = 4,
        adaptation: Literal['examples', 'finetune'] = 'examples',
        use_y: Literal['auto'] | bool = 'auto',
        backend: str | Backend | None = None,
        batch_size: int = 8,
        on_error: Literal['raise', 'ignore'] = 'raise',
        random_state: int | None = None,
    ) -> None:
        self.instructions = instructions
        self.operation = operation
        self.n_frames = n_frames
        self.max_length = max_length
        self.k_shot = k_shot
        self.max_examples = max_examples
        self.adaptation = adaptation
        self.use_y = use_y
        self.backend = backend
        self.batch_size = batch_size
        self.on_error = on_error
        self.random_state = random_state
