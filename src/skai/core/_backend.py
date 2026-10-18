"""Define the base classes of the backends that run component tasks."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from abc import abstractmethod
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from typing import Any, ClassVar, Self

import numpy as np
from numpy.typing import NDArray
from PIL import Image
from sklearn.base import BaseEstimator, clone

from ._errors import UnsupportedBackendError
from ._media import Audio, Video
from ._prompts import (
    FineTuneRequest,
    Prompt,
    build_classification_prompt,
    build_description_prompt,
    build_extraction_prompt,
    build_summary_prompt,
    build_translation_prompt,
    parse_fields,
    parse_labels,
)
from ._requests import (
    AnimationRequest,
    AvatarRequest,
    ClassificationRequest,
    DescriptionRequest,
    EmbeddingRequest,
    ExtractionRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    SpeechRequest,
    SummaryRequest,
    Transcript,
    TranscriptionRequest,
    TranslationRequest,
    VideoGenerationRequest,
)

PROMPTED_AUDIO_TASKS = frozenset({'classify_audio', 'describe_audio'})
TASKS = (
    'classify_text',
    'summarize',
    'translate',
    'embed_text',
    'synthesize_speech',
    'generate_image',
    'describe_image',
    'classify_image',
    'transcribe',
    'animate_avatar',
    'classify_audio',
    'classify_video',
    'describe_audio',
    'describe_video',
    'extract_fields',
    'embed_image',
    'embed_audio',
    'generate_video',
    'edit_image',
    'animate_image',
    'fine_tune',
)


def _call(method: Callable[[Any], object], request: object) -> object:
    """Call a task method and return its exception instead of raising it."""
    try:
        return method(request)
    except Exception as error:  # noqa: BLE001 - failures are returned per sample so components can apply `on_error`
        return error


class Backend(BaseEstimator):
    """Run component tasks on a model provider.

    A backend implements a task by overriding the method of the same name. The tasks a backend supports are the
    methods it overrides. Attributes that start with an underscore hold runtime state, such as a client, and are not
    pickled.

    Args:
        model:
            The model to use. The default model of the task is used when it is `None`.

        options:
            Extra keyword arguments passed to the provider on every call.

        max_concurrency:
            The maximum number of requests that run at the same time.
    """

    default_models: ClassVar[dict[str, str]] = {}
    supports_voice_cloning: ClassVar[bool] = False
    supports_masks: ClassVar[bool] = False
    accepts_audio: ClassVar[bool] = False
    accepts_context: ClassVar[bool] = False
    shortens_embeddings: ClassVar[bool] = False
    avatar_size: ClassVar[int | None] = None
    min_finetune_examples: ClassVar[int] = 10
    sends_video_audio: ClassVar[bool] = False
    timestamp_granularities: ClassVar[frozenset[str]] = frozenset()
    shared_params: ClassVar[frozenset[str]] = frozenset({'client'})

    def __init__(self, model: str | None = None, *, options: dict | None = None, max_concurrency: int = 1) -> None:
        self.model = model
        self.options = options
        self.max_concurrency = max_concurrency

    def __getstate__(self) -> dict[str, Any]:
        """Return the state to pickle without the runtime attributes."""
        state = super().__getstate__()
        return {key: value for key, value in state.items() if not key.startswith('_') or key == '_sklearn_version'}

    def __sklearn_clone__(self) -> Self:
        """Clone the backend, sharing live objects such as clients, including those inside `options`."""
        params = self.get_params(deep=False)
        cloned = {
            name: value if name in self.shared_params else clone(value, safe=False)
            for name, value in params.items()
            if name != 'options'
        }
        cloned['options'] = None if self.options is None else dict(self.options)
        return type(self)(**cloned)

    @classmethod
    def supported_tasks(cls) -> frozenset[str]:
        """Return the tasks that the backend implements.

        Returns:
            The names of the tasks.
        """
        return frozenset(task for task in TASKS if getattr(cls, task) is not getattr(Backend, task))

    @property
    def call_options(self) -> dict[str, Any]:
        """The extra keyword arguments passed to the provider."""
        return dict(self.options or {})

    def resolve_model(self, task: str) -> str:
        """Return the model that runs a task.

        Args:
            task:
                The name of the task.

        Returns:
            The `model` parameter, or the default model of the task.

        Raises:
            UnsupportedBackendError:
                If no model is given and the task has no default model.
        """
        if self.model is not None:
            return self.model
        if task not in self.default_models:
            msg = f'{type(self).__name__} has no default model for `{task}`. Set the `model` parameter.'
            raise UnsupportedBackendError(msg)
        return self.default_models[task]

    def connect(self) -> Self:
        """Prepare the backend to run requests, for example by creating a client.

        Returns:
            The backend.
        """
        return self

    def run(self, task: str, requests: list[Any]) -> list[Any]:
        """Run a task on several requests, keeping their order.

        Args:
            task:
                The name of the task.

            requests:
                The requests.

        Returns:
            The result of every request, or the exception it raised.
        """
        method = partial(_call, getattr(self, task))
        if self.max_concurrency <= 1 or len(requests) <= 1:
            return [method(request) for request in requests]
        with ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
            return list(executor.map(method, requests))

    def classify_text(self, request: ClassificationRequest) -> dict[str, float]:
        """Score a text against the labels of a request.

        Args:
            request:
                The classification request.

        Returns:
            A score per label.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def summarize(self, request: SummaryRequest) -> str:
        """Summarize a text.

        Args:
            request:
                The summary request.

        Returns:
            The summary.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def translate(self, request: TranslationRequest) -> str:
        """Translate a text.

        Args:
            request:
                The translation request.

        Returns:
            The translation.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def embed_text(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Embed a text.

        Args:
            request:
                The embedding request.

        Returns:
            The embedding vector.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def synthesize_speech(self, request: SpeechRequest) -> Audio:
        """Speak a text.

        Args:
            request:
                The speech request.

        Returns:
            The speech.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def generate_image(self, request: ImageGenerationRequest) -> Image.Image:
        """Generate an image from a prompt.

        Args:
            request:
                The image generation request.

        Returns:
            The image.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def describe_image(self, request: DescriptionRequest) -> str:
        """Describe an image.

        Args:
            request:
                The description request.

        Returns:
            The description.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def classify_image(self, request: ClassificationRequest) -> dict[str, float]:
        """Score an image against the labels of a request.

        Args:
            request:
                The classification request.

        Returns:
            A score per label.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        """Transcribe speech.

        Args:
            request:
                The transcription request.

        Returns:
            The transcript.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def animate_avatar(self, request: AvatarRequest) -> Video:
        """Animate a portrait so that it speaks an audio track.

        Args:
            request:
                The avatar request.

        Returns:
            The video.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def classify_audio(self, request: ClassificationRequest) -> dict[str, float]:
        """Score audio against the labels of a request.

        Args:
            request:
                The request.

        Returns:
            A score per label.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def classify_video(self, request: ClassificationRequest) -> dict[str, float]:
        """Score sampled video frames against the labels of a request.

        Args:
            request:
                The request.

        Returns:
            A score per label.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def describe_audio(self, request: DescriptionRequest) -> str:
        """Describe audio.

        Args:
            request:
                The request.

        Returns:
            The description.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def describe_video(self, request: DescriptionRequest) -> str:
        """Describe sampled video frames.

        Args:
            request:
                The request.

        Returns:
            The description.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def extract_fields(self, request: ExtractionRequest) -> dict[str, Any]:
        """Extract named fields from a text.

        Args:
            request:
                The request.

        Returns:
            The value of every field, or `None` for a field the text does not contain.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def embed_image(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Embed an image.

        Args:
            request:
                The request.

        Returns:
            The embedding vector.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def embed_audio(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Embed audio.

        Args:
            request:
                The request.

        Returns:
            The embedding vector.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def generate_video(self, request: VideoGenerationRequest) -> Video:
        """Generate a video from a prompt.

        Args:
            request:
                The request.

        Returns:
            The video.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def edit_image(self, request: ImageEditRequest) -> Image.Image:
        """Edit an image with an instruction.

        Args:
            request:
                The request.

        Returns:
            The edited image.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def animate_image(self, request: AnimationRequest) -> Video:
        """Turn an image into a short video that starts from it.

        Args:
            request:
                The request.

        Returns:
            The video.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def fine_tune(self, request: FineTuneRequest) -> str:
        """Train a model on prompts and their expected answers, and wait until it is ready.

        Args:
            request:
                The fine-tuning request.

        Returns:
            The id of the trained model.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError


class PromptedBackend(Backend):
    """Run text and image tasks by prompting a chat or vision model.

    A subclass implements `complete`, and inherits classification, summarization, translation, description and field
    extraction. Audio tasks are supported only when the subclass sets `accepts_audio`, and video tasks include the
    audio track only when it sets `sends_video_audio`.

    Args:
        model:
            The model to use. The default model of the task is used when it is `None`.

        options:
            Extra keyword arguments passed to the provider on every call.

        max_concurrency:
            The maximum number of requests that run at the same time.
    """

    @classmethod
    def supported_tasks(cls) -> frozenset[str]:
        """Return the tasks that the backend implements, without prompted audio tasks when it does not accept audio.

        Returns:
            The names of the tasks.
        """
        tasks = super().supported_tasks()
        if cls.accepts_audio:
            return tasks
        prompted = {task for task in PROMPTED_AUDIO_TASKS if getattr(cls, task) is getattr(PromptedBackend, task)}
        return tasks - prompted

    def sends_video_audio_now(self) -> bool:
        """Return whether video requests include the audio track with the current settings.

        Returns:
            The class flag `sends_video_audio`, unless a subclass makes it depend on its parameters.
        """
        return self.sends_video_audio

    @abstractmethod
    def complete(self, prompt: Prompt, model: str) -> str:
        """Return the answer of a model to a prompt.

        Args:
            prompt:
                The prompt.

            model:
                The model to prompt.

        Returns:
            The answer of the model.
        """

    def classify_text(self, request: ClassificationRequest) -> dict[str, float]:
        """Score a text by asking the model for its labels.

        Args:
            request:
                The classification request.

        Returns:
            A score of one for every label the model names and zero for the others.
        """
        answer = self.complete(build_classification_prompt(request), self.resolve_model('classify_text'))
        return parse_labels(answer, request.labels, request.multi_label)

    def summarize(self, request: SummaryRequest) -> str:
        """Summarize a text by prompting the model.

        Args:
            request:
                The summary request.

        Returns:
            The summary.
        """
        return self.complete(build_summary_prompt(request), self.resolve_model('summarize')).strip()

    def translate(self, request: TranslationRequest) -> str:
        """Translate a text by prompting the model.

        Args:
            request:
                The translation request.

        Returns:
            The translation.
        """
        return self.complete(build_translation_prompt(request), self.resolve_model('translate')).strip()

    def describe_image(self, request: DescriptionRequest) -> str:
        """Describe an image by prompting the vision model.

        Args:
            request:
                The description request.

        Returns:
            The description.
        """
        return self.complete(build_description_prompt(request), self.resolve_model('describe_image')).strip()

    def classify_image(self, request: ClassificationRequest) -> dict[str, float]:
        """Score an image by asking the vision model for its labels.

        Args:
            request:
                The classification request.

        Returns:
            A score of one for every label the model names and zero for the others.
        """
        answer = self.complete(build_classification_prompt(request), self.resolve_model('classify_image'))
        return parse_labels(answer, request.labels, request.multi_label)

    def classify_audio(self, request: ClassificationRequest) -> dict[str, float]:
        """Score audio by asking an audio model for its labels.

        Args:
            request:
                The classification request.

        Returns:
            A score of one for every label the model names and zero for the others.
        """
        answer = self.complete(build_classification_prompt(request), self.resolve_model('classify_audio'))
        return parse_labels(answer, request.labels, request.multi_label)

    def classify_video(self, request: ClassificationRequest) -> dict[str, float]:
        """Score sampled video frames by asking the vision model for their labels.

        Args:
            request:
                The classification request.

        Returns:
            A score of one for every label the model names and zero for the others.
        """
        prompt = build_classification_prompt(request, include_audio=self.sends_video_audio_now())
        answer = self.complete(prompt, self.resolve_model('classify_video'))
        return parse_labels(answer, request.labels, request.multi_label)

    def describe_audio(self, request: DescriptionRequest) -> str:
        """Describe audio by prompting an audio model.

        Args:
            request:
                The description request.

        Returns:
            The description.
        """
        return self.complete(build_description_prompt(request), self.resolve_model('describe_audio')).strip()

    def describe_video(self, request: DescriptionRequest) -> str:
        """Describe sampled video frames by prompting the vision model.

        Args:
            request:
                The description request.

        Returns:
            The description.
        """
        prompt = build_description_prompt(request, include_audio=self.sends_video_audio_now())
        return self.complete(prompt, self.resolve_model('describe_video')).strip()

    def extract_fields(self, request: ExtractionRequest) -> dict[str, Any]:
        """Extract fields by asking the model for a JSON object.

        Args:
            request:
                The extraction request.

        Returns:
            The value of every field, or `None` for a field the text does not contain.
        """
        answer = self.complete(build_extraction_prompt(request), self.resolve_model('extract_fields'))
        return parse_fields(answer, tuple(name for name, _ in request.fields))
