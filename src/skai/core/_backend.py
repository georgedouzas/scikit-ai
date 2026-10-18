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
    build_detection_prompt,
    build_extraction_prompt,
    build_image_transcription_prompt,
    build_moment_prompt,
    build_text_generation_prompt,
    build_thumbnail_prompt,
    parse_fields,
    parse_frame_number,
    parse_labels,
    parse_moments,
    parse_regions,
)
from ._requests import (
    AnimationRequest,
    AudioGenerationRequest,
    AudioOperationRequest,
    AvatarRequest,
    ClassificationRequest,
    DiarizationRequest,
    EmbeddingRequest,
    ExtractionRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    ImageOperationRequest,
    ImageTranscriptionRequest,
    LipSyncRequest,
    Moment,
    MomentRequest,
    Region,
    RegionRequest,
    SpeechRequest,
    TextGenerationRequest,
    ThumbnailRequest,
    Transcript,
    TranscriptionRequest,
    VideoFrames,
    VideoGenerationRequest,
)

PROMPTED_TASKS = frozenset(
    {
        'classify_text',
        'classify_image',
        'classify_audio',
        'classify_video',
        'classify_multimodal',
        'generate_text',
        'extract_fields',
        'transcribe_image',
        'detect_objects',
        'find_moments',
        'pick_thumbnail',
    },
)
TASKS = (
    'classify_text',
    'classify_image',
    'classify_audio',
    'classify_video',
    'classify_multimodal',
    'generate_text',
    'extract_fields',
    'transcribe',
    'transcribe_image',
    'embed_text',
    'embed_image',
    'embed_audio',
    'detect_objects',
    'segment_image',
    'diarize',
    'find_moments',
    'synthesize_speech',
    'generate_music',
    'generate_sound',
    'separate_audio',
    'enhance_audio',
    'convert_voice',
    'generate_image',
    'edit_image',
    'upscale_image',
    'remove_background',
    'pick_thumbnail',
    'generate_video',
    'animate_image',
    'animate_avatar',
    'lip_sync',
    'fine_tune',
)
INPUT_KINDS = ('text', 'image', 'audio', 'video', 'multimodal')


def input_kind_of(sample: object) -> str:
    """Return the input kind of a read sample.

    Args:
        sample:
            A text, image, audio, sampled video frames or multimodal sample.

    Returns:
        One of `'text'`, `'image'`, `'audio'`, `'video'` and `'multimodal'`.
    """
    if isinstance(sample, str):
        return 'text'
    if isinstance(sample, Audio):
        return 'audio'
    if isinstance(sample, VideoFrames | Video):
        return 'video'
    if isinstance(sample, dict):
        return 'multimodal'
    return 'image'


def _has_audio(sample: object) -> bool:
    """Return whether a read sample holds audio, directly or in one of its named parts."""
    if isinstance(sample, Audio):
        return True
    if isinstance(sample, dict):
        return any(_has_audio(value) for value in sample.values())
    return False


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
    pickled. A key of `default_models` is a task, or `'<task>:<input kind>'` for a default that depends on the input.

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
    supports_reference_images: ClassVar[bool] = False
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

    @classmethod
    def supports(cls, task: str, input_kind: str | None = None) -> bool:
        """Return whether the backend runs a task on samples of an input kind.

        Args:
            task:
                The name of the task.

            input_kind:
                The input kind of the samples, or `None` for any.

        Returns:
            Whether the task is supported.
        """
        return task in cls.supported_tasks()

    @property
    def call_options(self) -> dict[str, Any]:
        """The extra keyword arguments passed to the provider."""
        return dict(self.options or {})

    def resolve_model(self, task: str, input_kind: str | None = None) -> str:
        """Return the model that runs a task.

        Args:
            task:
                The name of the task.

            input_kind:
                The input kind of the sample, which selects a default model of `'<task>:<input kind>'` first.

        Returns:
            The `model` parameter, or the default model of the task.

        Raises:
            UnsupportedBackendError:
                If no model is given and the task has no default model.
        """
        if self.model is not None:
            return self.model
        key = f'{task}:{input_kind}'
        if input_kind is not None and key in self.default_models:
            return self.default_models[key]
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
                The request.

        Returns:
            A score per label.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def classify_image(self, request: ClassificationRequest) -> dict[str, float]:
        """Score an image against the labels of a request.

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

    def classify_multimodal(self, request: ClassificationRequest) -> dict[str, float]:
        """Score a multimodal sample against the labels of a request.

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

    def generate_text(self, request: TextGenerationRequest) -> str:
        """Produce text from a sample by following instructions.

        Args:
            request:
                The request.

        Returns:
            The text.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def extract_fields(self, request: ExtractionRequest) -> dict[str, Any]:
        """Extract named fields from a sample.

        Args:
            request:
                The request.

        Returns:
            The value of every field, or `None` for a field the sample does not contain.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        """Transcribe speech.

        Args:
            request:
                The request.

        Returns:
            The transcript.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def transcribe_image(self, request: ImageTranscriptionRequest) -> str:
        """Transcribe the text written in an image.

        Args:
            request:
                The request.

        Returns:
            The text.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def embed_text(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Embed a text.

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

    def detect_objects(self, request: RegionRequest) -> list[Region]:
        """Find the boxes of the regions of an image.

        Args:
            request:
                The request.

        Returns:
            The regions, without masks.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def segment_image(self, request: RegionRequest) -> list[Region]:
        """Find the masks of the regions of an image.

        Args:
            request:
                The request.

        Returns:
            The regions, with masks.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def diarize(self, request: DiarizationRequest) -> Transcript:
        """Find who speaks when, with what they say.

        Args:
            request:
                The request.

        Returns:
            The transcript, with a speaker per segment.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def find_moments(self, request: MomentRequest) -> list[Moment]:
        """Find the parts of audio or a video that match a description.

        Args:
            request:
                The request.

        Returns:
            The moments, in time order.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def synthesize_speech(self, request: SpeechRequest) -> Audio:
        """Speak a text.

        Args:
            request:
                The request.

        Returns:
            The speech.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def generate_music(self, request: AudioGenerationRequest) -> Audio:
        """Generate music from a description.

        Args:
            request:
                The request.

        Returns:
            The music.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def generate_sound(self, request: AudioGenerationRequest) -> Audio:
        """Generate a sound from a description.

        Args:
            request:
                The request.

        Returns:
            The sound.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def separate_audio(self, request: AudioOperationRequest) -> Audio:
        """Separate a stem of audio.

        Args:
            request:
                The request.

        Returns:
            The stem.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def enhance_audio(self, request: AudioOperationRequest) -> Audio:
        """Remove the noise of audio.

        Args:
            request:
                The request.

        Returns:
            The denoised audio.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def convert_voice(self, request: AudioOperationRequest) -> Audio:
        """Say the speech of audio in another voice.

        Args:
            request:
                The request.

        Returns:
            The converted speech.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def generate_image(self, request: ImageGenerationRequest) -> Image.Image:
        """Generate an image from a prompt.

        Args:
            request:
                The request.

        Returns:
            The image.

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

    def upscale_image(self, request: ImageOperationRequest) -> Image.Image:
        """Upscale an image.

        Args:
            request:
                The request.

        Returns:
            The upscaled image.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def remove_background(self, request: ImageOperationRequest) -> Image.Image:
        """Remove the background of an image.

        Args:
            request:
                The request.

        Returns:
            An RGBA image whose background is transparent.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError

    def pick_thumbnail(self, request: ThumbnailRequest) -> int:
        """Pick the frame that best represents a video.

        Args:
            request:
                The request.

        Returns:
            The index of the frame.

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

    def animate_avatar(self, request: AvatarRequest) -> Video:
        """Animate a portrait so that it speaks an audio track.

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

    def lip_sync(self, request: LipSyncRequest) -> Video:
        """Make a video speak new audio.

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
                The request.

        Returns:
            The id of the trained model.

        Raises:
            NotImplementedError:
                If the backend does not support the task.
        """
        raise NotImplementedError


class PromptedBackend(Backend):
    """Run analysis and text tasks by prompting a chat, vision or audio model.

    A subclass implements `complete`, and inherits classification, text generation, field extraction, image
    transcription, object detection, moment finding and thumbnail picking for every input kind its models accept:
    text, images and video frames always, and audio only when the subclass sets `accepts_audio`. Video requests
    include the audio track only when it sets `sends_video_audio`.

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
        """Return the tasks that the backend implements, without `classify_audio` when it does not accept audio.

        Returns:
            The names of the tasks.
        """
        tasks = super().supported_tasks()
        if cls.accepts_audio or cls.classify_audio is not PromptedBackend.classify_audio:
            return tasks
        return tasks - {'classify_audio'}

    @classmethod
    def supports(cls, task: str, input_kind: str | None = None) -> bool:
        """Return whether the backend runs a task on samples of an input kind, refusing prompted audio without audio.

        Args:
            task:
                The name of the task.

            input_kind:
                The input kind of the samples, or `None` for any.

        Returns:
            Whether the task is supported.
        """
        if not super().supports(task, input_kind):
            return False
        prompted = task in PROMPTED_TASKS and getattr(cls, task) is getattr(PromptedBackend, task)
        return not (prompted and input_kind == 'audio' and not cls.accepts_audio)

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

    def _prompt(self, task: str, prompt: Prompt, sample: object) -> str:
        """Send a prompt for a task, choosing the model by the input kind of the sample, and audio models for audio."""
        has_audio = _has_audio(sample)
        if has_audio and not self.accepts_audio:
            msg = f'{type(self).__name__} does not accept audio for `{task}`.'
            raise UnsupportedBackendError(msg)
        return self.complete(prompt, self.resolve_model(task, 'audio' if has_audio else input_kind_of(sample)))

    def _classify(self, task: str, request: ClassificationRequest) -> dict[str, float]:
        """Score a sample by asking the model for its labels."""
        prompt = build_classification_prompt(request, include_audio=self.sends_video_audio_now())
        answer = self._prompt(task, prompt, request.sample)
        return parse_labels(answer, request.labels, request.multi_label)

    def classify_text(self, request: ClassificationRequest) -> dict[str, float]:
        """Score a text by asking the model for its labels.

        Args:
            request:
                The classification request.

        Returns:
            A score of one for every label the model names and zero for the others.
        """
        return self._classify('classify_text', request)

    def classify_image(self, request: ClassificationRequest) -> dict[str, float]:
        """Score an image by asking the vision model for its labels.

        Args:
            request:
                The classification request.

        Returns:
            A score of one for every label the model names and zero for the others.
        """
        return self._classify('classify_image', request)

    def classify_audio(self, request: ClassificationRequest) -> dict[str, float]:
        """Score audio by asking an audio model for its labels.

        Args:
            request:
                The classification request.

        Returns:
            A score of one for every label the model names and zero for the others.
        """
        return self._classify('classify_audio', request)

    def classify_video(self, request: ClassificationRequest) -> dict[str, float]:
        """Score sampled video frames by asking the vision model for their labels.

        Args:
            request:
                The classification request.

        Returns:
            A score of one for every label the model names and zero for the others.
        """
        return self._classify('classify_video', request)

    def classify_multimodal(self, request: ClassificationRequest) -> dict[str, float]:
        """Score a multimodal sample by asking the model for its labels.

        Args:
            request:
                The classification request.

        Returns:
            A score of one for every label the model names and zero for the others.
        """
        return self._classify('classify_multimodal', request)

    def generate_text(self, request: TextGenerationRequest) -> str:
        """Produce text from a sample by prompting the model with the instructions.

        Args:
            request:
                The text generation request.

        Returns:
            The text.
        """
        prompt = build_text_generation_prompt(request, include_audio=self.sends_video_audio_now())
        return self._prompt('generate_text', prompt, request.sample).strip()

    def extract_fields(self, request: ExtractionRequest) -> dict[str, Any]:
        """Extract fields by asking the model for a JSON object.

        Args:
            request:
                The extraction request.

        Returns:
            The value of every field, or `None` for a field the sample does not contain.
        """
        prompt = build_extraction_prompt(request, include_audio=self.sends_video_audio_now())
        answer = self._prompt('extract_fields', prompt, request.sample)
        return parse_fields(answer, tuple(name for name, _ in request.fields))

    def transcribe_image(self, request: ImageTranscriptionRequest) -> str:
        """Transcribe the text of an image by prompting the vision model.

        Args:
            request:
                The image transcription request.

        Returns:
            The text.
        """
        prompt = build_image_transcription_prompt(request)
        return self._prompt('transcribe_image', prompt, request.image).strip()

    def detect_objects(self, request: RegionRequest) -> list[Region]:
        """Find the boxes of regions by asking the vision model for them as JSON.

        Args:
            request:
                The region request.

        Returns:
            The regions, without masks.
        """
        answer = self._prompt('detect_objects', build_detection_prompt(request), request.image)
        return parse_regions(answer, request.image.size)

    def find_moments(self, request: MomentRequest) -> list[Moment]:
        """Find moments by asking the model for their times as JSON.

        Args:
            request:
                The moment request.

        Returns:
            The moments, in time order.
        """
        prompt = build_moment_prompt(request, include_audio=self.sends_video_audio_now())
        answer = self._prompt('find_moments', prompt, request.sample)
        return parse_moments(answer, request.sample.duration)

    def pick_thumbnail(self, request: ThumbnailRequest) -> int:
        """Pick the frame that best represents a video by asking the vision model.

        Args:
            request:
                The thumbnail request.

        Returns:
            The index of the frame.
        """
        answer = self._prompt('pick_thumbnail', build_thumbnail_prompt(request), request.frames.frames[0])
        return parse_frame_number(answer, len(request.frames.frames))
