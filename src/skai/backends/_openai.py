"""Run component tasks on the OpenAI API."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import base64
import io
import json
import time
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skai.core import (
    AnimationRequest,
    Audio,
    EmbeddingRequest,
    FineTuneError,
    FineTuneRequest,
    ImageEditRequest,
    ImageGenerationRequest,
    InvalidResponseError,
    Prompt,
    PromptedBackend,
    Segment,
    SpeechRequest,
    Transcript,
    TranscriptionRequest,
    Video,
    VideoGenerationRequest,
    build_image_prompt,
    encode_png,
    import_optional,
    read_env,
    to_chat_messages,
)

CHAT_MODEL = 'gpt-5-mini'
FINETUNE_MODEL = 'gpt-4.1-mini-2025-04-14'
DEFAULT_VOICE = 'alloy'
AUDIO_MODEL = 'gpt-audio-mini'
VIDEO_MODEL = 'sora-2'
FINISHED_STATUSES = frozenset({'succeeded', 'failed', 'cancelled'})
VIDEO_FINISHED_STATUSES = frozenset({'completed', 'failed'})
REFUSAL_STATUS_CODES = frozenset({400, 403})
VIDEO_SECONDS = (4, 8, 12)
VIDEO_SIZES = ((1280, 720), (720, 1280))
DEFAULT_MOTION_PROMPT = 'Animate this image with natural, gentle motion.'


def _closest_video_size(size: tuple[int, int]) -> tuple[int, int]:
    """Return the Sora size whose aspect ratio is closest to a width and height."""
    width, height = size
    return min(VIDEO_SIZES, key=lambda candidate: abs(candidate[0] / candidate[1] - width / height))


def _closest_video_seconds(duration: float) -> int:
    """Return the shortest Sora duration that covers a duration, or the longest one."""
    return next((seconds for seconds in VIDEO_SECONDS if seconds >= duration), VIDEO_SECONDS[-1])


def _to_png_file(image: Image.Image, name: str) -> tuple[str, bytes, str]:
    """Encode an image as a named PNG upload."""
    return name, encode_png(image), 'image/png'


def _to_alpha_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Turn a mask whose white marks the edit region into the transparent-region mask that OpenAI expects."""
    alpha = mask.convert('L').resize(size).point(lambda value: 255 - value)
    transparent = Image.new('RGBA', size, (0, 0, 0, 255))
    transparent.putalpha(alpha)
    return transparent


def _to_segments(response: Any, timestamps: str | None) -> tuple[Segment, ...]:
    """Read the timed segments or words of a verbose transcription response."""
    if timestamps == 'word':
        return tuple(Segment(float(word.start), float(word.end), word.word.strip()) for word in response.words)
    if timestamps == 'segment':
        return tuple(Segment(float(item.start), float(item.end), item.text.strip()) for item in response.segments)
    return ()


class OpenAIBackend(PromptedBackend):
    """Run component tasks on the OpenAI API.

    The backend supports text and image classification, summarization, translation, embeddings, speech synthesis,
    image generation, image description and transcription. It needs the `openai` extra.

    Args:
        model:
            The model to use. The default model of the task is used when it is `None`.

        api_key_env:
            The name of the environment variable that holds the API key.

        base_url:
            The base URL of an OpenAI-compatible API.

        client:
            A preconfigured `openai.OpenAI` client, used instead of creating one.

        voice:
            The voice used for speech synthesis when the request names none.

        finetune_model:
            The base model that fine-tuning starts from.

        finetune_timeout:
            The number of seconds to wait for a fine-tuning job before giving up.

        video_timeout:
            The number of seconds to wait for a video job before giving up.

        poll_interval:
            The number of seconds between checks of a fine-tuning or video job.

        options:
            Extra keyword arguments passed to every API call.

        max_concurrency:
            The maximum number of requests that run at the same time.
    """

    default_models: ClassVar[dict[str, str]] = {
        'classify_text': CHAT_MODEL,
        'summarize': CHAT_MODEL,
        'translate': CHAT_MODEL,
        'describe_image': CHAT_MODEL,
        'classify_image': CHAT_MODEL,
        'embed_text': 'text-embedding-3-small',
        'synthesize_speech': 'gpt-4o-mini-tts',
        'generate_image': 'gpt-image-1',
        'transcribe': 'whisper-1',
        'classify_audio': AUDIO_MODEL,
        'describe_audio': AUDIO_MODEL,
        'classify_video': CHAT_MODEL,
        'describe_video': CHAT_MODEL,
        'extract_fields': CHAT_MODEL,
        'edit_image': 'gpt-image-1',
        'generate_video': VIDEO_MODEL,
        'animate_image': VIDEO_MODEL,
    }
    supports_masks: ClassVar[bool] = True
    accepts_audio: ClassVar[bool] = True
    timestamp_granularities: ClassVar[frozenset[str]] = frozenset({'segment', 'word'})
    accepts_context: ClassVar[bool] = True
    shortens_embeddings: ClassVar[bool] = True

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key_env: str = 'OPENAI_API_KEY',
        base_url: str | None = None,
        client: Any = None,
        voice: str = DEFAULT_VOICE,
        finetune_model: str = FINETUNE_MODEL,
        finetune_timeout: float = 3600.0,
        video_timeout: float = 900.0,
        poll_interval: float = 15.0,
        options: dict | None = None,
        max_concurrency: int = 8,
    ) -> None:
        super().__init__(model, options=options, max_concurrency=max_concurrency)
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.client = client
        self.voice = voice
        self.finetune_model = finetune_model
        self.finetune_timeout = finetune_timeout
        self.video_timeout = video_timeout
        self.poll_interval = poll_interval

    @property
    def _api(self) -> Any:
        """The OpenAI client, created on first use."""
        if not hasattr(self, '_client'):
            self.connect()
        return self._client

    def connect(self) -> 'OpenAIBackend':
        """Use the given client, or create one with the API key read from `api_key_env`.

        Returns:
            The backend.
        """
        if self.client is not None:
            self._client = self.client
            return self
        openai = import_optional('openai', 'openai')
        self._client = openai.OpenAI(api_key=read_env(self.api_key_env), base_url=self.base_url)
        return self

    def complete(self, prompt: Prompt, model: str) -> str:
        """Return the answer of a chat model to a prompt.

        Args:
            prompt:
                The prompt.

            model:
                The model to prompt.

        Returns:
            The answer of the model.
        """
        options = {'response_format': {'type': 'json_object'}} if prompt.json_output else {}
        response = self._api.chat.completions.create(
            model=model,
            messages=to_chat_messages(prompt),
            **options,
            **self.call_options,
        )
        return response.choices[0].message.content or ''

    def embed_text(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Embed a text with an embedding model.

        Args:
            request:
                The embedding request.

        Returns:
            The embedding vector.
        """
        dimensions = {'dimensions': request.dimensions} if request.dimensions else {}
        response = self._api.embeddings.create(
            model=self.resolve_model('embed_text'),
            input=[request.sample],
            **dimensions,
            **self.call_options,
        )
        return np.asarray(response.data[0].embedding)

    def synthesize_speech(self, request: SpeechRequest) -> Audio:
        """Speak a text with a text-to-speech model.

        Args:
            request:
                The speech request.

        Returns:
            The speech.
        """
        response = self._api.audio.speech.create(
            model=self.resolve_model('synthesize_speech'),
            voice=request.voice if isinstance(request.voice, str) else self.voice,
            input=request.text,
            response_format='wav',
            **self.call_options,
        )
        return Audio.from_bytes(response.content)

    def generate_image(self, request: ImageGenerationRequest) -> Image.Image:
        """Generate an image with an image model.

        Args:
            request:
                The image generation request.

        Returns:
            The image.
        """
        width, height = request.size
        response = self._api.images.generate(
            model=self.resolve_model('generate_image'),
            prompt=build_image_prompt(request),
            size=f'{width}x{height}',
            n=1,
            **self.call_options,
        )
        image = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json)))
        image.load()
        return image

    def transcribe(self, request: TranscriptionRequest) -> Transcript:
        """Transcribe speech with a transcription model.

        Args:
            request:
                The transcription request.

        Returns:
            The transcript.
        """
        options: dict[str, Any] = {'language': request.language} if request.language else {}
        options |= {'prompt': ', '.join(request.vocabulary)} if request.vocabulary else {}
        if request.timestamps:
            options |= {'response_format': 'verbose_json', 'timestamp_granularities': [request.timestamps]}
        response = self._api.audio.transcriptions.create(
            model=self.resolve_model('transcribe'),
            file=('audio.wav', request.audio.to_bytes()),
            **options,
            **self.call_options,
        )
        return Transcript(response.text.strip(), _to_segments(response, request.timestamps))

    def _wait_for(self, job: Any, retrieve: Any, finished: frozenset[str], timeout: float, label: str) -> Any:
        """Poll a job until its status is one of `finished`, or raise `TimeoutError` naming it."""
        deadline = time.monotonic() + timeout
        while job.status not in finished:
            if time.monotonic() >= deadline:
                msg = f'{label} {job.id} did not finish within {timeout} seconds.'
                raise TimeoutError(msg)
            time.sleep(self.poll_interval)
            job = retrieve(job.id)
        return job

    def _create_video(self, model: str, prompt: str, seconds: int, size: tuple[int, int], **inputs: Any) -> Video:
        """Create a video job, wait for it, and download the MP4."""
        width, height = size
        job = self._api.videos.create(
            model=model,
            prompt=prompt,
            seconds=str(seconds),
            size=f'{width}x{height}',
            **inputs,
            **self.call_options,
        )
        job = self._wait_for(job, self._api.videos.retrieve, VIDEO_FINISHED_STATUSES, self.video_timeout, 'Video job')
        if job.status != 'completed':
            msg = f'Video job {job.id} ended with status `{job.status}` ({getattr(job, "error", None)}).'
            raise InvalidResponseError(msg)
        content = self._api.videos.download_content(job.id)
        return Video(data=content.read() if hasattr(content, 'read') else content.content)

    def _start_fine_tuning(self, training_data: bytes) -> Any:
        """Upload the training data and create the job, turning a refusal by OpenAI into `FineTuneError`."""
        try:
            training_file = self._api.files.create(file=('train.jsonl', training_data), purpose='fine-tune')
            return self._api.fine_tuning.jobs.create(model=self.finetune_model, training_file=training_file.id)
        except Exception as error:
            if getattr(error, 'status_code', None) not in REFUSAL_STATUS_CODES:
                raise
            reason = getattr(error, 'message', str(error))
            msg = f'OpenAI refused the fine-tuning job: {reason}'
            raise FineTuneError(msg) from error

    def fine_tune(self, request: FineTuneRequest) -> str:
        """Train a chat model on the prompts and answers, and wait until the job finishes.

        A job that does not finish within `finetune_timeout` seconds raises `TimeoutError` naming the job, whose model
        can be used later through the `model` parameter.

        Args:
            request:
                The fine-tuning request.

        Returns:
            The id of the fine-tuned model.

        Raises:
            FineTuneError:
                If OpenAI refuses the job, or the job fails or is cancelled.
        """
        lines = [
            json.dumps({'messages': [*to_chat_messages(prompt), {'role': 'assistant', 'content': answer}]})
            for prompt, answer in request.examples
        ]
        job = self._start_fine_tuning('\n'.join(lines).encode())
        retrieve = self._api.fine_tuning.jobs.retrieve
        job = self._wait_for(job, retrieve, FINISHED_STATUSES, self.finetune_timeout, 'Fine-tuning job')
        if job.status != 'succeeded':
            msg = f'Fine-tuning job {job.id} ended with status `{job.status}` ({job.error}).'
            raise FineTuneError(msg)
        return job.fine_tuned_model

    def edit_image(self, request: ImageEditRequest) -> Image.Image:
        """Edit an image with an image model, inside the mask when one is given.

        Args:
            request:
                The image edit request.

        Returns:
            The edited image, resized to the size of the input.
        """
        image = request.image.convert('RGBA')
        mask = {'mask': _to_png_file(_to_alpha_mask(request.mask, image.size), 'mask.png')} if request.mask else {}
        response = self._api.images.edit(
            model=self.resolve_model('edit_image'),
            image=_to_png_file(image, 'image.png'),
            prompt=request.instruction,
            **mask,
            **self.call_options,
        )
        edited = Image.open(io.BytesIO(base64.b64decode(response.data[0].b64_json))).convert('RGB')
        return edited if edited.size == request.image.size else edited.resize(request.image.size)

    def generate_video(self, request: VideoGenerationRequest) -> Video:
        """Generate a video with Sora, then fit it to the requested duration and size.

        Sora makes videos of 4, 8 or 12 seconds at a few sizes, so the closest are requested, and the video is trimmed
        and resized. Trimming and resizing need the `video` extra.

        Args:
            request:
                The video generation request.

        Returns:
            The video.
        """
        prompt = build_image_prompt(ImageGenerationRequest(request.prompt, request.size, request.style))
        prompt = f'{prompt}. Avoid: {request.negative_prompt}' if request.negative_prompt else prompt
        size = _closest_video_size(request.size)
        seconds = _closest_video_seconds(request.duration)
        video = self._create_video(self.resolve_model('generate_video'), prompt, seconds, size)
        return video.trim(request.duration).resize(request.size)

    def animate_image(self, request: AnimationRequest) -> Video:
        """Animate an image with Sora, using it as the first frame.

        Args:
            request:
                The animation request.

        Returns:
            The video, trimmed to the requested duration and resized to the size of the image.
        """
        size = _closest_video_size(request.image.size)
        reference = _to_png_file(request.image.convert('RGB').resize(size), 'image.png')
        video = self._create_video(
            self.resolve_model('animate_image'),
            request.motion_prompt or DEFAULT_MOTION_PROMPT,
            _closest_video_seconds(request.duration),
            size,
            input_reference=reference,
        )
        return video.trim(request.duration).resize(request.image.size)
