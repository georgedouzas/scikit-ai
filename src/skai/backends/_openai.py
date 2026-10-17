"""Run component tasks on the OpenAI API."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import base64
import io
from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skai.core import (
    Audio,
    EmbeddingRequest,
    ImageGenerationRequest,
    Prompt,
    PromptedBackend,
    Segment,
    SpeechRequest,
    Transcript,
    TranscriptionRequest,
    build_image_prompt,
    import_optional,
    read_env,
    to_data_url,
)

CHAT_MODEL = 'gpt-5-mini'
DEFAULT_VOICE = 'alloy'


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
    }
    timestamp_granularities: ClassVar[frozenset[str]] = frozenset({'segment', 'word'})

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key_env: str = 'OPENAI_API_KEY',
        base_url: str | None = None,
        client: Any = None,
        voice: str = DEFAULT_VOICE,
        options: dict | None = None,
        max_concurrency: int = 8,
    ) -> None:
        super().__init__(model, options=options, max_concurrency=max_concurrency)
        self.api_key_env = api_key_env
        self.base_url = base_url
        self.client = client
        self.voice = voice

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
        content: list[dict[str, Any]] = [{'type': 'text', 'text': prompt.user}]
        content += [{'type': 'image_url', 'image_url': {'url': to_data_url(image)}} for image in prompt.images]
        messages = [{'role': 'system', 'content': prompt.system}, {'role': 'user', 'content': content}]
        response = self._api.chat.completions.create(model=model, messages=messages, **self.call_options)
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
        if request.timestamps:
            options |= {'response_format': 'verbose_json', 'timestamp_granularities': [request.timestamps]}
        response = self._api.audio.transcriptions.create(
            model=self.resolve_model('transcribe'),
            file=('audio.wav', request.audio.to_bytes()),
            **options,
            **self.call_options,
        )
        return Transcript(response.text.strip(), _to_segments(response, request.timestamps))
