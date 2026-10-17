"""Run generation tasks on models hosted by Replicate."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

import io
from typing import Any, ClassVar

import httpx
from PIL import Image

from skai.core import (
    Audio,
    AvatarRequest,
    Backend,
    ImageGenerationRequest,
    InvalidResponseError,
    SpeechRequest,
    Video,
    build_image_prompt,
    encode_png,
    import_optional,
    read_env,
)

DOWNLOAD_TIMEOUT = 300.0
DEFAULT_VOICE = 'af_bella'
ASPECT_RATIOS = ('1:1', '16:9', '21:9', '3:2', '2:3', '4:5', '5:4', '3:4', '4:3', '9:16', '9:21')


def _to_file(data: bytes, name: str) -> io.BytesIO:
    """Wrap bytes in a named file object that the Replicate client uploads."""
    file = io.BytesIO(data)
    file.name = name
    return file


def _closest_aspect_ratio(size: tuple[int, int]) -> str:
    """Return the supported aspect ratio closest to a width and height."""
    width, height = size

    def distance(ratio: str) -> float:
        """Return how far an aspect ratio is from the requested one."""
        numerator, denominator = (int(part) for part in ratio.split(':'))
        return abs(numerator / denominator - width / height)

    return min(ASPECT_RATIOS, key=distance)


def _read_output(output: Any) -> bytes:
    """Return the bytes of the first file that a Replicate model returns."""
    if isinstance(output, list | tuple):
        if not output:
            msg = 'The Replicate model returned no output.'
            raise InvalidResponseError(msg)
        output = output[0]
    if hasattr(output, 'read'):
        return output.read()
    if isinstance(output, str):
        response = httpx.get(output, timeout=DOWNLOAD_TIMEOUT, follow_redirects=True)
        response.raise_for_status()
        return response.content
    msg = f'Cannot read the Replicate output of type {type(output).__name__}.'
    raise InvalidResponseError(msg)


class ReplicateBackend(Backend):
    """Run talking avatars, image generation and speech synthesis on models hosted by Replicate.

    The default models are open-source. A model is named `owner/name` or `owner/name:version`, and a version should be
    pinned for reproducible results. Inputs are sent under the names the default models use, and `options` adds or
    overrides inputs for other models. The backend needs the `replicate` extra.

    Args:
        model:
            The model to use. The default model of the task is used when it is `None`.

        api_token_env:
            The name of the environment variable that holds the API token.

        client:
            A preconfigured `replicate.Client`, used instead of creating one.

        options:
            Extra inputs sent to every model run.

        max_concurrency:
            The maximum number of requests that run at the same time.
    """

    default_models: ClassVar[dict[str, str]] = {
        'animate_avatar': 'cjwbw/sadtalker',
        'generate_image': 'black-forest-labs/flux-schnell',
        'synthesize_speech': 'jaaari/kokoro-82m',
        'clone_voice': 'lucataco/xtts-v2',
    }
    supports_voice_cloning: ClassVar[bool] = True

    def __init__(
        self,
        model: str | None = None,
        *,
        api_token_env: str = 'REPLICATE_API_TOKEN',  # noqa: S107 - the default names an environment variable
        client: Any = None,
        options: dict | None = None,
        max_concurrency: int = 4,
    ) -> None:
        super().__init__(model, options=options, max_concurrency=max_concurrency)
        self.api_token_env = api_token_env
        self.client = client

    @property
    def _api(self) -> Any:
        """The Replicate client, created on first use."""
        if not hasattr(self, '_client'):
            self.connect()
        return self._client

    def _run_model(self, model: str, inputs: dict[str, Any]) -> bytes:
        """Run a model and return the bytes of its output file."""
        return _read_output(self._api.run(model, input=inputs | self.call_options))

    def connect(self) -> 'ReplicateBackend':
        """Use the given client, or create one with the API token read from `api_token_env`.

        Returns:
            The backend.
        """
        if self.client is not None:
            self._client = self.client
            return self
        replicate = import_optional('replicate', 'replicate')
        self._client = replicate.Client(api_token=read_env(self.api_token_env))
        return self

    def synthesize_speech(self, request: SpeechRequest) -> Audio:
        """Speak a text, cloning the voice when the request holds reference audio.

        Args:
            request:
                The speech request.

        Returns:
            The speech.
        """
        if isinstance(request.voice, Audio):
            model = self.resolve_model('clone_voice')
            speaker = _to_file(request.voice.to_bytes(), 'speaker.wav')
            inputs: dict[str, Any] = {'text': request.text, 'speaker': speaker, 'language': request.language or 'en'}
        else:
            model = self.resolve_model('synthesize_speech')
            inputs = {'text': request.text, 'voice': request.voice or DEFAULT_VOICE}
        return Audio.from_bytes(self._run_model(model, inputs))

    def generate_image(self, request: ImageGenerationRequest) -> Image.Image:
        """Generate an image and resize it to the requested size.

        Args:
            request:
                The image generation request.

        Returns:
            The image.
        """
        inputs: dict[str, Any] = {
            'prompt': build_image_prompt(request),
            'aspect_ratio': _closest_aspect_ratio(request.size),
            'output_format': 'png',
        }
        inputs |= {'seed': request.seed} if request.seed is not None else {}
        inputs |= {'negative_prompt': request.negative_prompt} if request.negative_prompt else {}
        image = Image.open(io.BytesIO(self._run_model(self.resolve_model('generate_image'), inputs))).convert('RGB')
        return image if image.size == request.size else image.resize(request.size)

    def animate_avatar(self, request: AvatarRequest) -> Video:
        """Animate a portrait with a talking-head model.

        Args:
            request:
                The avatar request.

        Returns:
            The video.
        """
        inputs = {
            'source_image': _to_file(encode_png(request.avatar.convert('RGB')), 'avatar.png'),
            'driven_audio': _to_file(request.audio.to_bytes(), 'audio.wav'),
        }
        return Video(data=self._run_model(self.resolve_model('animate_avatar'), inputs))
