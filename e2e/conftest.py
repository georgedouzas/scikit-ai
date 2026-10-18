"""Load credentials from `.env`, skip without them, and record provider responses without secrets."""

import json
import os
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from dotenv import load_dotenv

from skai.backends import OpenAIBackend
from skai.core import import_optional

ROOT = Path(__file__).parents[1]
PAYLOADS = ROOT / 'tests' / 'payloads' / 'openai'
MAX_STRING = 500
TIMEOUT = 300.0
SHOT_FRAMES = 12

load_dotenv(ROOT / '.env', override=False)


def _sanitize(value: Any) -> Any:
    """Shorten long strings, such as base64 media, so recordings stay small."""
    if isinstance(value, str) and len(value) > MAX_STRING:
        return f'<{len(value)} characters>'
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _body_of(content: bytes, content_type: str) -> Any:
    """Return a JSON body as data, and any other body as its type and size."""
    if 'json' in content_type:
        return _sanitize(json.loads(content))
    return {'content_type': content_type, 'bytes': len(content)}


class Recorder:
    """Write the method, path and bodies of every exchange of a test, and never the headers."""

    def __init__(self, name: str) -> None:
        """Start the recording of one test."""
        self.name = re.sub(r'[^\w-]+', '-', name).strip('-')
        self.count = 0

    def __call__(self, response: httpx.Response) -> None:
        """Record one exchange."""
        response.read()
        request = response.request
        request_type = request.headers.get('content-type', '')
        record = {
            'method': request.method,
            'path': request.url.path,
            'status': response.status_code,
            'request': _body_of(request.content, request_type) if 'json' in request_type else None,
            'response': _body_of(response.content, response.headers.get('content-type', '')),
        }
        self.count += 1
        PAYLOADS.mkdir(parents=True, exist_ok=True)
        path = PAYLOADS / f'{self.name}-{self.count}.json'
        path.write_text(json.dumps(record, indent=2, sort_keys=True) + '\n')


@pytest.fixture(scope='session', autouse=True)
def require_openai_key() -> None:
    """Skip the session when no OpenAI key is available."""
    if not os.environ.get('OPENAI_API_KEY'):
        pytest.skip('Set OPENAI_API_KEY in `.env` or the environment to run the end-to-end tests.')


@pytest.fixture
def openai_backend(request: pytest.FixtureRequest) -> Iterator[OpenAIBackend]:
    """Return an OpenAI backend whose client records every response of the test."""
    openai = import_optional('openai', 'openai')
    hooks = {'response': [Recorder(request.node.name)]}
    with httpx.Client(event_hooks=hooks, timeout=TIMEOUT) as http_client:
        yield OpenAIBackend(client=openai.OpenAI(http_client=http_client))


@pytest.fixture
def litellm_backend(request: pytest.FixtureRequest) -> Iterator[Any]:
    """Return a LiteLLM backend whose OpenAI routes go through a client that records every response."""
    openai = import_optional('openai', 'openai')
    from skai.backends import LiteLLMBackend  # noqa: PLC0415 - imported after the credentials are loaded

    hooks = {'response': [Recorder(request.node.name)]}
    with httpx.Client(event_hooks=hooks, timeout=TIMEOUT) as http_client:
        yield LiteLLMBackend(options={'client': openai.OpenAI(http_client=http_client)})


@pytest.fixture(scope='session')
def hf_enabled() -> bool:
    """Return whether the Hugging Face tests, which download models, were requested."""
    return os.environ.get('SKAI_E2E_HF') == '1'


@pytest.fixture(scope='session')
def shared_images(require_openai_key: None) -> list[Any]:
    """Generate an apple and a car image once for every test that needs them."""
    openai = import_optional('openai', 'openai')
    from skai.generation import TextToImageGenerator  # noqa: PLC0415 - imported after the credentials are loaded

    with httpx.Client(event_hooks={'response': [Recorder('shared-images')]}, timeout=TIMEOUT) as http_client:
        backend = OpenAIBackend(client=openai.OpenAI(http_client=http_client))
        prompts = ['A single red apple on a white table', 'A small blue car parked on an empty street']
        return list(TextToImageGenerator(size=(1024, 1024), backend=backend).fit().predict(prompts))


@pytest.fixture(scope='session')
def shared_speech(require_openai_key: None) -> Any:
    """Synthesize one sentence of speech once for every test that needs it."""
    openai = import_optional('openai', 'openai')
    from skai.generation import TextToAudioGenerator  # noqa: PLC0415 - imported after the credentials are loaded

    with httpx.Client(event_hooks={'response': [Recorder('shared-speech')]}, timeout=TIMEOUT) as http_client:
        backend = OpenAIBackend(client=openai.OpenAI(http_client=http_client))
        sentence = 'Good morning, the apple harvest starts today.'
        return TextToAudioGenerator(backend=backend).fit().predict([sentence])[0]


def _frames_of(image: Any, count: int, side: int = 320) -> np.ndarray:
    """Return `count` copies of an image as square RGB frames."""
    frame = np.asarray(image.convert('RGB').resize((side, side)))
    return np.repeat(frame[np.newaxis], count, axis=0)


@pytest.fixture(scope='session')
def apple_video(shared_images: list[Any], shared_speech: Any) -> Any:
    """Build, without a provider, a video of the generated apple that speaks the synthesized sentence."""
    from skai.core import Video  # noqa: PLC0415 - imported after the credentials are loaded

    return Video(_frames_of(shared_images[0], 25), fps=25 / shared_speech.duration, audio=shared_speech)


@pytest.fixture(scope='session')
def scenes_video(shared_images: list[Any], shared_speech: Any) -> Any:
    """Build, without a provider, a video of an apple shot then a car shot that speaks the synthesized sentence."""
    from skai.core import Video  # noqa: PLC0415 - imported after the credentials are loaded

    apple, car = shared_images
    frames = np.concatenate([_frames_of(apple, SHOT_FRAMES), _frames_of(car, SHOT_FRAMES)])
    return Video(frames, fps=2 * SHOT_FRAMES / shared_speech.duration, audio=shared_speech)


@pytest.fixture(scope='session')
def video_enabled() -> bool:
    """Return whether the costly video generation tests were requested."""
    return os.environ.get('SKAI_E2E_VIDEO') == '1'


@pytest.fixture(scope='session')
def finetune_enabled() -> bool:
    """Return whether the slow and costly fine-tuning tests were requested."""
    return os.environ.get('SKAI_E2E_FINETUNE') == '1'
