"""Fake provider modules and clients that record calls instead of reaching the network."""

import io
import pickle  # nosec B403 - the tests pickle only objects they create
import sys
import types
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
from PIL import Image

from skai.core import Audio, PromptedBackend


class Recorder:
    """Record the calls of a fake method and answer them with a function."""

    def __init__(self, answer: Callable[..., Any]) -> None:
        """Store the function that answers the calls."""
        self.answer = answer
        self.calls: list[dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Record a call and return the answer."""
        self.calls.append({'args': args, **kwargs})
        return self.answer(*args, **kwargs)


def namespace(**attributes: Any) -> types.SimpleNamespace:
    """Return an object with the given attributes."""
    return types.SimpleNamespace(**attributes)


def install_module(monkeypatch: pytest.MonkeyPatch, name: str, **attributes: Any) -> types.ModuleType:
    """Install a fake module under a name for the duration of a test."""
    module = types.ModuleType(name)
    for key, value in attributes.items():
        setattr(module, key, value)
    monkeypatch.setitem(sys.modules, name, module)
    return module


def chat_response(text: str) -> types.SimpleNamespace:
    """Return a chat completion response that holds one answer."""
    return namespace(choices=[namespace(message=namespace(content=text))])


def round_trip(obj: Any) -> Any:
    """Pickle an object and load it back."""
    return pickle.loads(pickle.dumps(obj))  # noqa: S301 # nosec B301 - loads the bytes the test just pickled


def png_of(size: tuple[int, int]) -> bytes:
    """Return the PNG bytes of a black image."""
    buffer = io.BytesIO()
    Image.new('RGB', size).save(buffer, format='PNG')
    return buffer.getvalue()


def wav_of(seconds: float) -> bytes:
    """Return the WAV bytes of silence."""
    return Audio(np.zeros(int(16000 * seconds), dtype=np.float32), 16000).to_bytes()


class ScriptedBackend(PromptedBackend):
    """Answer every prompt with a fixed answer and keep the prompts."""

    default_models = dict.fromkeys(
        ['classify_audio', 'classify_video', 'describe_audio', 'describe_video', 'extract_fields'],
        'scripted',
    )
    accepts_audio = True
    sends_video_audio = True

    def __init__(
        self,
        model: str | None = None,
        *,
        answer: str = '',
        options: dict | None = None,
        max_concurrency: int = 1,
    ) -> None:
        """Store the fixed answer."""
        super().__init__(model, options=options, max_concurrency=max_concurrency)
        self.answer = answer

    def complete(self, prompt: Any, model: str) -> str:
        """Keep the prompt and return the fixed answer."""
        self.prompts = [*getattr(self, 'prompts', []), prompt]
        return self.answer


class SilentBackend(ScriptedBackend):
    """Accept neither audio prompts nor the audio of videos."""

    accepts_audio = False
    sends_video_audio = False
