"""Run prompted component tasks on the providers that LiteLLM routes to."""

# Author: Georgios Douzas <gdouzas@icloud.com>
# License: MIT

from typing import Any, ClassVar

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from skai.core import (
    EmbeddingRequest,
    Prompt,
    PromptedBackend,
    import_optional,
    read_env,
    to_chat_messages,
    to_data_url,
)

CHAT_MODEL = 'anthropic/claude-haiku-5-5'
MEDIA_MODEL = 'gemini/gemini-2.5-flash'


class LiteLLMBackend(PromptedBackend):
    """Run prompted tasks and embeddings on any provider that LiteLLM supports.

    Models are named as LiteLLM names them, for example `anthropic/claude-haiku-5-5`, `gemini/gemini-2.5-flash` or
    `ollama/llama3.2`. LiteLLM reads the standard credential variable of each provider. The backend needs the `litellm`
    extra.

    Args:
        model:
            The model to use. The default model of the task is used when it is `None`.

        api_key_env:
            The name of an environment variable that holds the API key, used instead of the provider's standard one.

        api_base:
            The base URL of the provider's API.

        options:
            Extra keyword arguments passed to every LiteLLM call.

        max_concurrency:
            The maximum number of requests that run at the same time.
    """

    default_models: ClassVar[dict[str, str]] = {
        'classify_text': CHAT_MODEL,
        'summarize': CHAT_MODEL,
        'translate': CHAT_MODEL,
        'describe_image': CHAT_MODEL,
        'classify_image': CHAT_MODEL,
        'embed_text': 'openai/text-embedding-3-small',
        'classify_audio': MEDIA_MODEL,
        'describe_audio': MEDIA_MODEL,
        'classify_video': MEDIA_MODEL,
        'describe_video': MEDIA_MODEL,
        'extract_fields': CHAT_MODEL,
        'embed_image': 'cohere/embed-v4.0',
    }
    accepts_audio: ClassVar[bool] = True
    sends_video_audio: ClassVar[bool] = True

    shortens_embeddings: ClassVar[bool] = True

    def __init__(
        self,
        model: str | None = None,
        *,
        api_key_env: str | None = None,
        api_base: str | None = None,
        options: dict | None = None,
        max_concurrency: int = 8,
    ) -> None:
        super().__init__(model, options=options, max_concurrency=max_concurrency)
        self.api_key_env = api_key_env
        self.api_base = api_base

    @property
    def _litellm(self) -> Any:
        """The LiteLLM module, imported on first use."""
        if not hasattr(self, '_module'):
            self.connect()
        return self._module

    def _credentials(self) -> dict[str, str]:
        """Return the API key and base URL arguments of a LiteLLM call."""
        credentials = {'api_key': read_env(self.api_key_env)} if self.api_key_env else {}
        return credentials | ({'api_base': self.api_base} if self.api_base else {})

    def connect(self) -> 'LiteLLMBackend':
        """Import LiteLLM and check that the credential variable is set.

        Returns:
            The backend.
        """
        self._module = import_optional('litellm', 'litellm')
        self._credentials()
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
        response = self._litellm.completion(
            model=model,
            messages=to_chat_messages(prompt),
            **options,
            **self._credentials(),
            **self.call_options,
        )
        return response.choices[0].message.content or ''

    def _embed(self, task: str, sample: str, dimensions: int | None) -> NDArray[np.floating]:
        """Embed one text or data URL with the embedding model of a task."""
        options = {'dimensions': dimensions} if dimensions else {}
        response = self._litellm.embedding(
            model=self.resolve_model(task),
            input=[sample],
            **options,
            **self._credentials(),
            **self.call_options,
        )
        item = response.data[0]
        return np.asarray(item['embedding'] if isinstance(item, dict) else item.embedding)

    def embed_text(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Embed a text with an embedding model.

        Args:
            request:
                The embedding request.

        Returns:
            The embedding vector.
        """
        return self._embed('embed_text', str(request.sample), request.dimensions)

    def embed_image(self, request: EmbeddingRequest) -> NDArray[np.floating]:
        """Embed an image, sent as a data URL, with a multimodal embedding model.

        Args:
            request:
                The embedding request.

        Returns:
            The embedding vector.
        """
        assert isinstance(request.sample, Image.Image)
        return self._embed('embed_image', to_data_url(request.sample), request.dimensions)
