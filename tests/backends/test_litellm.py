"""Test the LiteLLM backend against a fake module."""

import numpy as np
import pytest
from PIL import Image

from skai.backends import LiteLLMBackend
from skai.classification import TextClassifier
from skai.core import MissingCredentialError
from skai.description import ImageDescriptor
from skai.embedding import TextEmbedder
from skai.text import TextSummarizer
from tests.fakes import Recorder, chat_response, install_module, namespace, round_trip


@pytest.fixture
def litellm(monkeypatch):
    """Install a fake LiteLLM module."""
    return install_module(
        monkeypatch,
        'litellm',
        completion=Recorder(lambda **kwargs: chat_response('cat')),
        embedding=Recorder(lambda **kwargs: namespace(data=[{'embedding': [1.0, 2.0]}])),
    )


def test_prompted_tasks_call_completion(litellm):
    """Send prompts with images to the default chat model."""
    assert TextClassifier(labels=['cat', 'dog'], backend='litellm').fit().predict(['meow']).tolist() == ['cat']
    assert litellm.completion.calls[-1]['model'] == 'anthropic/claude-haiku-5-5'
    ImageDescriptor(backend='litellm:gemini/gemini-2.5-flash').fit().predict([Image.new('RGB', (4, 4))])
    call = litellm.completion.calls[-1]
    assert call['model'] == 'gemini/gemini-2.5-flash'
    assert call['messages'][1]['content'][1]['type'] == 'image_url'


def test_embedding_reads_dict_and_object_items(litellm, monkeypatch):
    """Read embeddings returned as dicts or as objects."""
    np.testing.assert_allclose(TextEmbedder(backend='litellm').fit().transform(['a']), [[1.0, 2.0]])
    monkeypatch.setattr(litellm, 'embedding', Recorder(lambda **kwargs: namespace(data=[namespace(embedding=[3.0])])))
    np.testing.assert_allclose(TextEmbedder(dimensions=1, backend='litellm').fit().transform(['a']), [[3.0]])
    assert litellm.embedding.calls[-1]['dimensions'] == 1


def test_named_credential_and_base_are_passed(litellm, monkeypatch):
    """Pass the key read from `api_key_env` and the base URL to every call."""
    monkeypatch.setenv('MY_KEY', 'secret')
    backend = LiteLLMBackend(model='ollama/llama3.2', api_key_env='MY_KEY', api_base='http://localhost:11434')
    TextSummarizer(backend=backend).fit().predict(['text'])
    call = litellm.completion.calls[-1]
    assert (call['api_key'], call['api_base']) == ('secret', 'http://localhost:11434')


def test_missing_named_credential_fails_at_fit(litellm, monkeypatch):
    """Fail at `fit` when the named credential variable is not set."""
    monkeypatch.delenv('MY_KEY', raising=False)
    with pytest.raises(MissingCredentialError, match='MY_KEY'):
        TextSummarizer(backend=LiteLLMBackend(api_key_env='MY_KEY')).fit()


def test_unpickled_backend_imports_module_again(litellm):
    """Import LiteLLM again on first use after unpickling."""
    summarizer = round_trip(TextSummarizer(backend='litellm').fit())
    assert summarizer.predict(['text'])[0] == 'cat'
