"""Test the LiteLLM backend against a fake module."""

import numpy as np
import pytest
from PIL import Image

from skai.backends import LiteLLMBackend
from skai.classification import AudioClassifier, TextClassifier, VideoClassifier
from skai.core import MissingCredentialError, Video
from skai.embedding import ImageEmbedder, TextEmbedder
from skai.extraction import TextExtractor
from skai.generation import ImageToTextGenerator, TextToTextGenerator
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
    descriptor = ImageToTextGenerator(instructions='Describe the image.', backend='litellm:gemini/gemini-2.5-flash')
    descriptor.fit().predict([Image.new('RGB', (4, 4))])
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
    TextToTextGenerator(instructions='Summarize the text.', backend=backend).fit().predict(['text'])
    call = litellm.completion.calls[-1]
    assert (call['api_key'], call['api_base']) == ('secret', 'http://localhost:11434')


def test_missing_named_credential_fails_at_fit(litellm, monkeypatch):
    """Fail at `fit` when the named credential variable is not set."""
    monkeypatch.delenv('MY_KEY', raising=False)
    with pytest.raises(MissingCredentialError, match='MY_KEY'):
        TextToTextGenerator(instructions='Summarize the text.', backend=LiteLLMBackend(api_key_env='MY_KEY')).fit()


def test_unpickled_backend_imports_module_again(litellm):
    """Import LiteLLM again on first use after unpickling."""
    summarizer = round_trip(TextToTextGenerator(instructions='Summarize the text.', backend='litellm').fit())
    assert summarizer.predict(['text'])[0] == 'cat'


def test_media_tasks_send_audio_and_frames(litellm, audio):
    """Send audio parts and video frames with their audio track to the default media model."""
    AudioClassifier(labels=['cat', 'dog'], backend='litellm').fit().predict([audio])
    call = litellm.completion.calls[-1]
    assert call['model'] == 'gemini/gemini-2.5-flash'
    assert call['messages'][1]['content'][-1]['type'] == 'input_audio'
    video = Video(np.zeros((10, 4, 4, 3), dtype=np.uint8), fps=5, audio=audio)
    VideoClassifier(labels=['cat', 'dog'], n_frames=2, backend='litellm').fit().predict([video])
    types = [part['type'] for part in litellm.completion.calls[-1]['messages'][1]['content']]
    assert types == ['text', 'image_url', 'image_url', 'input_audio']


def test_extraction_uses_json_mode(litellm, monkeypatch):
    """Ask for a JSON object and read it into the fields."""
    monkeypatch.setattr(litellm, 'completion', Recorder(lambda **kwargs: chat_response('{"name": "Ada"}')))
    records = TextExtractor(fields={'name': 'person name'}, backend='litellm').fit().predict(['Ada.'])
    assert records.tolist() == [{'name': 'Ada'}]
    assert litellm.completion.calls[-1]['response_format'] == {'type': 'json_object'}


def test_image_embeddings_send_data_urls(litellm):
    """Embed images as data URLs with the multimodal embedding model."""
    ImageEmbedder(backend='litellm').fit().transform([Image.new('RGB', (4, 4))])
    call = litellm.embedding.calls[-1]
    assert call['model'] == 'cohere/embed-v4.0'
    assert call['input'][0].startswith('data:image/png;base64,')


def test_video_audio_can_be_turned_off(litellm, audio):
    """Leave the audio track out of video requests when `video_audio=False`."""
    video = Video(np.zeros((10, 4, 4, 3), dtype=np.uint8), fps=5, audio=audio)
    backend = LiteLLMBackend(model='openai/gpt-5-mini', video_audio=False)
    VideoClassifier(labels=['cat', 'dog'], n_frames=2, backend=backend).fit().predict([video])
    types = [part['type'] for part in litellm.completion.calls[-1]['messages'][1]['content']]
    assert types == ['text', 'image_url', 'image_url']
