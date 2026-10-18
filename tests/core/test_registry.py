"""Test selecting, configuring and registering backends."""

import threading
from typing import ClassVar

import numpy as np
import pytest
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV

from skai.backends import DummyBackend, HFBackend, OpenAIBackend
from skai.classification import TextClassifier
from skai.core import (
    Backend,
    MissingCredentialError,
    MissingExtraError,
    PromptedBackend,
    UnsupportedBackendError,
    list_backends,
    register,
)
from skai.generation import AudioToTextGenerator, AudioToVideoGenerator, TextToTextGenerator
from tests.fakes import install_module, namespace, round_trip


def summarizer(**params):
    """Return a text generator that summarizes."""
    return TextToTextGenerator(instructions='Summarize the text.', **params)


class RecordingBackend(DummyBackend):
    """Record the model that classifies every text."""

    def classify_text(self, request):
        """Return a score of one for the first label when the model is `first`."""
        scores = dict.fromkeys(request.labels, 0.0)
        scores[request.labels[0] if self.model == 'first' else request.labels[-1]] = 1.0
        return scores


def test_backend_string_selects_provider_and_model():
    """Parse `'<provider>:<model>'` into the provider and its model."""
    generator = summarizer(backend='dummy:tiny').fit()
    assert isinstance(generator.backend_, DummyBackend)
    assert generator.backend_.model == 'tiny'


def test_backend_string_uses_provider_default_model(monkeypatch):
    """Use the documented default model of the provider for a provider-only string."""
    install_module(monkeypatch, 'transformers', pipeline=None)
    transcriber = AudioToTextGenerator(backend='hf').fit()
    assert transcriber.backend_.resolve_model('transcribe') == HFBackend.default_models['transcribe']


def test_backend_object_is_cloned():
    """Clone a backend object so the component never changes it."""
    backend = DummyBackend(model='m')
    generator = summarizer(backend=backend).fit()
    assert generator.backend_ is not backend
    assert generator.backend_.get_params() == backend.get_params()


def test_grid_search_tunes_nested_backend_parameters():
    """Search `backend__model` like any nested scikit-learn parameter."""
    X = np.array(['a', 'b', 'c', 'd'] * 3)
    y = np.array(['x', 'y'] * 6)
    search = GridSearchCV(
        TextClassifier(backend=RecordingBackend()),
        {'backend__model': ['first', 'last']},
        cv=2,
    )
    search.fit(X, y)
    assert [params['backend__model'] for params in search.cv_results_['params']] == ['first', 'last']


def test_default_backend_runs_locally(monkeypatch):
    """Use the local Hugging Face backend when `backend=None`, without credentials."""
    install_module(monkeypatch, 'transformers', pipeline=None)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    assert isinstance(summarizer().fit().backend_, HFBackend)


def test_default_backend_without_extra_names_install_command(monkeypatch):
    """Name the pip command when the extra of the default backend is missing."""
    monkeypatch.setitem(__import__('sys').modules, 'transformers', None)
    with pytest.raises(MissingExtraError, match=r'pip install scikit-ai\[hf\]'):
        summarizer().fit()
    assert issubclass(MissingExtraError, ImportError)


def test_unknown_provider_lists_supported_providers():
    """Reject an unknown provider and list the providers that support the task."""
    with pytest.raises(UnsupportedBackendError, match=r'Unknown backend `nope`.*dummy'):
        summarizer(backend='nope').fit()


def test_provider_without_task_lists_supported_providers():
    """Reject a provider that does not support the task and list those that do."""
    match = r"OpenAIBackend does not support `animate_avatar` on audio samples.*\['dummy', 'replicate'\]"
    with pytest.raises(UnsupportedBackendError, match=match):
        AudioToVideoGenerator(avatar=np.zeros((4, 4, 3), dtype=np.uint8), backend='openai').fit()
    assert issubclass(UnsupportedBackendError, ValueError)


def test_credentials_are_read_from_named_variable(monkeypatch):
    """Create the client with the value of the variable named by `api_key_env`."""
    created = {}
    install_module(monkeypatch, 'openai', OpenAI=lambda **kwargs: created.update(kwargs) or namespace())
    monkeypatch.setenv('MY_OPENAI_KEY', 'sk-123')
    summarizer(backend=OpenAIBackend(api_key_env='MY_OPENAI_KEY')).fit()
    assert created['api_key'] == 'sk-123'


def test_missing_credentials_name_the_variable(monkeypatch):
    """Name the variable to set when the credential is missing."""
    install_module(monkeypatch, 'openai', OpenAI=lambda **kwargs: namespace())
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(MissingCredentialError, match='OPENAI_API_KEY'):
        summarizer(backend='openai').fit()


def test_secret_absent_from_parameters_repr_and_pickle(monkeypatch):
    """Keep the credential out of the parameters, the repr and the pickled state."""
    install_module(monkeypatch, 'openai', OpenAI=lambda **kwargs: namespace(secret=kwargs['api_key']))
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-secret')
    backend = summarizer(backend='openai').fit().backend_
    assert 'sk-secret' not in repr(backend.get_params())
    assert 'sk-secret' not in repr(backend)
    restored = round_trip(backend)
    assert not hasattr(restored, '_client')
    assert restored.get_params() == backend.get_params()


def test_clone_shares_injected_client():
    """Share an injected client between clones instead of copying it."""
    client = namespace()
    backend = OpenAIBackend(client=client)
    assert clone(backend).client is client


def test_registered_provider_is_usable_by_string():
    """Use a registered provider name in backend strings."""

    class AcmeBackend(Backend):
        """Generate text by shouting."""

        def generate_text(self, request):
            """Return the text in upper case, prefixed with the model."""
            return f'{self.model}: {request.sample.upper()}'

    register('acme', AcmeBackend)
    assert summarizer(backend='acme:v1').fit().predict(['hi'])[0] == 'v1: HI'
    assert 'acme' in list_backends('generate_text')
    assert 'acme' not in list_backends('transcribe')


def test_register_rejects_non_backend_classes():
    """Reject classes that do not subclass `Backend`."""
    with pytest.raises(TypeError, match='subclass of `Backend`'):
        register('bad', dict)


def test_list_backends_filters_by_predicate():
    """Keep only the backends for which the predicate holds."""
    assert list_backends('synthesize_speech', lambda backend: backend.supports_voice_cloning) == ['dummy', 'replicate']


def test_resolve_model_requires_model_without_default():
    """Ask for the `model` parameter when a task has no default model."""
    with pytest.raises(UnsupportedBackendError, match='Set the `model` parameter'):
        DummyBackend().resolve_model('generate_text')
    assert DummyBackend(model='m').resolve_model('generate_text') == 'm'


def test_clone_shares_objects_in_options():
    """Copy the options dict but share the live objects inside it."""
    client = threading.Lock()
    backend = DummyBackend(options={'client': client, 'temperature': 0})
    cloned = clone(backend)
    assert cloned.options == backend.options
    assert cloned.options is not backend.options
    assert cloned.options['client'] is client


class KindBackend(Backend):
    """Declare default models per task and per input kind."""

    default_models: ClassVar[dict[str, str]] = {'generate_text': 'text-model', 'generate_text:image': 'vision-model'}

    def generate_text(self, request):
        """Return the sample."""
        return str(request.sample)


def test_resolve_model_prefers_input_kind_default():
    """Pick a `'<task>:<input kind>'` default before the task default, and the `model` parameter before both."""
    backend = KindBackend()
    assert backend.resolve_model('generate_text') == 'text-model'
    assert backend.resolve_model('generate_text', 'text') == 'text-model'
    assert backend.resolve_model('generate_text', 'image') == 'vision-model'
    assert backend.resolve_model('generate_text', 'audio') == 'text-model'
    assert KindBackend(model='m').resolve_model('generate_text', 'image') == 'm'


def test_resolve_model_ignores_kind_keys_without_task_default():
    """Ask for the model when only another input kind has a default."""
    backend = KindBackend()
    backend.default_models = {'extract_fields:audio': 'audio-model'}
    assert backend.resolve_model('extract_fields', 'audio') == 'audio-model'
    with pytest.raises(UnsupportedBackendError, match='no default model for `extract_fields`'):
        backend.resolve_model('extract_fields', 'image')


def test_supports_checks_task_and_input_kind():
    """Support the overridden tasks for every input kind and nothing else."""
    assert KindBackend.supports('generate_text')
    assert KindBackend.supports('generate_text', 'audio')
    assert not KindBackend.supports('transcribe')
    assert KindBackend.supported_tasks() == frozenset({'generate_text'})


def test_list_backends_filters_by_input_kind():
    """Leave out prompted backends that do not accept audio for audio samples."""

    class TextOnlyBackend(PromptedBackend):
        """Prompt a model that reads no audio."""

        def complete(self, prompt, model):
            """Answer nothing."""
            return ''

    assert TextOnlyBackend.supports('extract_fields', 'text')
    assert not TextOnlyBackend.supports('extract_fields', 'audio')
    assert 'dummy' in list_backends('extract_fields', input_kind='audio')
    assert 'hf' in list_backends('extract_fields', input_kind='audio')
