"""Test selecting, configuring and registering backends."""

import threading

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
    UnsupportedBackendError,
    list_backends,
    register,
)
from skai.generation import TalkingAvatarGenerator
from skai.text import TextSummarizer
from skai.transcription import SpeechTranscriber
from tests.fakes import install_module, namespace, round_trip


class RecordingBackend(DummyBackend):
    """Record the model that classifies every text."""

    def classify_text(self, request):
        """Return a score of one for the first label when the model is `first`."""
        scores = dict.fromkeys(request.labels, 0.0)
        scores[request.labels[0] if self.model == 'first' else request.labels[-1]] = 1.0
        return scores


def test_backend_string_selects_provider_and_model():
    """Parse `'<provider>:<model>'` into the provider and its model."""
    summarizer = TextSummarizer(backend='dummy:tiny').fit()
    assert isinstance(summarizer.backend_, DummyBackend)
    assert summarizer.backend_.model == 'tiny'


def test_backend_string_uses_provider_default_model(monkeypatch):
    """Use the documented default model of the provider for a provider-only string."""
    install_module(monkeypatch, 'transformers', pipeline=None)
    transcriber = SpeechTranscriber(backend='hf').fit()
    assert transcriber.backend_.resolve_model('transcribe') == HFBackend.default_models['transcribe']


def test_backend_object_is_cloned():
    """Clone a backend object so the component never changes it."""
    backend = DummyBackend(model='m')
    summarizer = TextSummarizer(backend=backend).fit()
    assert summarizer.backend_ is not backend
    assert summarizer.backend_.get_params() == backend.get_params()


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
    assert isinstance(TextSummarizer().fit().backend_, HFBackend)


def test_default_backend_without_extra_names_install_command(monkeypatch):
    """Name the pip command when the extra of the default backend is missing."""
    monkeypatch.setitem(__import__('sys').modules, 'transformers', None)
    with pytest.raises(MissingExtraError, match=r'pip install scikit-ai\[hf\]'):
        TextSummarizer().fit()
    assert issubclass(MissingExtraError, ImportError)


def test_unknown_provider_lists_supported_providers():
    """Reject an unknown provider and list the providers that support the task."""
    with pytest.raises(UnsupportedBackendError, match=r'Unknown backend `nope`.*dummy'):
        TextSummarizer(backend='nope').fit()


def test_provider_without_task_lists_supported_providers():
    """Reject a provider that does not support the task and list those that do."""
    with pytest.raises(UnsupportedBackendError, match=r"OpenAIBackend does not support.*\['dummy', 'replicate'\]"):
        TalkingAvatarGenerator(avatar=np.zeros((4, 4, 3), dtype=np.uint8), backend='openai').fit()
    assert issubclass(UnsupportedBackendError, ValueError)


def test_credentials_are_read_from_named_variable(monkeypatch):
    """Create the client with the value of the variable named by `api_key_env`."""
    created = {}
    install_module(monkeypatch, 'openai', OpenAI=lambda **kwargs: created.update(kwargs) or namespace())
    monkeypatch.setenv('MY_OPENAI_KEY', 'sk-123')
    TextSummarizer(backend=OpenAIBackend(api_key_env='MY_OPENAI_KEY')).fit()
    assert created['api_key'] == 'sk-123'


def test_missing_credentials_name_the_variable(monkeypatch):
    """Name the variable to set when the credential is missing."""
    install_module(monkeypatch, 'openai', OpenAI=lambda **kwargs: namespace())
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(MissingCredentialError, match='OPENAI_API_KEY'):
        TextSummarizer(backend='openai').fit()


def test_secret_absent_from_parameters_repr_and_pickle(monkeypatch):
    """Keep the credential out of the parameters, the repr and the pickled state."""
    install_module(monkeypatch, 'openai', OpenAI=lambda **kwargs: namespace(secret=kwargs['api_key']))
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-secret')
    backend = TextSummarizer(backend='openai').fit().backend_
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
        """Summarize by shouting."""

        def summarize(self, request):
            """Return the text in upper case, prefixed with the model."""
            return f'{self.model}: {request.text.upper()}'

    register('acme', AcmeBackend)
    assert TextSummarizer(backend='acme:v1').fit().predict(['hi'])[0] == 'v1: HI'
    assert 'acme' in list_backends('summarize')
    assert 'acme' not in list_backends('translate')


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
        DummyBackend().resolve_model('summarize')
    assert DummyBackend(model='m').resolve_model('summarize') == 'm'


def test_clone_shares_objects_in_options():
    """Copy the options dict but share the live objects inside it."""
    client = threading.Lock()
    backend = DummyBackend(options={'client': client, 'temperature': 0})
    cloned = clone(backend)
    assert cloned.options == backend.options
    assert cloned.options is not backend.options
    assert cloned.options['client'] is client
