# Backends

The `backend` parameter of every component chooses where the component runs. The interface of the component stays the
same whatever the backend.

## Choosing a backend

`backend` takes a string or a backend object:

- `'<provider>'` uses the default model of the provider for the task, for example `'openai'`.
- `'<provider>:<model>'` uses a specific model, for example `'openai:gpt-5-mini'` or
  `'hf:Salesforce/blip-image-captioning-large'`.
- A backend object from `skai.backends` holds the model and the provider settings, such as the credential variable,
  a preconfigured client, an endpoint and extra call options.
- `None` uses the default backend of the component, which runs an open-source model.

```python exec="true" source="above" result="text"
from skai.backends import DummyBackend
from skai.text import TextTranslator

by_string = TextTranslator(target_language='Greek', backend='dummy:any-model').fit()
by_object = TextTranslator(target_language='Greek', backend=DummyBackend(model='any-model')).fit()
print(by_string.backend_, by_object.backend_)
```

Backend objects are scikit-learn estimators, so their parameters can be tuned like any nested parameter:

```python exec="true" source="above" result="text"
from sklearn.model_selection import GridSearchCV
from skai.backends import DummyBackend
from skai.text import TextClassifier

X = ['good', 'bad', 'great', 'awful'] * 3
y = ['positive', 'negative'] * 6
search = GridSearchCV(TextClassifier(backend=DummyBackend()), {'backend__model': ['model-a', 'model-b']}, cv=2)
print(search.fit(X, y).best_params_)
```

## Providers

--8<-- "README.md:providers"

A component raises `UnsupportedBackendError`, a `ValueError`, at `fit` when its provider is unknown or does not support
its task, and the message lists the providers that do. A missing extra raises `MissingExtraError`, an `ImportError`,
that names the `pip install` command.

## Default models

--8<-- "README.md:defaults"

Local Hugging Face models are downloaded on first use. Replicate models should be pinned to a version, as in
`'replicate:cjwbw/sadtalker:<version>'`, for reproducible results.

Prompted backends (`openai`, `litellm`, and `hf` for summarization, translation and description) score
classification with one for every label the model names and zero for the others. The `hf` backend classifies with
zero-shot models that return real scores, and ignores few-shot examples.

## Credentials

Cloud backends never take a secret as an argument. They take the name of the environment variable that holds it, and
read it when the client is created in `fit`:

| Backend | Parameter | Default variable |
|---|---|---|
| `OpenAIBackend` | `api_key_env` | `OPENAI_API_KEY` |
| `ReplicateBackend` | `api_token_env` | `REPLICATE_API_TOKEN` |
| `LiteLLMBackend` | `api_key_env` | the standard variable of each provider, such as `ANTHROPIC_API_KEY` |

A missing variable raises `MissingCredentialError` naming the variable. `OpenAIBackend` and `ReplicateBackend` also
take a preconfigured `client`, which clones of the backend share. Clients are not pickled, and are created again when
a restored component is used.

## Registering a backend

A backend subclasses `skai.core.Backend` and overrides the methods of the tasks it supports, such as `summarize` or
`generate_image`. A subclass of `skai.core.PromptedBackend` implements `complete` and gets classification,
summarization, translation and image description from it. `skai.core.register` makes the backend available by name.

```python exec="true" source="above" result="text"
from skai.core import Backend, register
from skai.text import TextSummarizer

class FirstSentenceBackend(Backend):
    def summarize(self, request):
        return request.text.split('.')[0] + '.'

register('first-sentence', FirstSentenceBackend)
summarizer = TextSummarizer(backend='first-sentence').fit()
print(summarizer.predict(['It works. Everything else is detail.']))
```

Packages can also declare backends as entry points in the `skai.backends` group, which scikit-ai loads on first use:

```toml
[project.entry-points."skai.backends"]
my-provider = "my_package:MyBackend"
```
